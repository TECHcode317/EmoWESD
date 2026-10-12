import os, gc, glob, time, pickle, json, math, copy, warnings, random
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch, torch.nn as nn, torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix)
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold
from scipy import stats
warnings.filterwarnings('ignore')

CFG = dict(
    ROOT=os.environ.get('WESAD_ROOT', '/kaggle/input/datasets/toriotmd/wesad-ch/WESAD'),
    OUT=os.environ.get('WESAD_OUT', '/kaggle/working'),
    SEED=42, WIN_S=60, STEP_S=5,            
    PURITY_STRESS=0.6, PURITY_NON=0.1,      
    T=8, H1=128, H2=64, TAU=2.0, THR=1.0,   
    LR=1e-3, BS=64, EPOCHS=20,
    CV_EPOCHS=10, ABL_EPOCHS=8, N_FOLDS=5,
    SMOTE_RATIO=1.0,                        
    N_TEST_SUBJ=3, N_VAL_SUBJ=2,           
    RUN_MULTICLASS=True, RUN_ABLATION=True, RUN_CV=True, RUN_MAML=True, RUN_TSNE=True,
    E_MAC=4.6e-12, E_AC=0.9e-12,            
    E_MAC8=0.2e-12, E_AC8=0.03e-12,         
)
RES, FIG = os.path.join(CFG['OUT'], 'results'), os.path.join(CFG['OUT'], 'figures')
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
CACHE = os.path.join(CFG['OUT'], 'wesad_windows_cache.npz')
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:7.1f}s]', *a, flush=True)
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(CFG['SEED']); torch.backends.cudnn.deterministic = True
plt.rcParams.update({'figure.dpi': 110, 'axes.grid': True, 'grid.alpha': .3})
def savefig(name): plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=150, bbox_inches='tight'); plt.close()
log('device:', DEV)


FS = dict(chest=dict(ACC=700, ECG=700, EMG=700, EDA=700, Temp=700, Resp=700),
          wrist=dict(ACC=32, BVP=64, EDA=4, TEMP=4))
LABEL_MAP = {0: 'transient/not defined', 1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'meditation',
             5: 'ignored', 6: 'ignored', 7: 'ignored'}
FEAT = ['EDA_mean', 'EDA_std', 'EDA_min', 'EDA_max', 'EDA_slope', 'BVP_mean', 'BVP_std', 'BVP_mad',
        'BVP_peaks', 'TEMP_mean', 'TEMP_std', 'TEMP_slope', 'ACC_energy_mean', 'ACC_energy_std', 'ACC_energy_max']
GROUPS = dict(EDA=[0, 1, 2, 3, 4], BVP=[5, 6, 7, 8], TEMP=[9, 10, 11], ACC=[12, 13, 14])

def inspect_csvs(root, max_files=30):
    files = sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True))[:max_files]
    rows = []
    for fp in files:
        try:
            n, miss, head, tail, dt, cols = 0, None, None, None, None, None
            for ch in pd.read_csv(fp, chunksize=200000, low_memory=False):
                if head is None: head, dt, cols = ch.head(3), ch.dtypes.astype(str).to_dict(), list(ch.columns)
                n += len(ch); tail = ch.tail(3)
                m = ch.isna().sum(); miss = m if miss is None else miss.add(m, fill_value=0)
            print(f'\n[CSV] {os.path.relpath(fp, root)}  rows={n} cols={len(cols)}\n  columns={cols}\n  dtypes={dt}'
                  f'\n  missing={int(miss.sum())}\n  first3:\n{head.to_string()}\n  last3:\n{tail.to_string()}')
            rows.append(dict(file=os.path.relpath(fp, root), rows=n, cols=len(cols), missing=int(miss.sum())))
        except Exception as e:
            print('[CSV] could not read', fp, e)
    if not files: print('[CSV] no csv files found')
    return pd.DataFrame(rows)

def sensor_stats(sid, loc, name, arr):
    a = np.asarray(arr); fl = a.reshape(len(a), -1)
    fin = np.isfinite(a.astype(np.float64)) if a.dtype.kind in 'fiu' else np.ones(a.shape, bool)
    v = a[fin].astype(np.float64)
    return dict(subject=sid, loc=loc, sensor=name, shape=str(a.shape), n_samples=len(a), n_channels=fl.shape[1],
                dtype=str(a.dtype), fs_hz=FS[loc].get(name, np.nan), n_nan=int(np.isnan(a.astype(np.float64)).sum()),
                n_inf=int(np.isinf(a.astype(np.float64)).sum()), min=float(v.min()), max=float(v.max()),
                mean=float(v.mean()), std=float(v.std()), first_vals=np.round(fl[:3].astype(np.float64), 4).tolist())

def win_view(x, fs):
    return sliding_window_view(x, CFG['WIN_S'] * fs)[::CFG['STEP_S'] * fs]

def slope(x):
    t = np.arange(x.shape[1], dtype=np.float32); t -= t.mean()
    return (x * t).sum(1) / (t ** 2).sum()

def window_features(wrist, lab):
    eda = wrist['EDA'].ravel().astype(np.float32); tmp = wrist['TEMP'].ravel().astype(np.float32)
    bvp = wrist['BVP'].ravel().astype(np.float32)
    acc = np.sqrt((wrist['ACC'].astype(np.float32) ** 2).sum(1))         
    E, Tm, B, A, L = win_view(eda, 4), win_view(tmp, 4), win_view(bvp, 64), win_view(acc, 32), win_view(lab4, 4)
    nw = min(len(E), len(Tm), len(B), len(A), len(L))
    E, Tm, B, A, L = E[:nw], Tm[:nw], B[:nw], A[:nw], L[:nw]
    mu, sd = B.mean(1, keepdims=True), B.std(1, keepdims=True)
    d = np.diff(B, axis=1)
    peaks = ((d[:, :-1] > 0) & (d[:, 1:] <= 0) & (B[:, 1:-1] > mu + 0.5 * sd)).sum(1)  
    X = np.stack([E.mean(1), E.std(1), E.min(1), E.max(1), slope(E), B.mean(1), B.std(1), np.abs(d).mean(1),
                  peaks, Tm.mean(1), Tm.std(1), slope(Tm), A.mean(1), A.std(1), A.max(1)], 1).astype(np.float32)
    X = np.nan_to_num(X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)                  
    frac = np.stack([(L == k).mean(1) for k in (1, 2, 3, 4)], 1).astype(np.float32)
    return X, frac

def build_dataset():
    pk = sorted(glob.glob(os.path.join(CFG['ROOT'], '**', 'S*.pkl'), recursive=True))
    if not pk:
        print('No .pkl found. Top level of ROOT:', os.listdir(CFG['ROOT']) if os.path.exists(CFG['ROOT']) else 'ROOT missing')
        raise FileNotFoundError('WESAD subject pkl files not found - check CFG["ROOT"]')
    log(f'found {len(pk)} subject files'); inv, lab_rows, Xs, Fs, Ss, raw = [], [], [], [], [], {}
    for fp in pk:
        with open(fp, 'rb') as f: d = pickle.load(f, encoding='latin1')
        sid = str(d.get('subject', os.path.basename(fp)[:-4])); sig = d['signal']
        lab = np.asarray(d['label']).ravel().astype(np.int64)
        for loc in ('chest', 'wrist'):
            for name, arr in sig[loc].items():
                r = sensor_stats(sid, loc, name, arr); inv.append(r)
                print(f"{sid:>4} {loc:5} {name:5} shape={r['shape']:>12} n={r['n_samples']:>8} ch={r['n_channels']} "
                      f"{r['dtype']:8} fs={r['fs_hz']} nan={r['n_nan']} inf={r['n_inf']} min={r['min']:.3g} max={r['max']:.3g} "
                      f"mean={r['mean']:.3g} std={r['std']:.3g} first={r['first_vals']}")
        u, c = np.unique(lab, return_counts=True)
        for k, n in zip(u, c):
            lab_rows.append(dict(subject=sid, label=int(k), meaning=LABEL_MAP.get(int(k), '?'), samples=int(n), pct=100 * n / len(lab)))
        print(f'{sid:>4} labels: n={len(lab)} unique={u.tolist()} counts={c.tolist()} pct={np.round(100*c/len(lab),2).tolist()}')
        wrist = {k: np.asarray(v) for k, v in sig['wrist'].items()}
        X, fr = window_features(wrist, lab); Xs.append(X); Fs.append(fr); Ss += [sid] * len(X)
        raw[sid] = (wrist['EDA'].ravel().astype(np.float32), lab[::175].astype(np.int8))
        del d, sig, wrist, lab; gc.collect()
    pd.DataFrame(inv).to_csv(os.path.join(RES, 'sensor_inventory.csv'), index=False)
    pd.DataFrame(lab_rows).to_csv(os.path.join(RES, 'label_inventory.csv'), index=False)
    arrs = dict(X=np.concatenate(Xs), frac=np.concatenate(Fs), sid=np.array(Ss))
    for s, (e, l) in raw.items(): arrs[f'eda_{s}'] = e; arrs[f'lab_{s}'] = l
    np.savez_compressed(CACHE, **arrs); return arrs

inspect_csvs(CFG['ROOT']).to_csv(os.path.join(RES, 'csv_inventory.csv'), index=False)
if os.path.exists(CACHE):
    log('loading cached windows (delete the cache file to re-inspect raw pkl files)'); D = dict(np.load(CACHE, allow_pickle=False))
else:
    D = build_dataset()
X_all, FR, SID = D['X'].astype(np.float32), D['frac'], D['sid'].astype(str)
valid = FR.sum(1) >= 0.9
ybin = np.full(len(X_all), -1)
ybin[valid & (FR[:, 1] >= CFG['PURITY_STRESS'])] = 1
ybin[valid & (FR[:, 1] <= CFG['PURITY_NON'])] = 0
kb = ybin >= 0
Xb, yb, sb = X_all[kb], ybin[kb], SID[kb]
subjects = sorted(set(map(str, sb)), key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)
log(f'binary windows={len(yb)} stress={int(yb.sum())} ({100*yb.mean():.1f}%)  subjects={subjects}')

rng = np.random.RandomState(CFG['SEED']); perm = rng.permutation(subjects)
perm = [str(x) for x in perm]; TEST_S = perm[:CFG['N_TEST_SUBJ']]; VAL_S = perm[CFG['N_TEST_SUBJ']:CFG['N_TEST_SUBJ'] + CFG['N_VAL_SUBJ']]
TRAIN_S = [s for s in subjects if s not in TEST_S + VAL_S]
log('train subj', TRAIN_S, '| val', VAL_S, '| test', TEST_S)
def split(X, y, s, names): m = np.isin(s, names); return X[m], y[m], s[m]
Xtr0, ytr0, str0 = split(Xb, yb, sb, TRAIN_S); Xva, yva, sva = split(Xb, yb, sb, VAL_S); Xte, yte, ste = split(Xb, yb, sb, TEST_S)

def smote(X, y, ratio=1.0, k=5, seed=0):
    r = np.random.RandomState(seed); cl, cnt = np.unique(y, return_counts=True); maj = cnt.max(); Xo, yo = [X], [y]
    for c, n in zip(cl, cnt):
        need = int(maj * ratio) - n
        if need <= 0 or n < 3: continue
        Xc = X[y == c]; kk = min(k, len(Xc) - 1)
        nb = NearestNeighbors(n_neighbors=kk + 1).fit(Xc).kneighbors(Xc, return_distance=False)[:, 1:]
        base = r.randint(0, len(Xc), need); nn_ = nb[base, r.randint(0, kk, need)]
        Xo.append(Xc[base] + r.rand(need, 1) * (Xc[nn_] - Xc[base])); yo.append(np.full(need, c))
    return np.concatenate(Xo).astype(np.float32), np.concatenate(yo)
Xtr, ytr = smote(Xtr0, ytr0, CFG['SMOTE_RATIO'], seed=CFG['SEED'])        
log(f'train {len(ytr0)} -> {len(ytr)} after SMOTE | val {len(yva)} | test {len(yte)} (natural class ratio, no SMOTE)')

class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ctx.save_for_backward(x); return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, g): x, = ctx.saved_tensors; return g / (1 + 25 * x.abs()) ** 2   
spike = SpikeFn.apply

class SpikingLayer(nn.Module):
    def __init__(s, din, dout, neuron, tau, thr, learn=False):
        super().__init__(); s.fc = nn.Linear(din, dout); s.learn = learn
        beta = 1.0 if neuron == 'IF' else 1.0 - 1.0 / tau       
            s.wb = nn.Parameter(torch.full((dout,), math.log(beta / (1 - beta + 1e-6) + 1e-6)))
            s.th = nn.Parameter(torch.full((dout,), float(thr)))
        else:
            s.register_buffer('b', torch.tensor(float(beta))); s.register_buffer('th', torch.tensor(float(thr)))
    def forward(s, x):                                          
        cur = s.fc(x); beta = torch.sigmoid(s.wb) if s.learn else s.b
        m = torch.zeros_like(cur[:, 0]); out = []
        for t in range(cur.shape[1]):
            m = beta * m + cur[:, t]; sp = spike(m - s.th); out.append(sp)
            m = m * (1 - sp.detach())                          
        return torch.stack(out, 1)

class SNN(nn.Module):
    def __init__(s, din, nout, neuron='LIF', T=8, enc='direct', learn=False):
        super().__init__(); s.T, s.enc, s.neuron, s.din = T, enc, neuron, din
        s.l1 = SpikingLayer(din, CFG['H1'], neuron, CFG['TAU'], CFG['THR'], learn)
        s.l2 = SpikingLayer(CFG['H1'], CFG['H2'], neuron, CFG['TAU'], CFG['THR'], learn)
        s.out = nn.Linear(CFG['H2'], nout)
    def encode(s, x):
        if s.enc == 'direct': return x.unsqueeze(1).expand(-1, s.T, -1)           
        return (torch.rand(x.shape[0], s.T, x.shape[1], device=x.device) < torch.sigmoid(x).unsqueeze(1)).float() 
    def spikes(s, x): z = s.encode(x); s1 = s.l1(z); s2 = s.l2(s1); return z, s1, s2
    def forward(s, x): return s.out(s.spikes(x)[2]).mean(1)                      

class MLP(nn.Module):
    def __init__(s, din, nout):
        super().__init__(); s.net = nn.Sequential(nn.Linear(din, CFG['H1']), nn.ReLU(), nn.Linear(CFG['H1'], CFG['H2']), nn.ReLU(), nn.Linear(CFG['H2'], nout))
    def forward(s, x): return s.net(x)

class CNNBiLSTM(nn.Module):
    def __init__(s, din, nout):
        super().__init__(); s.c = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 32, 3, padding=1), nn.ReLU())
        s.r = nn.LSTM(32, 32, batch_first=True, bidirectional=True); s.f = nn.Linear(64, nout)
    def forward(s, x): h = s.c(x.unsqueeze(1)).transpose(1, 2); o, _ = s.r(h); return s.f(o.mean(1))

class TransformerEdge(nn.Module):
    def __init__(s, din, nout, d=32):
        super().__init__(); s.e = nn.Linear(1, d); s.p = nn.Parameter(torch.zeros(1, din, d))
        s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 64, 0.1, batch_first=True), 2); s.f = nn.Linear(d, nout)
    def forward(s, x): return s.f(s.t(s.e(x.unsqueeze(-1)) + s.p).mean(1))

def make(kind, din, nout, **kw):
    if kind in ('IF', 'LIF'): return SNN(din, nout, kind, kw.get('T', CFG['T']), kw.get('enc', 'direct'), kw.get('learn', False))
    return dict(MLP=MLP, CNNBiLSTM=CNNBiLSTM, Transformer=TransformerEdge)[kind](din, nout)
nparams = lambda m: sum(p.numel() for p in m.parameters())


def metrics(y, p):
    pred = p.argmax(1); C = p.shape[1]
    try:
        if C == 2: f1, auc = f1_score(y, pred), roc_auc_score(y, p[:, 1])
        else: f1, auc = f1_score(y, pred, average='macro'), roc_auc_score(y, p, multi_class='ovr', average='macro')
    except Exception: f1, auc = f1_score(y, pred, average='macro'), float('nan')
    return dict(acc=accuracy_score(y, pred), f1=f1, auc=auc)

@torch.no_grad()
def predict(model, X, bs=4096):
    model.eval(); X = torch.as_tensor(X, dtype=torch.float32)
    p = torch.cat([F.softmax(model(X[i:i + bs].to(DEV)).float(), 1).cpu() for i in range(0, len(X), bs)]).numpy().astype(np.float64)
    return p / p.sum(1, keepdims=True)

def train(model, Xt, yt, Xv, yv, epochs, seed=0, lr=None, bs=None, verbose=True):
    seed_all(seed); model.to(DEV); lr = lr or CFG['LR']; bs = bs or CFG['BS']
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xg, yg = torch.as_tensor(Xt, dtype=torch.float32, device=DEV), torch.as_tensor(yt, dtype=torch.long, device=DEV)
    sub = np.random.RandomState(0).choice(len(Xt), min(6000, len(Xt)), replace=False)
    H = dict(loss=[], tr_acc=[], tr_f1=[], va_acc=[], va_f1=[]); best, bf = None, -1
    for ep in range(epochs):
        model.train(); pm = torch.randperm(len(Xg), device=DEV); tot = torch.zeros((), device=DEV)
        for i in range(0, len(Xg), bs):
            ix = pm[i:i + bs]; opt.zero_grad(); loss = F.cross_entropy(model(Xg[ix]), yg[ix]); loss.backward(); opt.step(); tot += loss.detach() * len(ix)
        mt, mv = metrics(yt[sub], predict(model, Xt[sub])), metrics(yv, predict(model, Xv))
        for k, v in zip(H, [tot.item() / len(Xg), mt['acc'], mt['f1'], mv['acc'], mv['f1']]): H[k].append(v)
        if mv['f1'] > bf: bf, best = mv['f1'], copy.deepcopy(model.state_dict())
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            log(f'  ep{ep+1:02d} loss={H["loss"][-1]:.4f} train_acc={mt["acc"]:.4f} val_acc={mv["acc"]:.4f} val_f1={mv["f1"]:.4f}')
    model.load_state_dict(best); return model, H          

def boot_ci(y, p, n=1000, seed=0):
    c = (p.argmax(1) == y).astype(float); r = np.random.RandomState(seed)
    m = [c[r.randint(0, len(c), len(c))].mean() for _ in range(n)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


MODELS, HIST, PROB = {}, {}, {}
din = Xtr.shape[1]
for name, kind, kw in [('SNN-IF', 'IF', {}), ('SNN-LIF', 'LIF', {}), ('SNN-PLIF (learn tau,thr)', 'LIF', dict(learn=True)),
                       ('MLP-ANN', 'MLP', {}), ('CNN-BiLSTM', 'CNNBiLSTM', {}), ('Transformer-Edge', 'Transformer', {})]:
    log('training', name); m, h = train(make(kind, din, 2, **kw), Xtr, ytr, Xva, yva, CFG['EPOCHS'])
    MODELS[name], HIST[name], PROB[name] = m, h, predict(m, Xte)
sv_idx = np.random.RandomState(0).choice(len(Xtr), min(12000, len(Xtr)), replace=False)
log('training SVM-RBF (12k subsample for speed)'); svm = SVC(kernel='rbf', probability=True, random_state=0).fit(Xtr[sv_idx], ytr[sv_idx])
PROB['SVM-RBF'] = svm.predict_proba(Xte)

rows = []
for n, p in PROB.items():
    mt = metrics(yte, p); lo, hi = boot_ci(yte, p)
    rows.append(dict(model=n, accuracy=100 * mt['acc'], acc_CI95=f'[{100*lo:.2f}, {100*hi:.2f}]', f1=mt['f1'], auc=mt['auc'],
                     params=nparams(MODELS[n]) if n in MODELS else np.nan))
main_df = pd.DataFrame(rows); main_df.to_csv(os.path.join(RES, 'main_results.csv'), index=False)
print('\n=== MAIN RESULTS (held-out unseen subjects, natural class ratio) ===\n', main_df.round(4).to_string(index=False))

ps = []
for s in TEST_S:
    for n, p in PROB.items():
        mk = ste == s; mt = metrics(yte[mk], p[mk]); ps.append(dict(subject=s, model=n, n=int(mk.sum()), acc=mt['acc'], f1=mt['f1']))
pd.DataFrame(ps).to_csv(os.path.join(RES, 'per_subject_test.csv'), index=False)
print('\n=== PER-SUBJECT TEST ACCURACY ===\n', pd.DataFrame(ps).pivot(index='subject', columns='model', values='acc').round(4).to_string())

def mcnemar(y, p1, p2):
    c1, c2 = p1.argmax(1) == y, p2.argmax(1) == y; b, c = int((c1 & ~c2).sum()), int((~c1 & c2).sum())
    return b, c, (1.0 if b + c == 0 else stats.binomtest(b, b + c, 0.5).pvalue)
sig = {}
for a, b_ in [('SNN-IF', 'SNN-LIF'), ('SNN-IF', 'MLP-ANN'), ('SNN-LIF', 'MLP-ANN'), ('SNN-IF', 'Transformer-Edge')]:
    b, c, pv = mcnemar(yte, PROB[a], PROB[b_]); sig[f'{a} vs {b_}'] = dict(only_first_correct=b, only_second_correct=c, p=pv)
print('\n=== McNemar exact tests ===\n', json.dumps(sig, indent=1))


@torch.no_grad()
def snn_ops(model, X, n=3000, e_mac=CFG['E_MAC'], e_ac=CFG['E_AC']):
    model.eval(); x = torch.as_tensor(X[:n], dtype=torch.float32, device=DEV); z, s1, s2 = model.spikes(x)
    T, d, h1, h2, no = model.T, model.din, CFG['H1'], CFG['H2'], model.out.out_features
    in_spk = z.sum((1, 2)).mean().item(); a1, a2 = s1.sum((1, 2)).mean().item(), s2.sum((1, 2)).mean().item()
    mac = (T * d * h1) if model.enc == 'direct' else 0                          
    ac = (0 if model.enc == 'direct' else in_spk * h1) + a1 * h2 + a2 * no
    upd = T * (h1 + h2); mac += upd if model.neuron == 'LIF' else 0; ac += upd if model.neuron == 'IF' else 0   
    return dict(MACs=mac, ACs=ac, SynOps=mac + ac, rate_L1=s1.mean().item(), rate_L2=s2.mean().item(),
                dense_FLOPs_eq23=2 * T * (d * h1 + h1 * h2 + h2 * no), energy_J=mac * e_mac + ac * e_ac,
                energy_J_int8=mac * CFG['E_MAC8'] + ac * CFG['E_AC8'])
@torch.no_grad()
def latency_ms(model, X, n=300):
    m = copy.deepcopy(model).cpu().eval(); xs = torch.as_tensor(X[:n + 20], dtype=torch.float32); ts = []
    for i in range(n + 20):
        t = time.perf_counter(); m(xs[i:i + 1]); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.mean(ts[20:])), float(np.std(ts[20:]))
eff = []
mlp_mac = din * CFG['H1'] + CFG['H1'] * CFG['H2'] + CFG['H2'] * 2
for n in ['SNN-IF', 'SNN-LIF', 'SNN-PLIF (learn tau,thr)']:
    o = snn_ops(MODELS[n], Xte); l = latency_ms(MODELS[n], Xte); eff.append(dict(model=n, params=nparams(MODELS[n]), lat_ms=l[0], lat_std=l[1], **o))
l = latency_ms(MODELS['MLP-ANN'], Xte)
eff.append(dict(model='MLP-ANN', params=nparams(MODELS['MLP-ANN']), lat_ms=l[0], lat_std=l[1], MACs=mlp_mac, ACs=0, SynOps=mlp_mac,
                dense_FLOPs_eq23=2 * mlp_mac, energy_J=mlp_mac * CFG['E_MAC'], energy_J_int8=mlp_mac * CFG['E_MAC8']))
for n in ['CNN-BiLSTM', 'Transformer-Edge']:
    l = latency_ms(MODELS[n], Xte); eff.append(dict(model=n, params=nparams(MODELS[n]), lat_ms=l[0], lat_std=l[1]))
t = time.perf_counter(); [svm.predict(Xte[i:i + 1]) for i in range(100)]; eff.append(dict(model='SVM-RBF', lat_ms=(time.perf_counter() - t) * 10))
eff_df = pd.DataFrame(eff); eff_df.to_csv(os.path.join(RES, 'efficiency.csv'), index=False)
print('\n=== EFFICIENCY (energy = analytical op-count model, latency = CPU batch-1 wall clock; NOT hardware power) ===\n', eff_df.round(8).to_string(index=False))


cv_rows = []
if CFG['RUN_CV']:
    gkf = GroupKFold(n_splits=CFG['N_FOLDS'])
    for k, (tri, tei) in enumerate(gkf.split(Xb, yb, sb)):
        trs = np.unique(sb[tri]); vs = trs[:2]; trn = np.setdiff1d(trs, vs)            
        mtr, mva = np.isin(sb, trn), np.isin(sb, vs); Xf, yf = smote(Xb[mtr], yb[mtr], CFG['SMOTE_RATIO'], seed=k)
        for n, kind in [('SNN-IF', 'IF'), ('SNN-LIF', 'LIF'), ('MLP-ANN', 'MLP')]:
            m, _ = train(make(kind, din, 2), Xf, yf, Xb[mva], yb[mva], CFG['CV_EPOCHS'], seed=k, verbose=False)
            mt = metrics(yb[tei], predict(m, Xb[tei])); cv_rows.append(dict(fold=k + 1, model=n, test_subjects=','.join(np.unique(sb[tei])), **mt))
        log(f'CV fold {k+1}/{CFG["N_FOLDS"]} done')
    cv_df = pd.DataFrame(cv_rows); cv_df.to_csv(os.path.join(RES, 'cv_results.csv'), index=False)
    print('\n=== SUBJECT-WISE CV (mean +- std over folds) ===\n', cv_df.groupby('model')[['acc', 'f1', 'auc']].agg(['mean', 'std']).round(4).to_string())


def ptq(model, bits):
    m = copy.deepcopy(model); q = 2 ** (bits - 1) - 1
    for n_, p in m.named_parameters():
        if p.dim() >= 2: sc = p.detach().abs().max() / q; p.data = torch.round(p.data / sc).clamp(-q, q) * sc
    return m
abl = []
if CFG['RUN_ABLATION']:
    full = list(range(din))
    cfgs = [('IF full', 'IF', full, {}, True), ('LIF full', 'LIF', full, {}, True), ('IF no SMOTE', 'IF', full, {}, False),
            ('IF rate-coding', 'IF', full, dict(enc='rate'), True), ('LIF rate-coding', 'LIF', full, dict(enc='rate'), True),
            ('IF T=4', 'IF', full, dict(T=4), True), ('IF T=16', 'IF', full, dict(T=16), True),
            ('IF EDA only', 'IF', GROUPS['EDA'], {}, True), ('IF BVP only', 'IF', GROUPS['BVP'], {}, True),
            ('IF TEMP only', 'IF', GROUPS['TEMP'], {}, True), ('IF without ACC', 'IF', GROUPS['EDA'] + GROUPS['BVP'] + GROUPS['TEMP'], {}, True),
            ('IF without EDA', 'IF', GROUPS['BVP'] + GROUPS['TEMP'] + GROUPS['ACC'], {}, True)]
    for name, kind, idx, kw, use_smote in cfgs:
        a, b = (Xtr, ytr) if use_smote else (Xtr0, ytr0)
        m, _ = train(make(kind, len(idx), 2, **kw), a[:, idx], b, Xva[:, idx], yva, CFG['ABL_EPOCHS'], verbose=False)
        mt = metrics(yte, predict(m, Xte[:, idx])); abl.append(dict(experiment=name, **mt)); log(f'ablation {name}: acc={mt["acc"]:.4f}')
    for n in ['SNN-IF', 'SNN-LIF']:
        for bits in (8, 4):
            mt = metrics(yte, predict(ptq(MODELS[n], bits), Xte)); abl.append(dict(experiment=f'{n} {bits}-bit weights (PTQ)', **mt))
    abl_df = pd.DataFrame(abl); abl_df.to_csv(os.path.join(RES, 'ablation.csv'), index=False)
    print('\n=== ABLATION (test, unseen subjects) ===\n', abl_df.round(4).to_string(index=False))


def adapt(model, Xs, ys, steps=5, lr=0.01):
    m = copy.deepcopy(model); o = torch.optim.SGD(m.parameters(), lr=lr); m.train()
    for _ in range(steps): o.zero_grad(); F.cross_entropy(m(Xs), ys).backward(); o.step()
    return m.eval()
def fomaml(base, X, y, s, iters=150, meta_bs=4, k_sup=5, k_q=32, inner=3, in_lr=0.01, meta_lr=1e-3, seed=0):
    r = np.random.RandomState(seed); meta = copy.deepcopy(base).to(DEV); opt = torch.optim.Adam(meta.parameters(), meta_lr)
    Xg, yg = torch.as_tensor(X, dtype=torch.float32, device=DEV), torch.as_tensor(y, dtype=torch.long, device=DEV)
    ids = {u: [np.where((s == u) & (y == c))[0] for c in (0, 1)] for u in np.unique(s)}
    ids = {u: v for u, v in ids.items() if min(len(v[0]), len(v[1])) > 0}
    for it in range(iters):
        grads = [torch.zeros_like(p) for p in meta.parameters()]
        for u in r.choice(list(ids), min(meta_bs, len(ids)), replace=False):
            sup = np.concatenate([r.choice(ids[u][c], k_sup) for c in (0, 1)]); q = r.choice(np.concatenate(ids[u]), k_q)
            fast = copy.deepcopy(meta); fo = torch.optim.SGD(fast.parameters(), lr=in_lr); fast.train()
            for _ in range(inner): fo.zero_grad(); F.cross_entropy(fast(Xg[sup]), yg[sup]).backward(); fo.step()
            fast.zero_grad(); F.cross_entropy(fast(Xg[q]), yg[q]).backward()
            for g, p in zip(grads, fast.parameters()): g += p.grad / meta_bs
        for p, g in zip(meta.parameters(), grads): p.grad = g
        opt.step()
    return meta
def personalise_eval(model, X, y, s, k=5, draws=10, seed=0):
    r = np.random.RandomState(seed); out = []
    for u in np.unique(s):
        i0, i1 = np.where((s == u) & (y == 0))[0], np.where((s == u) & (y == 1))[0]
        if min(len(i0), len(i1)) <= k: continue
        for d in range(draws):
            sup = np.concatenate([r.choice(i0, k, replace=False), r.choice(i1, k, replace=False)])
            q = np.setdiff1d(np.where(s == u)[0], sup); xs = torch.as_tensor(X[sup], dtype=torch.float32, device=DEV)
            ys = torch.as_tensor(y[sup], dtype=torch.long, device=DEV); b = metrics(y[q], predict(model, X[q])); a = metrics(y[q], predict(adapt(model, xs, ys), X[q]))
            out.append(dict(subject=u, draw=d, acc_before=b['acc'], f1_before=b['f1'], acc_after=a['acc'], f1_after=a['f1']))
    return pd.DataFrame(out)
pers = pd.DataFrame()
if CFG['RUN_MAML']:
    Xreal, yreal, sreal = Xtr0, ytr0, str0           
    prow = []
    for n in ['SNN-IF', 'SNN-LIF']:
        log('MAML meta-training', n); meta = fomaml(MODELS[n], Xreal, yreal, sreal)
        for tag, mdl in [('supervised init + 5-shot fine-tune', MODELS[n]), ('MAML init + 5-shot adapt', meta)]:
            e = personalise_eval(mdl, Xte, yte, ste); e['model'], e['init'] = n, tag; prow.append(e)
    pers = pd.concat(prow); pers.to_csv(os.path.join(RES, 'personalization_raw.csv'), index=False)
    summ = pers.groupby(['model', 'init'])[['acc_before', 'acc_after', 'f1_before', 'f1_after']].agg(['mean', 'std']).round(4)
    print('\n=== PERSONALISATION on unseen subjects (10 labelled windows = 5/class) ===\n', summ.to_string())
    for (n, ini), g in pers.groupby(['model', 'init']):
        dlt = g['f1_after'] - g['f1_before']
        try: pv = stats.wilcoxon(dlt).pvalue
        except Exception: pv = float('nan')
        print(f'  {n:8} {ini:38} dF1 = {dlt.mean():+.4f} +- {dlt.std():.4f}  (Wilcoxon p={pv:.3g})')


mc_cm = {}
if CFG['RUN_MULTICLASS']:
    pur = FR[:, :3].max(1) >= 0.9; kc = pur & (FR[:, :3].sum(1) >= 0.9); ym = FR[:, :3].argmax(1)
    Xm, ymm, sm = X_all[kc], ym[kc], SID[kc]
    a, b_, c_ = [split(Xm, ymm, sm, g) for g in (TRAIN_S, VAL_S, TEST_S)]
    Xs_, ys_ = smote(a[0], a[1], 1.0, seed=1); mrows = []
    for n, kind in [('SNN-IF', 'IF'), ('SNN-LIF', 'LIF'), ('MLP-ANN', 'MLP')]:
        m, _ = train(make(kind, din, 3), Xs_, ys_, b_[0], b_[1], CFG['ABL_EPOCHS'], verbose=False); p = predict(m, c_[0])
        mrows.append(dict(model=n, **metrics(c_[1], p))); mc_cm[n] = confusion_matrix(c_[1], p.argmax(1))
    mc_df = pd.DataFrame(mrows); mc_df.to_csv(os.path.join(RES, 'multiclass_3class.csv'), index=False)
    print('\n=== 3-CLASS (baseline/stress/amusement), macro-F1 ===\n', mc_df.round(4).to_string(index=False))


def flow(boxes, name, horizontal=False, title=''):
    n = len(boxes); fig, ax = plt.subplots(figsize=(13, 3.6) if horizontal else (5.5, 1.0 * n + 1)); ax.axis('off'); ax.set_title(title)
    for i, t in enumerate(boxes):
        x, y = (i * 2.1, 0) if horizontal else (0, -i * 1.5); w, h = (1.85, 1.1) if horizontal else (5, 1.0)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05', fc='#e3f0ff', ec='#4a78b0')); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=8)
        if i < n - 1:
            if horizontal: ax.annotate('', (x + 2.1, y + h / 2), (x + w + .05, y + h / 2), arrowprops=dict(arrowstyle='->'))
            else: ax.annotate('', (x + w / 2, y - 0.5), (x + w / 2, y - .05), arrowprops=dict(arrowstyle='->'))
    ax.set_xlim(-.3, (n * 2.1 if horizontal else 5.3)); ax.set_ylim((-.3, 1.5) if horizontal else (-1.5 * n + .3, 1.3)); savefig(name)
flow(['Raw wrist signals\nEDA 4Hz | BVP 64Hz\nTEMP 4Hz | ACC 32Hz', f'{CFG["WIN_S"]}-s sliding windows\nstep {CFG["STEP_S"]}s, label purity rule',
      '15 statistical features\n(EDA/BVP/TEMP/ACC energy)', 'Per-subject z-score\n(label-free)', 'Subject-wise split\ntrain / val / test',
      'SMOTE on TRAIN only', 'Spike encoding\n(direct / rate)', 'IF / LIF SNN\n128-64-2, T=8', 'Eval: acc/F1/AUC\nSynOps, energy, latency'],
     'Fig1_preprocessing_pipeline.png', title='Fig.1 Data pre-processing pipeline (as implemented)')
flow(['Wearable input\n(wrist)', 'Feature\nextraction', 'Spiking NN\n(IF / LIF)', 'MAML (first-order)\nfew-shot personalisation', 'Op-count energy\n+ latency', 'Stress /\nNon-stress'],
     'Fig2_architecture.png', horizontal=True, title='Fig.2 Proposed lightweight SNN framework')

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a_, n in zip(ax, ['SNN-LIF', 'SNN-IF']):
    cm = confusion_matrix(yte, PROB[n].argmax(1)); a_.imshow(cm, cmap='Blues'); a_.set_title(f'{n} confusion matrix (n={len(yte)})')
    for i in range(2):
        for j in range(2): a_.text(j, i, cm[i, j], ha='center', va='center', color='k', fontsize=12)
    a_.set_xticks([0, 1], ['Non-stress', 'Stress']); a_.set_yticks([0, 1], ['Non-stress', 'Stress']); a_.set_xlabel('Predicted'); a_.set_ylabel('Actual'); a_.grid(False)
savefig('Fig3_confusion_matrices.png')

s0 = TEST_S[0]; eda, l4 = D[f'eda_{s0}'], D[f'lab_{s0}']
fig, ax = plt.subplots(2, 2, figsize=(12, 6))
for a_, k, t in [(ax[0, 0], 1, 'Non-stress (baseline) EDA'), (ax[0, 1], 2, 'Stress EDA')]:
    seg = eda[:len(l4)][l4 == k][:3000]; a_.plot(seg, lw=.8); a_.set_title(f'{t} - subject {s0} (raw, 4 Hz)'); a_.set_xlabel('Samples'); a_.set_ylabel('muS')
for n, c in [('SNN-LIF', 'C0'), ('SNN-IF', 'C1')]: ax[1, 0].plot(range(1, len(HIST[n]['loss']) + 1), HIST[n]['loss'], c, marker='o', ms=3, label=n)
ax[1, 0].set_title('Training loss'); ax[1, 0].set_xlabel('Epoch'); ax[1, 0].legend()
for n, c in [('SNN-LIF', 'C0'), ('SNN-IF', 'C1')]: ax[1, 1].plot(HIST[n]['va_acc'], c, marker='o', ms=3, label=f'{n} val acc'); ax[1, 1].plot(HIST[n]['va_f1'], c + '--', label=f'{n} val F1')
ax[1, 1].set_title('Validation (unseen subjects)'); ax[1, 1].set_xlabel('Epoch'); ax[1, 1].legend(fontsize=7); savefig('Fig4_waveforms_and_training.png')

E = eff_df.set_index('model').loc[['SNN-LIF', 'SNN-IF', 'MLP-ANN']]; M = np.log10(E[['lat_ms', 'energy_J']].values.astype(float))
Mn = (M - M.min(0)) / (M.max(0) - M.min(0) + 1e-12); fig, ax = plt.subplots(figsize=(5, 3.8)); ax.imshow(Mn, cmap='viridis'); ax.grid(False)
for i in range(3):
    ax.text(0, i, f'{E.lat_ms.iloc[i]:.4g}', ha='center', va='center', color='w'); ax.text(1, i, f'{E.energy_J.iloc[i]:.3g}', ha='center', va='center', color='w')
ax.set_xticks([0, 1], ['Latency (ms)', 'Energy/inference (J, est.)']); ax.set_yticks(range(3), E.index); ax.set_title('Latency vs energy (colour = log-scaled, column-normalised)'); savefig('Fig5_latency_energy_heatmap.png')

fig, ax = plt.subplots(figsize=(5.5, 5.2))
for n in ['SNN-LIF', 'SNN-IF', 'MLP-ANN', 'SVM-RBF']:
    f_, t_, _ = roc_curve(yte, PROB[n][:, 1]); ax.plot(f_, t_, label=f'{n} (AUC={roc_auc_score(yte, PROB[n][:, 1]):.3f})')
ax.plot([0, 1], [0, 1], 'k--'); ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.set_title('ROC (unseen subjects)'); ax.legend(); savefig('Fig6_ROC.png')

@torch.no_grad()
def raster(n, cls):
    i = np.where(yte == cls)[0][0]; _, s1, _ = MODELS[n].eval().spikes(torch.as_tensor(Xte[i:i + 1], dtype=torch.float32, device=DEV)); return s1[0].cpu().numpy()
fig, ax = plt.subplots(2, 2, figsize=(11, 6))
for r_, n in enumerate(['SNN-LIF', 'SNN-IF']):
    for c_, cls in enumerate([0, 1]):
        sp = raster(n, cls); tt, nn_ = np.where(sp > 0); ax[r_, c_].scatter(tt, nn_, s=6, c='r' if r_ == 0 else 'b')
        ax[r_, c_].set_title(f'{n} hidden-1 raster - {"stress" if cls else "non-stress"} (rate={sp.mean():.3f})'); ax[r_, c_].set_xlabel('Time step'); ax[r_, c_].set_ylabel('Neuron')
savefig('Fig7_spike_raster.png')

if CFG['RUN_TSNE']:
    @torch.no_grad()
    def feats(n):
        ix = np.random.RandomState(0).choice(len(Xte), min(2500, len(Xte)), replace=False); _, _, s2 = MODELS[n].eval().spikes(torch.as_tensor(Xte[ix], dtype=torch.float32, device=DEV))
        return TSNE(2, init='pca', perplexity=30, random_state=0).fit_transform(s2.sum(1).cpu().numpy()), yte[ix]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a_, n in zip(ax, ['SNN-LIF', 'SNN-IF']):
        z, yy = feats(n); [a_.scatter(z[yy == k, 0], z[yy == k, 1], s=5, alpha=.6, label=l) for k, l in [(0, 'Non-stress'), (1, 'Stress')]]; a_.set_title(f't-SNE of layer-2 spike counts - {n}'); a_.legend()
    savefig('Fig8_tSNE.png')

fig, ax = plt.subplots(1, 2, figsize=(12, 4)); ep = np.arange(1, CFG['EPOCHS'] + 1)
for n, c in [('SNN-LIF', 'C0'), ('SNN-IF', 'C1')]:
    for a_, k, t in [(ax[0], 'acc', 'Accuracy'), (ax[1], 'f1', 'F1-score')]:
        a_.plot(ep, HIST[n][f'tr_{k}'], c + '--', marker='.', label=f'{n} train'); a_.plot(ep, HIST[n][f'va_{k}'], c, marker='o', ms=3, label=f'{n} val (unseen subj.)'); a_.set_title(f'{t} per epoch (measured, unsmoothed)'); a_.set_xlabel('Epoch')
ax[0].legend(fontsize=7); savefig('Fig9_accuracy_F1_curves.png')

def zs(x): return (x - x.mean()) / (x.std() + 1e-8)
@torch.no_grad()
def dens(n, cls):
    ix = np.where(yte == cls)[0][:2000]; _, s1, _ = MODELS[n].eval().spikes(torch.as_tensor(Xte[ix], dtype=torch.float32, device=DEV)); return s1.mean((0, 2)).cpu().numpy()
fig, ax = plt.subplots(2, 1, figsize=(10, 6.5))
for k, c, l in [(1, 'C0', 'Non-stress'), (2, 'C3', 'Stress')]: ax[0].plot(zs(eda[:len(l4)][l4 == k][:480]), c, lw=.9, label=l)
ax[0].set_title(f'Real z-scored EDA, subject {s0} (2 min each)'); ax[0].legend()
for n, ls in [('SNN-LIF', '-'), ('SNN-IF', '--')]:
    for cls, c in [(0, 'C0'), (1, 'C3')]: ax[1].plot(dens(n, cls), c + ls, marker='o', ms=3, label=f'{n} {"stress" if cls else "non-stress"}')
ax[1].set_title('Mean hidden-layer-1 spike density per time step'); ax[1].set_xlabel('Time step'); ax[1].legend(fontsize=7); savefig('Fig10_spike_density_stress_vs_nonstress.png')

if cv_rows:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n, c in [('SNN-LIF', 'C0'), ('SNN-IF', 'C1'), ('MLP-ANN', 'C2')]:
        g = cv_df[cv_df.model == n]; ax.plot(g.fold, g.acc, c, marker='o', label=f'{n} mean={g.acc.mean():.4f}')
    ax.set_xlabel('Fold (disjoint subjects)'); ax.set_ylabel('Accuracy'); ax.set_title(f'{CFG["N_FOLDS"]}-fold subject-wise CV'); ax.legend(); savefig('Fig11_cross_validation.png')

fig, ax = plt.subplots(1, 2, figsize=(11, 4)); E2 = eff_df.dropna(subset=['SynOps']).set_index('model')
E2[['dense_FLOPs_eq23', 'SynOps']].plot.bar(ax=ax[0], logy=True, rot=20); ax[0].set_title('Dense FLOPs (Eq.23) vs executed synaptic ops / inference')
E2['energy_J'].plot.bar(ax=ax[1], logy=True, rot=20, color='C2'); ax[1].set_title('Estimated energy / inference (J, FP32 op-cost model)'); savefig('Fig12_computation_energy.png')

if abl:
    a_df = pd.DataFrame(abl); fig, ax = plt.subplots(figsize=(11, 4.5)); ax.bar(range(len(a_df)), a_df.acc, color='C0'); ax.set_xticks(range(len(a_df)), a_df.experiment, rotation=40, ha='right')
    ax.set_ylim(max(0, a_df.acc.min() - .05), 1.0); ax.set_ylabel('Test accuracy'); ax.set_title('Ablation (real runs, unseen subjects)'); savefig('Fig13_ablation.png')
if mc_cm:
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a_, (n, cm) in zip(ax, mc_cm.items()):
        a_.imshow(cm, cmap='Blues'); a_.grid(False); a_.set_title(f'3-class {n}')
        for i in range(3):
            for j in range(3): a_.text(j, i, cm[i, j], ha='center', va='center')
        a_.set_xticks(range(3), ['base', 'stress', 'amus']); a_.set_yticks(range(3), ['base', 'stress', 'amus'])
    savefig('Fig14_3class_confusion.png')

json.dump(dict(cfg={k: v for k, v in CFG.items()}, train=TRAIN_S, val=VAL_S, test=TEST_S, mcnemar=sig,
               n_windows=int(len(yb)), stress_pct=float(100 * yb.mean())), open(os.path.join(RES, 'run_summary.json'), 'w'), indent=1, default=str)
log('DONE. Tables ->', RES, '| Figures ->', FIG)
