import os
import zipfile
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from spikingjelly.activation_based import neuron


DATA_DIR = "WESAD/WESAD"
subjects = [d for d in os.listdir(DATA_DIR) if d.startswith('S')]
print(f"Found subjects: {subjects}")

X_list, y_list = [], []
window_seconds = 60
sample_rate = 4
window_size = window_seconds * sample_rate

for subj in subjects:
    subj_path = os.path.join(DATA_DIR, subj)
    e4_zip_path = os.path.join(subj_path, f"{subj}_E4_Data.zip")
    label_file_path = os.path.join(subj_path, f"{subj}.pkl")

    with open(label_file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        labels = data['label']

    with zipfile.ZipFile(e4_zip_path, 'r') as z:
        eda = pd.read_csv(z.open('EDA.csv'), header=None, skiprows=2).values
        bvp = pd.read_csv(z.open('BVP.csv'), header=None, skiprows=2).values
        temp = pd.read_csv(z.open('TEMP.csv'), header=None, skiprows=2).values
        acc = pd.read_csv(z.open('ACC.csv'), header=None, skiprows=2).values

    min_len = min(len(eda), len(bvp), len(temp), len(acc), len(labels))
    eda, bvp, temp, acc, labels = eda[:min_len], bvp[:min_len], temp[:min_len], acc[:min_len], labels[:min_len]

    for i in range(min_len // window_size):
        idx = slice(i * window_size, (i + 1) * window_size)
        label_window = labels[idx]
        label_majority = np.bincount(label_window).argmax()
      
        label_bin = 1 if label_majority == 1 else 0

        features = np.concatenate([
            eda[idx].mean(axis=0), eda[idx].std(axis=0),
            bvp[idx].mean(axis=0), bvp[idx].std(axis=0),
            temp[idx].mean(axis=0), temp[idx].std(axis=0),
            acc[idx].mean(axis=0), acc[idx].std(axis=0),
        ])

        X_list.append(features)
        y_list.append(label_bin)

print(f"Extracted {len(X_list)} windows total.")

X = np.vstack(X_list)
y = np.array(y_list)

print("Original class distribution:", Counter(y))


if np.isnan(X).any():
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    print("NaNs handled with mean imputation.")


if len(np.unique(y)) > 1:
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y)
    print("Balanced class distribution:", Counter(y))
else:
    print("Only one class present, skipping SMOTE.")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)


class SNN(nn.Module):
    def __init__(self, neuron_type='LIF', hidden=64, n_class=2):
        super().__init__()
        Neuron = neuron.LIFNode if neuron_type == 'LIF' else neuron.IFNode
        self.fc1 = nn.Linear(X_train.size(1), hidden)
        self.lif1 = Neuron()
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = Neuron()
        self.fc_out = nn.Linear(hidden, n_class)

    def forward(self, x):
        h = self.lif1(self.fc1(x))
        h = self.lif2(self.fc2(h))
        out = self.fc_out(h)
        return out

def reset_neurons(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()


def train_eval(model, train_loader, test_loader, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            reset_neurons(model)  

            optimizer.zero_grad()
            out = model(batch_x)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []
    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            reset_neurons(model)  

            out = model(batch_x)
            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(batch_y.cpu())
    latency = (time.time() - start_time) / len(test_loader.dataset)

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    acc = (all_preds == all_targets).mean()
    f1 = f1_score(all_targets, all_preds, average='macro')
    auc = roc_auc_score(all_targets, all_probs[:, 1])
    return acc, f1, auc, latency


model_LIF = SNN('LIF')
model_IF = SNN('IF')

acc_LIF, f1_LIF, auc_LIF, latency_LIF = train_eval(model_LIF, train_loader, test_loader)
acc_IF, f1_IF, auc_IF, latency_IF = train_eval(model_IF, train_loader, test_loader)

print(f"\n LIF Node:\nAccuracy: {acc_LIF:.4f}, F1: {f1_LIF:.4f}, AUC: {auc_LIF:.4f}, Latency: {latency_LIF*1000:.2f} ms")
print(f"\n IF Node:\nAccuracy: {acc_IF:.4f}, F1: {f1_IF:.4f}, AUC: {auc_IF:.4f}, Latency: {latency_IF*1000:.2f} ms")
