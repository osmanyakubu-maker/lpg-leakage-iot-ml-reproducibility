"""Sequence baselines (LSTM, 1D-CNN, Attentional Bi-LSTM) for Table 3. Requires PyTorch.

Input: the last `seq_len` samples of the five scaled readings [G, T, H, S, F] within a run
(left-padded with the run's first sample). SMOTE is not applied to sequences; the
critical-class cost weight is applied through the loss.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


def make_windows(X: np.ndarray, groups: np.ndarray, seq_len: int) -> np.ndarray:
    """Index array (n, seq_len) of past samples within the same run (data sorted by run, time)."""
    n = len(X)
    starts = np.zeros(n, int)
    change = np.r_[True, groups[1:] != groups[:-1]]
    run_start = np.maximum.accumulate(np.where(change, np.arange(n), 0))
    starts[:] = run_start
    offs = np.arange(-seq_len + 1, 1)
    idx = np.arange(n)[:, None] + offs[None, :]
    return np.maximum(idx, starts[:, None])


class LSTMNet(nn.Module):
    def __init__(self, d, hidden, layers, k=5):
        super().__init__()
        self.rnn = nn.LSTM(d, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, k)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


class CNN1D(nn.Module):
    def __init__(self, d, channels, kernel, k=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, channels, kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(channels, k)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1))


class AttnBiLSTM(nn.Module):
    def __init__(self, d, hidden, k=5):
        super().__init__()
        self.rnn = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.head = nn.Linear(2 * hidden, k)

    def forward(self, x):
        h, _ = self.rnn(x)
        a = torch.softmax(self.att(h).squeeze(-1), dim=1)
        return self.head((a.unsqueeze(-1) * h).sum(1))


def _build(name, p, d):
    if name == "LSTM":
        return LSTMNet(d, p["hidden"], p["layers"])
    if name == "1D-CNN":
        return CNN1D(d, p["channels"], p["kernel"])
    return AttnBiLSTM(d, p["hidden"])


def train_eval_sequence(name, params, data, cfg, seed, split, batch=1024):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = params["seq_len"]
    tr, ev = data["train"], data[split]
    Xtr = torch.tensor(tr["X"][:, :5], dtype=torch.float32)
    Wtr = torch.tensor(make_windows(tr["X"], tr["groups"], L))
    ytr = torch.tensor(tr["y"], dtype=torch.long)
    model = _build(name, params, 5).to(dev)
    cw = torch.ones(5); cw[cfg["critical_class"]] = cfg["imbalance"]["critical_cost_weight"]
    lossf = nn.CrossEntropyLoss(weight=cw.to(dev))
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    g = torch.Generator().manual_seed(seed)
    for _ in range(params["epochs"]):
        model.train()
        for b in torch.randperm(len(ytr), generator=g).split(batch):
            xb = Xtr[Wtr[b]].to(dev)
            opt.zero_grad()
            lossf(model(xb), ytr[b].to(dev)).backward()
            opt.step()
    model.eval()
    Xe = torch.tensor(ev["X"][:, :5], dtype=torch.float32)
    We = torch.tensor(make_windows(ev["X"], ev["groups"], L))
    probs = []
    with torch.no_grad():
        for b in torch.arange(len(We)).split(batch * 4):
            probs.append(torch.softmax(model(Xe[We[b]].to(dev)), 1).cpu().numpy())
    proba = np.vstack(probs)
    return classification_metrics(ev["y"], proba.argmax(1), proba, cfg["critical_class"])
