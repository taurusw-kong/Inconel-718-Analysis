"""
ParticleVision V4
=================

FT-Transformer (Feature Tokenizer + Transformer)
-----------------------------------------------

Implements the architecture of Gorishniy et al., "Revisiting Deep
Learning Models for Tabular Data" (NeurIPS 2021), for Section 4.8.2 of
the manuscript, wrapped in a scikit-learn-compatible estimator so it
drops into the SAME group-aware cross-validation harness as every other
model in classification.py — no separate evaluation path, no separate
preprocessing, no opportunity for an accidental advantage.

Written directly against torch rather than pulling in `rtdl`: the
architecture is small, and a local implementation keeps the dependency
surface (and the reviewer's "what exactly did you run?" question)
minimal and inspectable.

ARCHITECTURE
    Feature Tokenizer   each of the d scalar features becomes a
                        d_token-dimensional embedding via its own
                        learned weight + bias, so a transformer can
                        attend over features the way it would over
                        tokens in a sequence.
    [CLS] token         prepended; its final representation is what the
                        prediction head reads (standard BERT-style
                        pooling, as in the original paper).
    Transformer blocks  pre-norm multi-head self-attention + FFN, which
                        trains more stably at small depth than the
                        post-norm variant.
    Head                LayerNorm -> ReLU -> Linear to n_classes.

EXPECTATION, STATED UP FRONT
    Transformers are data-hungry. On a labeled set of this size this
    model is expected to UNDERPERFORM gradient-boosted trees. That is a
    legitimate, reportable finding (it is the central claim of
    Grinsztajn et al. 2022, cited in the manuscript), not a failure of
    the implementation — which is exactly why it is worth running
    rather than asserting.
"""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _pick_device() -> torch.device:
    """MPS (Apple GPU) when available, else CPU. CUDA covered for portability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# MODULES
# ============================================================

class FeatureTokenizer(nn.Module):
    """
    Turns a row of d scalar features into d tokens of width d_token.

    Each feature gets its OWN weight vector and bias (rather than one
    shared projection), which is what lets the transformer distinguish
    "the value 0.9 in the circularity column" from "the value 0.9 in the
    solidity column" — a shared projection would map them identically
    and throw away the column identity the whole architecture depends on.
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))

        # Uniform(-1/sqrt(d), 1/sqrt(d)) — the initialization used in the
        # reference implementation; wider init destabilizes attention at
        # this depth.
        bound = 1.0 / np.sqrt(d_token)
        for p in (self.weight, self.bias, self.cls_token):
            nn.init.uniform_(p, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features + 1, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class TransformerBlock(nn.Module):
    """Pre-norm attention + feed-forward residual block."""

    def __init__(self, d_token: int, n_heads: int, dropout: float, ffn_factor: float = 4 / 3):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.drop_attn = nn.Dropout(dropout)

        d_hidden = int(d_token * ffn_factor)
        self.norm_ffn = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_hidden, d_token),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_attn(attn_out)
        x = x + self.drop_ffn(self.ffn(self.norm_ffn(x)))
        return x


class _FTTransformerNet(nn.Module):

    def __init__(self, n_features: int, n_classes: int, d_token: int,
                 n_blocks: int, n_heads: int, dropout: float):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_token, n_heads, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])       # [CLS] position only


# ============================================================
# SKLEARN-COMPATIBLE WRAPPER
# ============================================================

class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-API wrapper so this model can be passed to
    cross_val_predict alongside XGBoost et al. without special-casing.

    A held-out slice of each training fold drives early stopping. That
    split is carved out of the TRAINING data only — the outer CV's test
    fold is never touched here, so early stopping cannot leak the
    quantity being measured.

    That inner split must ALSO be group-aware, and `fit` accepts the
    training fold's `groups` to make it so. Splitting the inner
    validation set at random instead puts rotated/flipped views of the
    same source particle on both sides of it; the network then trivially
    memorizes them, inner validation loss collapses toward zero, and
    early stopping fires on a signal that says nothing about
    generalization — handicapping the model with a badly-chosen stopping
    point rather than measuring it fairly. (Observed directly here:
    random inner splits drove val loss to ~0.002 within ~140 epochs.)
    """

    def __init__(self, d_token: int = 64, n_blocks: int = 3, n_heads: int = 8,
                 dropout: float = 0.2, lr: float = 1e-3, weight_decay: float = 1e-5,
                 batch_size: int = 64, max_epochs: int = 200, patience: int = 20,
                 val_fraction: float = 0.15, random_state: int = 42, verbose: bool = False):
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = _pick_device()

        # Inner validation split for early stopping. When groups are
        # supplied, hold out whole GROUPS (all augmented views of a
        # source particle move together) so inner validation measures
        # generalization to unseen particles, not recall of rotated
        # copies already seen during training.
        n_val = max(int(len(X) * self.val_fraction), n_classes)

        if groups is not None:
            groups = np.asarray(groups)
            unique = rng.permutation(np.unique(groups))
            held, count = [], 0
            for g in unique:
                if count >= n_val:
                    break
                held.append(g)
                count += int(np.sum(groups == g))
            val_mask = np.isin(groups, held)
            # Guard against a degenerate split when one group dominates.
            if val_mask.all() or not val_mask.any():
                perm = rng.permutation(len(X))
                val_idx, train_idx = perm[:n_val], perm[n_val:]
            else:
                val_idx = np.flatnonzero(val_mask)
                train_idx = np.flatnonzero(~val_mask)
        else:
            perm = rng.permutation(len(X))
            val_idx, train_idx = perm[:n_val], perm[n_val:]

        Xtr = torch.tensor(X[train_idx], device=device)
        ytr = torch.tensor(y_idx[train_idx], device=device)
        Xva = torch.tensor(X[val_idx], device=device)
        yva = torch.tensor(y_idx[val_idx], device=device)

        # d_token must divide evenly into n_heads for MultiheadAttention
        n_heads = self.n_heads
        while self.d_token % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        self.model_ = _FTTransformerNet(
            self.n_features_in_, n_classes, self.d_token,
            self.n_blocks, n_heads, self.dropout,
        ).to(device)

        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_loss, best_state, bad_epochs = float("inf"), None, 0

        for epoch in range(self.max_epochs):
            self.model_.train()
            order = torch.randperm(len(Xtr), device=device)
            for start in range(0, len(Xtr), self.batch_size):
                idx = order[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model_(Xva), yva).item()

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    if self.verbose:
                        print(f"  early stop at epoch {epoch} (val loss {best_loss:.4f})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.device_ = device
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X, device=self.device_))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


__all__ = ["FTTransformerClassifier"]
