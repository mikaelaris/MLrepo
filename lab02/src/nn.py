"""Two-hidden-layer neural network from scratch (NumPy only).

Architecture: 64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax
Shapes:  W1 (64,W)  b1 (W,)   W2 (W,W)  b2 (W,)   W3 (W,10)  b3 (10,)

Loss: mean cross-entropy over the batch.

Backward pass (batch size B, row-vector convention, see report.md):
    dZ3 = (P - Y) / B
    dW3 = A2.T @ dZ3          db3 = dZ3.sum(0)
    dA2 = dZ3 @ W3.T          dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2          db2 = dZ2.sum(0)
    dA1 = dZ2 @ W2.T          dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T  @ dZ1          db1 = dZ1.sum(0)
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

PARAM_NAMES = ["W1", "b1", "W2", "b2", "W3", "b3"]
WEIGHT_NAMES = ["W1", "W2", "W3"]


def load_data(seed=42):
    """load_digits, inputs scaled by 1/16, one-hot labels, 70/30 stratified split."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    Y = np.eye(10)[y]
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
        X, Y, y, test_size=0.30, stratify=y, random_state=seed
    )
    return X_tr, X_te, Y_tr, Y_te, y_tr, y_te


class MLP:
    def __init__(self, width, seed=42, init_std=0.1):
        rng = np.random.default_rng(seed)
        W = width
        self.params = {
            "W1": rng.normal(0.0, init_std, (64, W)),
            "b1": np.zeros(W),
            "W2": rng.normal(0.0, init_std, (W, W)),
            "b2": np.zeros(W),
            "W3": rng.normal(0.0, init_std, (W, 10)),
            "b3": np.zeros(10),
        }

    # ---------------- forward ----------------
    def forward(self, X):
        p = self.params
        Z1 = X @ p["W1"] + p["b1"]
        A1 = np.maximum(Z1, 0.0)
        Z2 = A1 @ p["W2"] + p["b2"]
        A2 = np.maximum(Z2, 0.0)
        Z3 = A2 @ p["W3"] + p["b3"]
        Z3s = Z3 - Z3.max(axis=1, keepdims=True)  # numerical stability
        E = np.exp(Z3s)
        P = E / E.sum(axis=1, keepdims=True)
        cache = (X, Z1, A1, Z2, A2, P)
        return P, cache

    # ---------------- loss ----------------
    @staticmethod
    def cross_entropy(P, Y):
        return float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-300, None)), axis=1)))

    def loss(self, X, Y):
        P, _ = self.forward(X)
        return self.cross_entropy(P, Y)

    # ---------------- backward ----------------
    def backward(self, cache, Y):
        X, Z1, A1, Z2, A2, P = cache
        p = self.params
        B = X.shape[0]
        dZ3 = (P - Y) / B
        g = {}
        g["W3"] = A2.T @ dZ3
        g["b3"] = dZ3.sum(axis=0)
        dZ2 = (dZ3 @ p["W3"].T) * (Z2 > 0)
        g["W2"] = A1.T @ dZ2
        g["b2"] = dZ2.sum(axis=0)
        dZ1 = (dZ2 @ p["W2"].T) * (Z1 > 0)
        g["W1"] = X.T @ dZ1
        g["b1"] = dZ1.sum(axis=0)
        return g

    def step(self, grads, lr):
        for k in PARAM_NAMES:
            self.params[k] -= lr * grads[k]

    def predict(self, X):
        P, _ = self.forward(X)
        return P.argmax(axis=1)


def gradient_check(model, X, Y, n_checks=10, eps=1e-5, seed=42):
    """Central-difference check of n random weight entries (from W1, W2, W3).

    Returns a list of dicts with the analytic slope, numerical slope and the
    relative error |a - n| / max(|a|, |n|, 1e-12).
    """
    rng = np.random.default_rng(seed)
    P, cache = model.forward(X)
    grads = model.backward(cache, Y)
    # Sample distinct weight entries uniformly from all entries of W1, W2, W3.
    sizes = [model.params[n].size for n in WEIGHT_NAMES]
    flat = rng.choice(sum(sizes), size=n_checks, replace=False)
    bounds = np.cumsum([0] + sizes)
    rows = []
    for f in flat:
        k = int(np.searchsorted(bounds, f, side="right") - 1)
        name = WEIGHT_NAMES[k]
        arr = model.params[name]
        idx = tuple(int(i) for i in np.unravel_index(int(f - bounds[k]), arr.shape))
        old = arr[idx]
        arr[idx] = old + eps
        lp = model.loss(X, Y)
        arr[idx] = old - eps
        lm = model.loss(X, Y)
        arr[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = float(grads[name][idx])
        rel = abs(ana - num) / max(abs(ana), abs(num), 1e-12)
        rows.append({"param": name, "index": idx, "analytic": ana,
                     "numerical": num, "rel_error": rel})
    return rows
