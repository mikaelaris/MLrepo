"""Lab 2 driver: gradient check, training grid, loss curves, fwd/bwd timing.

Outputs in lab02/results/:
  gradient_check.csv, gradient_check_robustness.txt, training_log.csv,
  final_accuracy.csv, loss_curves.png, loss_curves_std0.01.png, timing.csv
"""
import json
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nn import MLP, gradient_check, load_data  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42
LRS = [0.001, 0.01, 0.1]
WIDTHS = [16, 64]
EPOCHS = 50
BATCH = 32


def train(width, lr, data, init_std, seed=SEED):
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = data
    model = MLP(width, seed=seed, init_std=init_std)
    rng = np.random.default_rng(seed)
    log = []
    n = len(X_tr)
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)  # shuffle every epoch
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            _, cache = model.forward(X_tr[b])
            model.step(model.backward(cache, Y_tr[b]), lr)
        log.append({
            "width": width, "lr": lr, "init_std": init_std, "epoch": epoch,
            "train_loss": model.loss(X_tr, Y_tr),
            "test_loss": model.loss(X_te, Y_te),
            "train_acc": float((model.predict(X_tr) == y_tr).mean()),
            "test_acc": float((model.predict(X_te) == y_te).mean()),
        })
    return model, log


def run_grid(data, init_std):
    logs = []
    for W in WIDTHS:
        for lr in LRS:
            _, log = train(W, lr, data, init_std)
            logs += log
            last = log[-1]
            print(f"std={init_std} W={W:>2} lr={lr:<5} train_loss={last['train_loss']:.4f} "
                  f"test_acc={last['test_acc']:.4f}")
    return pd.DataFrame(logs)


def plot_curves(df, path, title_suffix):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, W in zip(axes, WIDTHS):
        for lr in LRS:
            d = df[(df.width == W) & (df.lr == lr)]
            ax.plot(d.epoch, d.train_loss, label=f"lr={lr}")
        ax.axhline(np.log(10), color="grey", ls=":", lw=1, label="ln 10")
        ax.set_title(f"Hidden width W={W}{title_suffix}")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("training cross-entropy (log scale)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def time_passes(width, X, Y, runs=100, warmup=10):
    model = MLP(width, seed=SEED)
    for _ in range(warmup):
        _, cache = model.forward(X)
        model.backward(cache, Y)
    fwd, bwd = [], []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        _, cache = model.forward(X)
        t1 = time.perf_counter_ns()
        model.backward(cache, Y)
        t2 = time.perf_counter_ns()
        fwd.append((t1 - t0) / 1e3)
        bwd.append((t2 - t1) / 1e3)
    return statistics.median(fwd), statistics.median(bwd)


def time_matmuls(width, B, runs=100, warmup=10):
    """Median time (us) of only the matrix products of one forward / one backward pass."""
    rng = np.random.default_rng(0)
    W = width
    X, A1, A2 = rng.random((B, 64)), rng.random((B, W)), rng.random((B, W))
    W1, W2, W3 = rng.random((64, W)), rng.random((W, W)), rng.random((W, 10))
    d3, d2, d1 = rng.random((B, 10)), rng.random((B, W)), rng.random((B, W))

    def fwd():
        X @ W1; A1 @ W2; A2 @ W3

    def bwd():
        A2.T @ d3; d3 @ W3.T; A1.T @ d2; d2 @ W2.T; X.T @ d1

    out = []
    for fn in (fwd, bwd):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            fn()
            ts.append((time.perf_counter_ns() - t0) / 1e3)
        out.append(statistics.median(ts))
    return out


def matmul_flops(width, B):
    """Multiply-add FLOPs (2 per MAC) of the three matmuls in the forward pass."""
    macs = B * (64 * width + width * width + width * 10)
    return 2 * macs


def matmul_flops_bwd(width, B):
    """dW3, dA2, dW2, dA1, dW1 (dX of layer 1 is not needed)."""
    macs = B * (10 * width + 10 * width + width * width + width * width + 64 * width)
    return 2 * macs


def main():
    data = load_data(SEED)
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = data

    # ---- Step 5: gradient check (seed 42, reported) + robustness over seeds ----
    gc_rows = []
    robust_lines = []
    for W in WIDTHS:
        rows = gradient_check(MLP(W), X_tr, Y_tr)
        for r in rows:
            gc_rows.append({"width": W, **r, "index": str(r["index"]),
                            "pass_1e-6": r["rel_error"] < 1e-6})
        passed = sum(r["rel_error"] < 1e-6 for r in rows)
        print(f"gradient check W={W}: {passed}/10 pass, max rel err {max(r['rel_error'] for r in rows):.2e}")
        m = MLP(W)
        ok, ok_big = 0, 0
        for sd in range(200):
            rr = gradient_check(m, X_tr, Y_tr, seed=sd)
            ok += sum(r["rel_error"] < 1e-6 for r in rr) >= 9
            big = [r for r in rr if abs(r["analytic"]) > 1e-4]
            ok_big += all(r["rel_error"] < 1e-6 for r in big)
        robust_lines.append(
            f"W={W}: criterion (>=9/10 with rel err<1e-6) met for {ok}/200 random draws of 10 weights; "
            f"all entries with |grad|>1e-4 below 1e-6 in {ok_big}/200 draws")
    pd.DataFrame(gc_rows).to_csv(RESULTS / "gradient_check.csv", index=False)
    (RESULTS / "gradient_check_robustness.txt").write_text("\n".join(robust_lines) + "\n")
    print("\n".join(robust_lines))

    # ---- Step 6: training grid (N(0, 0.01) read as variance 0.01 -> std 0.1) ----
    df = run_grid(data, init_std=0.1)
    df.to_csv(RESULTS / "training_log.csv", index=False)
    plot_curves(df, RESULTS / "loss_curves.png", ", init std 0.1")
    final = df[df.epoch == EPOCHS][["width", "lr", "train_loss", "test_loss", "train_acc", "test_acc"]]
    final.round(4).to_csv(RESULTS / "final_accuracy.csv", index=False)

    # Sensitivity: the other reading of N(0, 0.01) (std 0.01)
    df2 = run_grid(data, init_std=0.01)
    df2.to_csv(RESULTS / "training_log_std0.01.csv", index=False)
    plot_curves(df2, RESULTS / "loss_curves_std0.01.png", ", init std 0.01")
    df2[df2.epoch == EPOCHS][["width", "lr", "train_loss", "test_loss", "train_acc", "test_acc"]] \
        .round(4).to_csv(RESULTS / "final_accuracy_std0.01.csv", index=False)

    # ---- Step 7: forward / backward timing ----
    trows = []
    for W in WIDTHS:
        for B in (32, len(X_tr)):
            f, b = time_passes(W, X_tr[:B], Y_tr[:B])
            fm, bm = time_matmuls(W, B)
            trows.append({
                "width": W, "batch": B, "forward_us": round(f, 1), "backward_us": round(b, 1),
                "backward_over_forward": round(b / f, 2),
                "forward_matmul_only_us": round(fm, 1), "backward_matmul_only_us": round(bm, 1),
                "forward_matmul_share": round(fm / f, 2), "backward_matmul_share": round(bm / b, 2),
                "forward_matmul_flops": matmul_flops(W, B),
                "backward_matmul_flops": matmul_flops_bwd(W, B),
            })
    tdf = pd.DataFrame(trows)
    tdf.to_csv(RESULTS / "timing.csv", index=False)
    print(tdf.to_string(index=False))
    (RESULTS / "run_info.json").write_text(json.dumps(
        {"seed": SEED, "epochs": EPOCHS, "batch": BATCH, "lrs": LRS, "widths": WIDTHS}, indent=2))


if __name__ == "__main__":
    main()
