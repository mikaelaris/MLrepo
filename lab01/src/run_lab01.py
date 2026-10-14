"""Lab 1"""
import json
import os
import subprocess
import sys
import warnings
import random
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

N_TRAIN_RUNS = 5
N_INFER_RUNS = 100

warnings.filterwarnings("ignore")
PROBE = Path(__file__).with_name("mem_probe.py")


def probe(name, phase):
    out = subprocess.run(
        [sys.executable, str(PROBE), name, phase],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1]
    return json.loads(out)


def make_models():
    return {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    }


def timed_fit(factory, X, y):
    model = factory()
    t0 = time.perf_counter()
    model.fit(X, y)
    return model, time.perf_counter() - t0


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    acc_rows, cost_rows = [], []
    for name, factory in make_models().items():
        # --- accuracy (single fit, fixed seed) ---
        model = factory().fit(X_tr, y_tr)
        acc = float(model.score(X_te, y_te))
        acc_rows.append({"model": name, "test_accuracy": round(acc, 4)})

        # --- training time: 1 warm-up + median of 5 ---
        timed_fit(factory, X_tr, y_tr)
        train_times = [timed_fit(factory, X_tr, y_tr)[1] for _ in range(N_TRAIN_RUNS)]
        train_med = statistics.median(train_times)

        # --- single-sample inference: 1 warm-up + 100 repeats, median ---
        x1 = X_te[:1]
        model.predict(x1)
        lat = []
        for _ in range(N_INFER_RUNS):
            t0 = time.perf_counter()
            model.predict(x1)
            lat.append((time.perf_counter() - t0) * 1000)
        lat_med_ms = statistics.median(lat)

        # --- model size ---
        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_bytes = os.path.getsize(path)

        # --- peak memory: fresh subprocess per phase, no torch loaded ---
        mem = {ph: probe(name, ph) for ph in ("train", "infer")}

        cost_rows.append({
            "model": name,
            "test_accuracy": round(acc, 4),
            "train_time_median_s": round(train_med, 4),
            "train_times_s": ";".join(f"{t:.4f}" for t in train_times),
            "infer_latency_median_ms": round(lat_med_ms, 4),
            "model_size_bytes": size_bytes,
            "model_size_kb": round(size_bytes / 1024, 2),
            "mem_baseline_mb": mem["train"]["baseline_rss_mb"],
            "train_peak_rss_mb": mem["train"]["peak_rss_mb"],
            "train_peak_delta_mb": mem["train"]["delta_mb"],
            "infer_baseline_mb": mem["infer"]["baseline_rss_mb"],
            "infer_peak_rss_mb": mem["infer"]["peak_rss_mb"],
            "infer_peak_delta_mb": mem["infer"]["delta_mb"],
        })
        print(name, cost_rows[-1])

    pd.DataFrame(acc_rows).to_csv(RESULTS / "baseline_accuracy.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(RESULTS / "system_cost.csv", index=False)

    # --- budgets and fit decision ---
    budgets = {
        # name: (mem_min_mb, mem_max_mb, latency_ms, size_kb)
        "Cloud": (1024, None, 100, 500 * 1024),
        "Edge": (256, 1024, 50, 50 * 1024),
        "Mobile": (64, 256, 20, 10 * 1024),
        "TinyML": (0, 0.25, 10, 100),
    }
    rows = []
    for c in cost_rows:
        for tier, (mmin, mmax, lat, size_kb) in budgets.items():
            mem_limit = mmax if mmax is not None else float("inf")
            rows.append({
                "model": c["model"],
                "tier": tier,
                "latency_ok": c["infer_latency_median_ms"] <= lat,
                "size_ok": c["model_size_kb"] <= size_kb,
                # Memory: the whole Python process RSS is the conservative figure.
                "memory_ok_process_peak": max(c["train_peak_rss_mb"], c["infer_peak_rss_mb"]) <= mem_limit,
                "memory_ok_model_only": c["model_size_kb"] / 1024 <= mem_limit,
            })
    pd.DataFrame(rows).to_csv(RESULTS / "budget_table.csv", index=False)


if __name__ == "__main__":
    main()
