import json
import random
import sys
import threading
import time
import warnings

import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

FACTORIES = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
}


def main():
    name, phase = sys.argv[1], sys.argv[2]
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    proc = psutil.Process()
    model = None
    if phase == "infer":
        model = FACTORIES[name]().fit(X_tr, y_tr)
        model.predict(X_te[:1])  # warm-up

    baseline = proc.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.001)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    if phase == "train":
        FACTORIES[name]().fit(X_tr, y_tr)
    else:
        for _ in range(100):
            model.predict(X_te[:1])
    peak[0] = max(peak[0], proc.memory_info().rss)
    stop.set()
    t.join()

    mb = 1024 * 1024
    print(json.dumps({
        "model": name, "phase": phase,
        "baseline_rss_mb": round(baseline / mb, 1),
        "peak_rss_mb": round(peak[0] / mb, 1),
        "delta_mb": round((peak[0] - baseline) / mb, 2),
    }))


if __name__ == "__main__":
    main()
