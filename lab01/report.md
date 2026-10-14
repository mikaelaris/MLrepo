# Lab 1: Environment and first system measurements

## 1. Goal

Build a reproducible, pinned Python environment, train two baseline classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset, measure their system cost (training time, single-sample latency, model size, peak memory), and decide which deployment tiers (Cloud, Edge, Mobile, TinyML) each model fits.

## 2. Method

- **Environment:** Python 3.11 virtual environment with the pinned `requirements.txt`; `src/print_versions.py` output is in `results/versions.txt`. The interpreter is 3.11.17, not the 3.11.8 named in the manual (the 3.11.8 build was not available on the machine). One pin differs from the manual: `pyarrow==15.0.2` instead of `16.1.0`, because `mlflow==2.14.1` requires `pyarrow<16` and pip cannot resolve the manual's pins together. Neither package is used in this lab.
- **Data:** `load_breast_cancer(return_X_y=True)`, 70/30 split, stratified, `random_state=42`. Seeds set to 42 for `random`, NumPy and PyTorch.
- **Models:** `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- **Training time:** one warm-up fit, then median of 5 timed fits (`time.perf_counter`).
- **Inference latency:** one warm-up call, then median of 100 single-sample `predict` calls.
- **Model size:** `joblib.dump` to disk, size in bytes and KB (one file per model: `results/model_<name>.joblib`).
- **Peak memory:** each phase (training, 100 inferences) runs in a fresh subprocess that does not import PyTorch. A background thread samples the process RSS (`psutil`) every millisecond. I report the peak RSS and the increase over the baseline taken after imports and data loading. Measuring inside the main script would have included about 440 MB from importing PyTorch, which is irrelevant to these models.
- **Scripts:** all code is in `src/` (`run_lab01.py`, `mem_probe.py`, `print_versions.py`).

## 3. Results

Raw data: `results/baseline_accuracy.csv`, `results/system_cost.csv`, `results/budget_table.csv`.

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Test accuracy | 0.9474 | 0.9357 |
| Training time, median of 5 (s) | 0.0762 | 0.1550 |
| Single-sample latency, median of 100 (ms) | 0.0484 | 2.3363 |
| Model size (bytes) | 1,103 | 290,905 |
| Model size (KB) | 1.08 | 284.09 |
| Peak RSS during training (MB) | 116.0 | 116.5 |
| Extra RSS during training (MB) | 0.59 | 0.95 |
| Peak RSS during inference (MB) | 115.8 | 116.5 |
| Extra RSS during inference (MB) | 0.02 | 0.20 |

### Deployment fit

Budgets: Cloud (memory at least 1 GB, latency at most 100 ms, size at most 500 MB), Edge (256-1024 MB, at most 50 ms, at most 50 MB), Mobile (64-256 MB, at most 20 ms, at most 10 MB), TinyML (at most 256 KB, at most 10 ms, at most 100 KB). "Process" is the whole Python process peak (about 116 MB); "model" is the model's own size only.

| Tier | LR latency | LR size | LR memory | RF latency | RF size | RF memory |
|---|---|---|---|---|---|---|
| Cloud | pass | pass | pass | pass | pass | pass |
| Edge | pass | pass | pass | pass | pass | pass |
| Mobile | pass | pass | pass | pass | pass | pass |
| TinyML | pass | pass | process fails, model alone passes | pass | **fail** (284 KB > 100 KB) | process fails, model alone fails (284 KB > 256 KB) |

- **Logistic Regression** can run on all four tiers, provided TinyML uses a stripped-down implementation. The 1 KB model and 0.05 ms latency leave a huge margin, and the 31 coefficients and intercept need only a few hundred bytes of RAM. The scikit-learn/Python runtime (about 116 MB) is what fails the TinyML memory budget, not the model.
- **Random Forest** fits Cloud, Edge and Mobile with large margins (284 KB against 10 MB; 2.3 ms against 20 ms). It does not fit TinyML: the serialised model is 2.8 times over the 100 KB size budget and larger than the 256 KB memory budget by itself.

## 4. Conclusions

1. Random Forest scored slightly lower than Logistic Regression here (0.9357 vs 0.9474) while costing about 2x the training time, about 48x the inference latency and about 260x the model size. On this small, nearly linearly separable dataset the extra system cost buys no accuracy.
2. Measured memory depends heavily on what is in the process: importing PyTorch alone added several hundred MB, and the Python/scikit-learn runtime sets a floor of about 116 MB. For small models, the runtime dominates, so deployment memory must be judged on the full serving stack, not just the model file.
3. Every deployment decision above comes from the median of repeated runs after a warm-up, and the budgets were met by wide margins for everything except TinyML. The model file size separates the two models at the tightest tier, and the latency gap (about 0.05 ms vs 2.3 ms) would matter only at much higher request rates.

## Notes and caveats

- `LogisticRegression` emits a convergence warning with `max_iter=1000` on the unscaled features. I kept the manual's settings; accuracy is still 0.9474. Scaling the features would likely remove the warning.
- Timings come from a shared 2-core cloud VM; absolute values will differ on other hardware, though the ratios should hold.
- Accuracy is from a single fixed split and seed, so a 1.2 point gap between the models is within what a different split could change.
