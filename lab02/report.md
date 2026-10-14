# Lab 2: neural network from scratch

Everything is plain NumPy (`src/nn.py`). The experiment script is `src/run_lab02.py`, tests are in `tests/test_nn.py` (run with `python -m pytest lab02/tests`), and the raw outputs live in `results/`.

## 1. Data

I used `load_digits` (1797 samples, 64 features). Pixels are divided by 16 so they sit in 0-1, labels are one-hot, and the split is 70/30 stratified with `random_state=42`, which gives 1257 train and 540 test. One thing I only noticed later: four pixels (0, 32, 39, 56) are zero in every training image, so their first-layer weights get zero gradient forever. Not a bug, just how the data is.

## 2. Network

64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax, with W = 16 and W = 64. Batches are rows of `X`, so each layer is `Z = A @ W + b`.

| Parameter | Shape | W=16 | W=64 |
|---|---|---|---|
| W1 | (64, W) | 1024 | 4096 |
| b1 | (W,) | 16 | 64 |
| W2 | (W, W) | 256 | 4096 |
| b2 | (W,) | 16 | 64 |
| W3 | (W, 10) | 160 | 640 |
| b3 | (10,) | 10 | 10 |
| Total | | 1482 | 8970 |

Weights come from N(0, 0.01) and biases start at zero. The manual doesn't say whether 0.01 is the variance or the standard deviation. I went with the usual N(mean, variance) reading, so variance 0.01, i.e. std 0.1, for all the main results. Section 6 shows what happens with std 0.01 instead (spoiler: mostly nothing learns).

## 3. Forward pass and loss

```
Z1 = X  W1 + b1     A1 = max(Z1, 0)
Z2 = A1 W2 + b2     A2 = max(Z2, 0)
Z3 = A2 W3 + b3     P  = softmax(Z3)   (row-wise, max subtracted for stability)
```

The tests check that the output is (batch, 10) and that each row sums to 1 within 1e-6, for both widths. Loss is the batch mean of `-sum_k Y_k log P_k`. Before training, the loss is within 0.05 of ln 10 = 2.3026 for both widths, which is what the manual says it should be.

## 4. Gradients and the gradient check

For a batch of size B the loss is `L = -(1/B) sum_i sum_k Y_ik log P_ik`.

Softmax plus cross-entropy is the nice case: for one sample `dL/dZ3_k = P_k - Y_k`, because the exponential terms from the softmax derivative cancel against the log. Averaging over the batch gives `dZ3 = (P - Y) / B`.

Then it's the usual chain. Since `Z3 = A2 W3 + b3`:
`dW3 = A2^T dZ3`, `db3 = sum over rows of dZ3`, `dA2 = dZ3 W3^T`.

ReLU has derivative 1 where `Z > 0` and 0 otherwise (I just use 0 at exactly `Z = 0`), so `dZ2 = dA2 * (Z2 > 0)`.

Layers 2 and 1 repeat the same thing:
`dW2 = A1^T dZ2`, `db2 = sum rows of dZ2`, `dA1 = dZ2 W2^T`, `dZ1 = dA1 * (Z1 > 0)`, `dW1 = X^T dZ1`, `db1 = sum rows of dZ1`.

This is all in `MLP.backward`, and every gradient has the same shape as its parameter.

**Numerical check.** On the full training set, in float64, I pick 10 distinct random entries from W1, W2, W3 (seed 42), nudge each by +-1e-5, and compute the central slope `(L(w+e) - L(w-e)) / 2e`. Relative error is `|a - n| / max(|a|, |n|, 1e-12)`. Full tables are in `results/gradient_check.csv`.

| Width | Entries with error < 1e-6 | Largest error | "At least 9 of 10" |
|---|---|---|---|
| 16 | 9 of 10 | 1.8e-4 | met |
| 64 | 8 of 10 | 5.1e-6 | **not met** |

So, honestly: W=16 passes the manual's criterion and W=64 doesn't, at least with this seed. Some context on why I don't think that means backprop is wrong:

- Two of the W=16 passes are trivial. W1[39,5] and W1[8,7] have gradient exactly 0 in both methods. Pixel 39 is always 0, and pixel 8 is non-zero in only 6 training images, none of which switch on that hidden unit. Leave those out and it's 7 of 8.
- The W=16 miss is W1[18,1] (gradient about 1.07e-4). Its error goes from 1.8e-1 at step 1e-3, to 1.8e-4 at 1e-5, to 1.7e-6 at 1e-6. The numerical slope is converging to the analytic one as the step shrinks, which is what a correct gradient looks like.
- One of the W=64 misses is W1[12,19] (gradient about 1.4e-6). Here the error is 1.2e-7 at step 1e-3 and gets *worse* as the step shrinks (5.1e-6 at 1e-5). That's floating-point rounding on a very small gradient.
- The other W=64 miss is W2[32,13], gradient about 6.8e-6, error 3.4e-6. It's the same kind of tiny-gradient situation and I'd bet on the same explanation, but I didn't vary the step for this one, so that's a guess.
- The criterion is also fairly fragile for random draws. Over 200 different draws of 10 weights, "at least 9 of 10 below 1e-6" held 129 times for W=16 and 151 times for W=64 (about 65% and 76%). If I only count entries with gradient above 1e-4, all of them were below 1e-6 in 169 and 182 of the 200 draws. The seed was fixed to 42 before I ran anything, I didn't shop around for a nice one. Counts are in `results/gradient_check_robustness.txt`.

## 5. Training

Plain SGD, no momentum, batch size 32, reshuffled each epoch, 50 epochs, seed 42, learning rates 0.001 / 0.01 / 0.1, widths 16 and 64. Per-epoch numbers are in `results/training_log.csv` and the curves (training loss, log scale) are in `results/loss_curves.png`.

| Width | LR | Final train loss | Final train acc | Final test acc |
|---|---|---|---|---|
| 16 | 0.001 | 2.2976 | 0.198 | 0.1778 |
| 16 | 0.01 | 1.4065 | 0.657 | 0.6556 |
| 16 | 0.1 | 0.0334 | 0.992 | 0.9648 |
| 64 | 0.001 | 2.2291 | 0.372 | 0.3574 |
| 64 | 0.01 | 0.3002 | 0.932 | 0.9130 |
| 64 | 0.1 | 0.0125 | 0.999 | 0.9685 |

What I take from this:

- Learning rate matters more than width here. At lr 0.001 neither network gets off the ln 10 plateau in 50 epochs, while at lr 0.1 both end up around 0.97 test accuracy.
- Width helps most when learning is slow. At lr 0.01, W=64 gets 0.913 test accuracy and W=16 only 0.656. At lr 0.1 the two are 0.4 points apart (0.9648 vs 0.9685), which another seed or split could easily flip.
- The lr 0.1 curves are noisy. There's a spike around epoch 21-22 in both widths, so the final-epoch number depends a bit on where in the noise training happens to stop. The train/test gap (0.992 vs 0.965, 0.999 vs 0.969) suggests mild overfitting.
- Everything is one seed. I didn't average over seeds.

## 6. Initialisation sensitivity

With the other reading of N(0, 0.01), std 0.01, the signal shrinks at every layer and the gradients are tiny, so the networks sit on the ln 10 plateau for a long time (`results/loss_curves_std0.01.png`, `results/final_accuracy_std0.01.csv`):

| Width | lr 0.001 | lr 0.01 | lr 0.1 |
|---|---|---|---|
| 16, test acc | 0.1019 | 0.1019 | 0.3259 |
| 64, test acc | 0.1019 | 0.1019 | 0.9389 |

Only W=64 at lr 0.1 really learns, and even that one only starts dropping around epoch 20. That's why I went with the variance reading for the main results. If your course means std 0.01, set `init_std=0.01` in `run_lab02.py` and use the numbers in this table instead.

## 7. Timing

Median of 100 runs after 10 warm-ups, using `time.perf_counter_ns`, with forward and backward timed separately (backward reuses the cache from the forward pass). The matmul-only columns time just the matrix products at the same shapes, to see how much of each pass they explain. Raw numbers are in `results/timing.csv`.

| Width | Batch | Forward (us) | Backward (us) | Bwd / fwd | Matmul share of fwd | Matmul share of bwd |
|---|---|---|---|---|---|---|
| 16 | 32 | 52.8 | 60.7 | 1.15 | 32% | 43% |
| 16 | 1257 | 505.4 | 495.2 | 0.98 | 36% | 59% |
| 64 | 32 | 70.4 | 93.0 | 1.32 | 45% | 58% |
| 64 | 1257 | 1670.5 | 2245.6 | 1.34 | 45% | 37% |

**Which is bigger?** Mostly the backward pass, but not always. It's slower in three of the four configurations (by about 1.15-1.34x), and for W=16 with the full batch the two are basically tied (0.98). The expected reason is that the forward pass has 3 matrix products and the backward pass has 5 (`dW3, dA2, dW2, dA1, dW1`; the input gradient of layer 1 is never needed). In FLOPs that's 1.29x for W=16 and 1.54x for W=64 (34.1 vs 22.2 MFLOP at W=64, batch 1257). The matmul-only time ratios are 1.57, 1.58, 1.68 and 1.10 for the four rows, so they mostly follow the FLOP story, except the last row where the matmul-only backward is barely slower than forward. I don't have a good explanation for that one.

**Caveats.** Matmuls are the biggest single cost but not the whole thing: they're only about 32-59% of measured time here. The rest is elementwise work (ReLU, exp and normalisation in softmax, masks, bias adds), temporary arrays, and Python/NumPy call overhead, which hurts most for tiny matrices. The measurements are also quite noisy on this shared 2-core cloud VM. The same W=64, batch 1257 forward pass came out around 1771 us in one earlier run and 1670 us in this one, and the W=16, batch 32 numbers have moved around a lot between runs too. I'd only trust the broad picture (backward usually above forward, cost rising with width and batch size), not the exact figures.

## 8. Conclusions

1. The hand-derived gradients look right to me, but the check is a bit messy. W=16 passes the manual's 9-of-10 criterion and W=64 gets 8 of 10. Every miss I looked into behaves like finite-difference error (it shrinks or grows with the step the way rounding or truncation should), and the one I didn't look into has a tiny gradient like the others. The check is also statistical: a different random draw of 10 weights fails the criterion roughly 25-36% of the time, more often for W=16.
2. Learning rate and initialisation scale decide whether the network trains at all in 50 epochs. Width helps mainly when the learning rate is too small to converge.
3. Cost is dominated by matrix products, and the backward pass costs roughly as much as or somewhat more than the forward pass (up to about 1.3x) because it needs five products instead of three. For small layers a big chunk is non-matmul overhead.

## Caveats

- One seed and one split throughout.
- Timings are from a shared VM and wander between runs.
- "N(0, 0.01)" is ambiguous and I picked variance (see sections 2 and 6).
