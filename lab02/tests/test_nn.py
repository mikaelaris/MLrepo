import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nn import MLP, gradient_check, load_data  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.mark.parametrize("W", [16, 64])
def test_param_shapes(W):
    p = MLP(W).params
    assert p["W1"].shape == (64, W) and p["b1"].shape == (W,)
    assert p["W2"].shape == (W, W) and p["b2"].shape == (W,)
    assert p["W3"].shape == (W, 10) and p["b3"].shape == (10,)


@pytest.mark.parametrize("W", [16, 64])
def test_forward_shape_and_rows_sum_to_one(W, data):
    X = data[0][:50]
    P, _ = MLP(W).forward(X)
    assert P.shape == (50, 10)
    assert np.all(np.abs(P.sum(axis=1) - 1) < 1e-6)


def test_split_and_scaling(data):
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = data
    assert len(X_tr) + len(X_te) == 1797 and len(X_te) == 540
    assert X_tr.max() <= 1.0 and Y_tr.sum(axis=1).min() == 1


@pytest.mark.parametrize("W", [16, 64])
def test_untrained_loss_near_ln10(W, data):
    X_tr, _, Y_tr, *_ = data
    assert abs(MLP(W).loss(X_tr, Y_tr) - np.log(10)) < 0.05


@pytest.mark.parametrize("W", [16, 64])
def test_gradient_check(W, data):
    X_tr, _, Y_tr, *_ = data
    rows = gradient_check(MLP(W), X_tr, Y_tr)
    assert sum(r["rel_error"] < 1e-6 for r in rows) >= 9
