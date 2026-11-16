"""Recalibration method C/D sanity on synthetic data generated from method C's own measurement model."""
import numpy as np

from til.phase4_estimators import _basis, fit_reduced, play_operator, reduced_features, reduced_predict
from til.two_tendon import ARM, HOME


def test_play_operator_is_causal_and_direction_dependent():
    x = np.r_[np.linspace(0, 1e-3, 50), np.linspace(1e-3, 0, 50)]
    s = play_operator(x, 2e-4)
    assert s[0] == 0 and s[49] == 1. and s[-1] == -1.
    assert np.all(np.abs(s) <= 1)
    assert np.allclose(play_operator(x[:60], 2e-4), s[:60])  # causal: a prefix gives the same states


def synthetic_run(rng, coef, width, split="train"):
    t = np.arange(800)*.005
    q = HOME + np.column_stack([.25*np.sin(.9*t)**2, .3*np.sin(.6*t + .4)**2])
    run = dict(id=f"syn{rng.integers(1e9)}", split=split, time=t, y=q.astype(np.float32), x=np.zeros((len(t), 8), np.float32))
    # motor currents encode a constant 10 N proxy tension so the compliance feature is zero
    run["x"][:, 2] = run["x"][:, 6] = .02*10/.1
    x = np.zeros((len(t), 2))
    for _ in range(8):  # fixed point: the play state depends on x itself
        run["x"][:, [0, 4]] = (x/.02).astype(np.float32)
        _, force, s = reduced_features(run, width)
        x = np.column_stack([_basis(q - HOME) @ c[:5] + force[:, j]*c[5] + s[:, j]*c[6] + c[7] for j, c in enumerate(coef)])
    run["x"][:, [0, 4]] = (x/.02).astype(np.float32)
    return run


def test_reduced_model_recovers_its_own_generating_model():
    rng = np.random.default_rng(0)
    coef = np.array([[ARM[0, 0], 0., .002, 0., 0., 2e-5, 1e-4, 0.], [ARM[1, 0], ARM[1, 1], 0., .001, -.002, 2e-5, -1e-4, 0.]])
    runs = [synthetic_run(rng, coef, 3e-4) for _ in range(3)]
    cal = fit_reduced(runs, 3e-4)
    test = synthetic_run(rng, coef, 3e-4, "test")
    q = reduced_predict(test, cal)
    assert np.max(abs(np.degrees(q - test["y"]))) < .1  # float32 motor storage limits the residual
