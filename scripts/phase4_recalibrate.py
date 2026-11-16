"""Milestone H: controlled recalibration (methods A-E) on P3 specimen families, by calibration budget.

Stage 'develop' fits every method on each budget of calibration (train) runs and scores the validation
specimens only; it freezes the hyperparameters (play width for C/D, GRU checkpoint by validation loss) and
writes a selection file with hashes. Stage 'test' refuses to run without that file, re-fits with the frozen
choices and scores the sealed test specimens once.
"""
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from til.phase4_estimators import (ReducedEKF, aggregate, corrected_A, design_estimates, fit_compliance_A, fit_reduced,
                                   fit_reduced_ekf, frozen_estimates, load_frozen, reduced_predict, rehomed, rigid_A, run_A, train_gru)
from til.two_tendon_observer import PhysicsObserver, corrected, fit_compliance, load_runs, predict, rigid, scores

BUDGETS = [2, 4, 8, 16, 24]
WIDTHS = [1e-4, 3e-4, 1e-3]
REGIMES = {
    "frozen-rigid": "none (geometry only)", "frozen-compliance": "P0 calibration, frozen",
    "frozen-physics": "P0 calibration, frozen", "frozen-gru": "P0 weights, frozen",
    "frozen-compliance+rehome": "P0 calibration + one-time known-home re-reference",
    "frozen-compliance-designA": "P0 calibration numbers, rig design winding matrix",
    "frozen-physics-designA": "P0 calibration numbers, rig design winding matrix",
    "A-rigid": "none (rig design winding matrix)", "A-rigid+rehome": "design matrix + one-time known-home re-reference",
    "B-compliance": "privileged offline (joint labels on calibration runs)",
    "B-compliance+rehome": "privileged offline + one-time known-home re-reference",
    "C-reduced": "privileged offline (joint labels on calibration runs)",
    "D-ekf": "privileged offline (joint labels on calibration runs, covariances from the same runs)",
    "E-gru": "privileged offline supervised training (joint labels), validation specimens for checkpoint selection",
}


def load(root, split):
    return [r for p in sorted((root/split).glob("*.h5")) for r in load_runs(str(p))]


def ordered(train):
    """Round-robin over specimens (session 0 of every specimen first), so small budgets span specimens."""
    by = {}
    for r in train:
        by.setdefault(r["config"]["specimen_id"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["config"]["session_index"])
    out = []
    for s in range(max(len(v) for v in by.values())):
        out += [v[s] for v in by.values() if s < len(v)]
    return out


def fit_methods(cal_runs, val_runs, width, seeds=(0, 1, 2), epochs=60):
    A = run_A(cal_runs[0])
    if any(not np.allclose(run_A(r), A) for r in cal_runs + val_runs):
        raise ValueError("calibration and validation runs must share one rig design")
    fitted = dict(A=A.tolist(), B=fit_compliance_A(cal_runs, A), C=fit_reduced(cal_runs, width), D=fit_reduced_ekf(cal_runs, width))
    with ProcessPoolExecutor(len(seeds)) as ex:
        grus = list(ex.map(_gru, [(cal_runs + val_runs, s, epochs) for s in seeds]))
    fitted["E"] = grus
    return fitted


def _gru(args):
    runs, seed, epochs = args
    model, scaler, info = train_gru(runs, seed, epochs)
    return dict(state={k: v.tolist() for k, v in model.state_dict().items()}, scaler={k: v.tolist() for k, v in scaler.items()}, info=info)


def _model(g):
    from til.two_tendon_observer import TwoMotorObserver
    m = TwoMotorObserver(); m.load_state_dict({k: torch.tensor(v) for k, v in g["state"].items()})
    return m, {k: np.array(v, np.float32) for k, v in g["scaler"].items()}


def estimate(run, fitted, frozen):
    cal0, models0 = frozen
    out = {}
    f = frozen_estimates(run, cal0, models0)
    out["frozen-rigid"], out["frozen-compliance"], out["frozen-physics"] = f["rigid"], f["compliance"], f["physics"]
    out["frozen-gru"] = np.mean([f[f"gru-seed{s}"] for s in range(3)], 0)
    out["frozen-compliance+rehome"] = corrected(rehomed(run), cal0)
    d = design_estimates(run, cal0)
    out["frozen-compliance-designA"], out["frozen-physics-designA"] = d["compliance-designA"], d["physics-designA"]
    out["A-rigid"] = rigid_A(run); out["A-rigid+rehome"] = rigid_A(rehomed(run))
    if fitted:
        out["B-compliance"] = corrected_A(run, fitted["B"]); out["B-compliance+rehome"] = corrected_A(rehomed(run), fitted["B"])
        out["C-reduced"] = reduced_predict(run, fitted["C"])
        out["D-ekf"] = ReducedEKF(fitted["D"]).predict(run)
        preds = [predict(*_model(g), run) for g in fitted["E"]]
        for s, p in enumerate(preds):
            out[f"E-gru-seed{s}"] = p
        out["E-gru"] = np.mean(preds, 0)
    return out


def _score_run(args):
    run, fitted, = args
    frozen = load_frozen()
    est = estimate(run, fitted, frozen)
    return run["id"], {m: scores(run, e) for m, e in est.items()}


def score_runs(runs, fitted, workers):
    with ProcessPoolExecutor(workers) as ex:
        results = dict(ex.map(_score_run, [(r, fitted) for r in runs]))
    methods = list(next(iter(results.values())).keys())
    return dict(per_run=results, aggregate={m: aggregate([results[r["id"]][m] for r in runs]) for m in methods})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["develop", "test"], required=True)
    ap.add_argument("--root", default=str(ROOT/"data/phase4/full/recal")); ap.add_argument("--out", default=str(ROOT/"figures/phase4/recal"))
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--epochs", type=int, default=60); a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train, val = ordered(load(root, "train")), load(root, "validation")
    selection_path = out/"selection.json"
    if a.stage == "develop":
        start = time.time(); results = {}
        # Hyperparameter: play width for C/D, chosen by validation q MAE at the full budget.
        width_scores = {}
        for w in WIDTHS:
            c = fit_reduced(train, w)
            width_scores[str(w)] = float(np.mean([np.mean(abs(reduced_predict(r, c) - r["y"])) for r in val]))
        width = float(min(width_scores, key=width_scores.get))
        for n in BUDGETS:
            cal = train[:n]
            fitted = fit_methods(cal, val, width, epochs=a.epochs)
            res = score_runs(val, fitted, a.workers)
            results[str(n)] = dict(calibration_ids=[r["id"] for r in cal], aggregate=res["aggregate"],
                                   gru=[g["info"] for g in fitted["E"]], C=fitted["C"], B=fitted["B"])
            print("budget", n, {m: round(v["all"]["q1_mae_deg"], 3) for m, v in res["aggregate"].items()}, flush=True)
        selection = dict(play_width_m=width, width_validation_mae_rad=width_scores, budgets=BUDGETS, epochs=a.epochs,
                         gru_selection="validation-loss checkpoint per seed", validation_ids=[r["id"] for r in val],
                         frozen_before_test=time.strftime("%Y-%m-%dT%H:%M:%S"))
        selection["sha256"] = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
        selection_path.write_text(json.dumps(selection, indent=1) + "\n", encoding="utf8")
        (out/"develop.json").write_text(json.dumps(dict(selection=selection, regimes=REGIMES, results=results, seconds=time.time() - start),
                                                   indent=1, default=float) + "\n", encoding="utf8")
        return
    if not selection_path.exists():
        raise SystemExit("sealed test requires the frozen selection from the develop stage")
    selection = json.loads(selection_path.read_text())
    test = load(root, "test"); start = time.time(); results = {}
    for n in selection["budgets"]:
        cal = train[:n]
        fitted = fit_methods(cal, val, selection["play_width_m"], epochs=selection["epochs"])
        res = score_runs(test, fitted, a.workers)
        results[str(n)] = dict(calibration_ids=[r["id"] for r in cal], aggregate=res["aggregate"], per_run=res["per_run"],
                               gru=[g["info"] for g in fitted["E"]], C=fitted["C"], D={k: v for k, v in fitted["D"].items()}, B=fitted["B"])
        print("budget", n, {m: round(v["all"]["q1_mae_deg"], 3) for m, v in res["aggregate"].items()}, flush=True)
    (out/"test.json").write_text(json.dumps(dict(selection=selection, regimes=REGIMES, results=results, test_ids=[r["id"] for r in test],
        test_specimens=sorted({r["config"]["specimen_id"] for r in test}), seconds=time.time() - start), indent=1, default=float) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
