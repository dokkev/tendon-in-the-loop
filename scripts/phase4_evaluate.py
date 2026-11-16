"""Milestone G: frozen P0 estimators (rigid, compliance, physics EKF, GRU x3) on Phase 4 datasets, no refitting."""
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from til.phase4_estimators import FROZEN_CALIBRATION, FROZEN_MODELS, aggregate, design_estimates, frozen_estimates, load_frozen
from til.two_tendon_observer import load_runs, scores, tension

METHODS = ["rigid", "compliance", "physics", "gru-seed0", "gru-seed1", "gru-seed2", "rigid-designA", "compliance-designA", "physics-designA"]


def privileged(path, run_id):
    with h5py.File(path) as f:
        g = f["episodes"][run_id]
        tout = np.column_stack([g[f"ground_truth{i}/output_tension_true"][::5] for i in (1, 2)])
        tin = np.column_stack([g[f"ground_truth{i}/input_tension_true"][::5] for i in (1, 2)])
        current = np.column_stack([g[f"ground_truth{i}/motor_current_true"][::5] for i in (1, 2)])
    return tin, tout, current


def evaluate_file(path):
    cal, models = load_frozen()
    rows = []
    for run in load_runs(path):
        est = frozen_estimates(run, cal, models); est.update(design_estimates(run, cal))
        tin, tout, current = privileged(path, run["id"])
        ti_hat, to_hat = tension(run)
        spec = run["config"]
        row = dict(id=run["id"], file=str(Path(path).relative_to(ROOT)), variant=spec.get("variant", spec.get("fidelity_level")),
                   task=spec.get("task"), level=spec.get("fidelity_level"), quality_flags=run["quality"]["flags"],
                   scores={m: scores(run, e) for m, e in est.items()},
                   distal_tension_proxy_rmse_N=np.sqrt(np.mean((to_hat - tout)**2, 0)).tolist(),
                   input_tension_proxy_rmse_N=np.sqrt(np.mean((ti_hat - tin)**2, 0)).tolist(),
                   motor_current_rms_A=np.sqrt(np.mean(current**2, 0)).tolist(),
                   peak_true_tension_N=np.max(tout, 0).tolist())
        rows.append(row)
    return path, rows


def summarize(rows):
    variants = sorted({r["variant"] for r in rows})
    out = {}
    for v in variants:
        sub = [r for r in rows if r["variant"] == v]
        out[v] = dict(methods={m: aggregate([r["scores"][m] for r in sub]) for m in METHODS},
                      gru_mean={mask: {k: float(np.mean([out_[mask][k] for out_ in [aggregate([r["scores"][f"gru-seed{s}"] for r in sub]) for s in range(3)]]))
                                       for k in aggregate([r["scores"]["gru-seed0"] for r in sub])[mask]}
                                for mask in aggregate([r["scores"]["gru-seed0"] for r in sub])},
                      distal_tension_proxy_rmse_N=np.mean([r["distal_tension_proxy_rmse_N"] for r in sub], 0).tolist(),
                      motor_current_rms_A=np.mean([r["motor_current_rms_A"] for r in sub], 0).tolist(),
                      runs=len(sub), flagged_runs=sum(bool(r["quality_flags"]) for r in sub),
                      by_task={t: {m: aggregate([r["scores"][m] for r in sub if r["task"] == t]) for m in ("rigid", "compliance", "physics", "gru-seed0")}
                               for t in sorted({r["task"] for r in sub})})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--scale", default="full")
    ap.add_argument("--root", default=str(ROOT/"data/phase4")); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
    files = sorted(str(p) for p in (Path(a.root)/a.scale/a.set).rglob("*.h5"))
    if not files:
        raise FileNotFoundError("no datasets")
    start = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        results = dict(ex.map(evaluate_file, files))
    rows = [r for f in files for r in results[f]]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    frozen = dict(calibration=str(FROZEN_CALIBRATION.relative_to(ROOT)), calibration_sha256=hashlib.sha256(FROZEN_CALIBRATION.read_bytes()).hexdigest(),
                  checkpoints={s: hashlib.sha256((FROZEN_MODELS/f"seed{s}/checkpoint.pt").read_bytes()).hexdigest() for s in range(3)})
    out.write_text(json.dumps(dict(set=a.set, scale=a.scale, frozen_artifacts=frozen, summary=summarize(rows), runs=rows,
        dataset_sha256={str(Path(f).relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        scope="Frozen P0 estimators, eight motor channels at 200 Hz; joints, tensions and currents used for scoring only.",
        seconds=time.time() - start), indent=1, default=float) + "\n", encoding="utf8")
    for v, s in summarize(rows).items():
        print(v, {m: round(s["methods"][m]["all"]["q1_mae_deg"], 3) for m in ("rigid", "compliance", "physics")},
              "gru", round(s["gru_mean"]["all"]["q1_mae_deg"], 3), "| q2", {m: round(s["methods"][m]["all"]["q2_mae_deg"], 3) for m in ("rigid", "compliance", "physics")},
              round(s["gru_mean"]["all"]["q2_mae_deg"], 3))


if __name__ == "__main__":
    main()
