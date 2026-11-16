"""Phase 4 datasets (two-channel schema), one HDF5 file per episode or specimen history, generated in parallel.

Sets:
  frozen      matched tasks on P0-P3 (frozen estimator transfer)
  ablation    one-mechanism variants on the same tasks
  robustness  one-factor sweeps around P2
  recal       P3 specimen families with persistent multi-session histories (train / validation / sealed test)
Scales: pilot (one task index, subset) before full.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from til.phase4_experiment import specification
from til.phase4_hardware import hardware_preset
from til.phase4_hil import preset
from til.phase4_study import ABLATIONS, ROBUSTNESS, TASKS, run_group, run_specimen_sessions, specimen_draw, task_name


def episode_jobs(set_name, scale, root):
    indices = [0] if scale == "pilot" else [0, 1]
    jobs = []
    if set_name == "frozen":
        variants = {"P0": (preset("P0"), "nominal"), **{lvl: (hardware_preset(lvl), "nominal") for lvl in ("H1", "H2", "H3")}}
    elif set_name == "ablation":
        variants = {k: (v, "nominal") for k, v in ABLATIONS.items() if k not in ("P0", "H1", "H2", "H3")}  # those come from 'frozen'
        if scale == "pilot":
            variants = {k: variants[k] for k in ("H1_coaxial", "H2_sliding_guides")}
    elif set_name == "robustness":
        variants = ROBUSTNESS
        indices = [0]
    else:
        raise ValueError(set_name)
    tasks = TASKS if scale == "full" else TASKS[2:5]
    for name, (fid, condition) in variants.items():
        for i in indices:
            for k, (motion, contact, cycles) in enumerate(tasks):
                spec = specification(i, "test", motion, contact, cycles, condition=condition, fidelity=fid.level,
                                     tag=f"_{set_name}_{name}_{task_name(motion, contact, cycles)}", seed_offset=100*k)
                spec["variant"] = name; spec["task"] = task_name(motion, contact, cycles)
                if condition in ("zero", "pretension"):
                    spec["condition"] = condition
                path = root/set_name/name/f"{spec['id']}.h5"
                jobs.append((str(path), [(spec, fid, {})]))
    return jobs


def recal_jobs(scale, root):
    counts = dict(train=8, validation=4, test=6) if scale == "full" else dict(train=2, validation=1, test=1)
    sessions = 3 if scale == "full" else 2
    jobs = []
    for split, n in counts.items():
        rng = np.random.default_rng({"train": 7101, "validation": 7202, "test": 7303}[split])
        for j in range(n):
            specimen = specimen_draw(f"{split[:3]}{j:02d}", rng)
            jobs.append((str(root/"recal"/split/f"{specimen['id']}.h5"), specimen, split, sessions, j*sessions))
    return jobs


def _episode(job):
    path, items = job
    if Path(path).exists():
        return path, "exists"
    t = time.time()
    m = run_group(path + ".part", items)
    Path(path + ".part").rename(path)
    return path, dict(manifest=m, seconds=time.time() - t)


def _specimen(job):
    path, specimen, split, sessions, start = job
    if Path(path).exists():
        return path, "exists"
    t = time.time()
    m = run_specimen_sessions(path + ".part", specimen, split, sessions, start)
    Path(path + ".part").rename(path)
    return path, dict(manifest=m, seconds=time.time() - t, specimen=specimen)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True, choices=["frozen", "ablation", "robustness", "recal"])
    ap.add_argument("--scale", choices=["pilot", "full"], default="pilot"); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--root", default=str(ROOT/"data/phase4")); a = ap.parse_args()
    root = Path(a.root)/a.scale
    if a.set == "recal":
        jobs, fn = recal_jobs(a.scale, root), _specimen
    else:
        jobs, fn = episode_jobs(a.set, a.scale, root), _episode
    for p in root.rglob("*.part"):
        p.unlink()
    manifest = {}
    start = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futures = [ex.submit(fn, j) for j in jobs]
        for f in as_completed(futures):
            path, info = f.result()
            manifest[str(Path(path).relative_to(root))] = info
            print(len(manifest), "/", len(jobs), Path(path).name, "" if info == "exists" else info.get("seconds"), flush=True)
    out = root/f"{a.set}.manifest.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update({k: v for k, v in manifest.items() if v != "exists"})
    out.write_text(json.dumps(old, indent=1, default=float) + "\n", encoding="utf8")
    print("done", a.set, a.scale, len(jobs), "jobs", time.time() - start, "s")


if __name__ == "__main__":
    main()
