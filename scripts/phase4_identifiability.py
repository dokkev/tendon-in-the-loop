"""Finite-horizon input-output sensitivity of the motor channels to hidden plant parameters (P2, noise-free).

For each hidden parameter p_i a central difference of the simulated eight motor channels and of the true joint
trajectory is taken on one matched task. Motor sensitivities are normalised by the P0 telemetry noise (so 1 = one
noise standard deviation per sample, RMS over the horizon). The weakest direction of the normalised motor
sensitivity matrix is then simulated as a combined perturbation, giving a candidate pair of distinct physical
configurations whose motor observations differ little while the joints differ. This is a local, finite search on
one task: it is not a global observability claim.
"""
import argparse, copy, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from til.phase4_experiment import specification, simulate
from til.phase4_hil import Fidelity

# name: (how to apply a relative/absolute step, step)
PARAMS = {
    "modulus": ("material", "modulus_pa", 1e10, .05),
    "relaxation_time": ("material", "relaxation_time_s", .5, .2),
    "maxwell_ratio": ("material", "maxwell_ratio", .35, .2),
    "guide_mu": ("contacts", "sliding.mu", .05, .2),
    "presliding_stiffness": ("contacts", "all.presliding_stiffness", 5000., .3),
    "bearing_friction": ("contacts", "rolling.bearing_mu", .002, .5),
    "conditioning_d": ("history", "d", .15, .1),  # absolute step around a mid-aged specimen
    "maxwell_preload": ("history", "maxwell_fraction", .85, .1),  # absolute step
    "idler_radius": ("route", "idler_radius", .012, .1),
    "virtual_guide_height": ("virtual_route", "guide_height", .0025, .1),
}
NOISE = np.array([2*np.pi/16384/np.sqrt(12), .002, np.hypot(.005, .005), 1e-3]*2)  # rad, rad/s, A, A (command: 1 mA floor)


def fidelity(values):
    f = Fidelity("ident")
    for name, value in values.items():
        kind, key, nominal, _ = PARAMS[name]
        if kind == "material":
            f.material[key] = value
        elif kind == "contacts":
            group, field = key.split(".")
            for g in (["sliding", "rolling"] if group == "all" else [group]):
                f.contacts.setdefault(g, {})[field] = value
        elif kind == "history":
            for h in f.initial_history:
                h[key] = value
        elif kind == "route":
            f.route[key] = value
        else:
            f.virtual_route[key] = value
    return f


BASE = {n: p[2] for n, p in PARAMS.items()}


def perturbed(name, sign):
    kind, key, nominal, step = PARAMS[name]
    value = nominal + sign*step if kind == "history" else nominal*(1 + sign*step)
    return dict(BASE, **{name: value})


def _run(job):
    label, values, spec = job
    summary, a = simulate(spec, fidelity(values), noise=False)
    return label, a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT/"figures/phase4/identifiability.json"))
    ap.add_argument("--workers", type=int, default=14); a = ap.parse_args()
    spec = specification(0, "test", "coordinated", contact=True, cycles=2, fidelity="P2", tag="_ident")
    jobs = [("nominal", dict(BASE), spec)] + [(f"{n}{'+-'[i]}", perturbed(n, s), spec) for n in PARAMS for i, s in enumerate((1, -1))]
    start = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = dict(ex.map(_run, jobs))
    base = res["nominal"]
    obs = lambda a: a[:, 5:13]
    joints = lambda a: a[:, 1:3]
    sens, rows = [], {}
    for n, (kind, key, nominal, step) in PARAMS.items():
        dobs = (obs(res[f"{n}+"]) - obs(res[f"{n}-"]))/2  # per one step of the parameter
        dq = (joints(res[f"{n}+"]) - joints(res[f"{n}-"]))/2
        normalized = dobs/NOISE
        sens.append(normalized.ravel())
        rows[n] = dict(step=step, step_kind="absolute" if kind == "history" else "relative",
                       motor_rms_in_noise_sd=float(np.sqrt(np.mean(normalized**2))),
                       motor_rms_per_channel_in_noise_sd=np.sqrt(np.mean(normalized**2, 0)).tolist(),
                       joint_rms_deg=np.degrees(np.sqrt(np.mean(dq**2, 0))).tolist(),
                       joint_max_deg=np.degrees(np.max(abs(dq), 0)).tolist())
    S = np.array(sens).T  # samples*channels x params
    U, sv, Vt = np.linalg.svd(S/np.sqrt(len(base)), full_matrices=False)
    weakest = Vt[-1]/np.max(abs(Vt[-1]))
    # Simulate the weakest combined direction at one step size
    values = dict(BASE)
    for c, n in zip(weakest, PARAMS):
        kind, key, nominal, step = PARAMS[n]
        values[n] = nominal + c*step if kind == "history" else nominal*(1 + c*step)
    _, comb = _run(("weakest", values, spec))
    combined = dict(direction=dict(zip(PARAMS, weakest.tolist())), values=values,
                    motor_rms_in_noise_sd=float(np.sqrt(np.mean(((obs(comb) - obs(base))/NOISE)**2))),
                    joint_rms_deg=np.degrees(np.sqrt(np.mean((joints(comb) - joints(base))**2, 0))).tolist(),
                    joint_max_deg=np.degrees(np.max(abs(joints(comb) - joints(base)), 0)).tolist())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(task=spec, noise_sd=NOISE.tolist(), parameters=rows, singular_values=sv.tolist(),
        right_singular_vectors=Vt.tolist(), parameter_order=list(PARAMS), weakest_combined=combined, seconds=time.time() - start,
        scope="Local finite-difference sensitivity on one noise-free P2 task; not a global observability result."),
        indent=1, default=float) + "\n", encoding="utf8")
    for n, r in rows.items():
        print(f"{n:22s} motor {r['motor_rms_in_noise_sd']:.3f} sd  joint rms {np.round(r['joint_rms_deg'], 4)} deg")
    print("weakest", combined)


if __name__ == "__main__":
    main()
