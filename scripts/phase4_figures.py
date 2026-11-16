"""Phase 4 report figures. Every plotted value is computed here or read from Phase 4 result JSON / HDF5."""
import argparse, json, sys
from math import pi
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from til.phase4_cable import CableChainParams, CableMaterial, ContactParams, DistributedCablePlant
from til.phase4_geometry import hardware_routes, rotate
from til.two_tendon import ARM, HOME

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed categorical order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "font.size": 9, "legend.frameon": False})
FIG = ROOT/"figures/phase4"


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG/f"{name}.png", dpi=150); plt.close(fig); print(name)


def _draw_route(ax, route, q, color, label):
    e = route.evaluate(q)
    for (pa, pb) in e["tangent_points"]:
        ax.plot([pa[0]*1e3, pb[0]*1e3], [pa[1]*1e3, pb[1]*1e3], color=color, lw=1.5)
    for n, c in zip(route.nodes, e["centers"]):
        if n.radius:
            style = dict(rolling=("-", MUTED), sliding=("--", INK), joint=("-", INK))[n.kind]
            ax.add_patch(plt.Circle(c*1e3, n.radius*1e3, fill=n.kind == "joint", alpha=.15 if n.kind == "joint" else 1,
                                    color=style[1], ls=style[0], lw=1))
    ax.plot([], [], color=color, label=label)


def routing():
    t1, t2 = hardware_routes()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, route, title in ((axes[0], t1, "tendon 1: actuator - P1 (MCP, filled) - emulator"),
                             (axes[1], t2, "tendon 2: grounded idlers, MCP coaxial idler, P2 (PIP, filled)")):
        for q, color, label in ((-pi/6, C[0], "q1 = -30 deg"), (pi/3, C[1], "q1 = +60 deg")):
            _draw_route(ax, route, q, color, label)
            if route is t2:
                ax.plot([0, rotate((.05, 0.), q)[0]*1e3], [0, rotate((.05, 0.), q)[1]*1e3], color=MUTED, lw=5, alpha=.25)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=8); ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)"); ax.legend(fontsize=7)
    save(fig, "routing")


def jacobian():
    qs = np.linspace(-pi/6, pi/3, 91)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, (label, kw, c) in enumerate((("coaxial idler on the MCP axis", {}, C[0]), ("coaxial idler offset 3 mm", dict(coax_offset=(-.003, 0.)), C[1]))):
        t1, t2 = hardware_routes(**kw)
        act = np.array([t2.segment_lengths(q)[:3].sum() for q in qs])
        axes[0].plot(np.degrees(qs), (act - act[np.argmin(abs(qs - HOME[0]))])*1e3, color=c, label=label)
        axes[1].plot(np.degrees(qs), [-t2.segment_jacobian(q)[:3].sum()*1e3 for q in qs], color=c, label=label)
    axes[0].set_xlabel("q1 (deg)"); axes[0].set_ylabel("tendon 2 actuator-side path change (mm)"); axes[0].legend(fontsize=7)
    axes[1].axhline(ARM[1, 0]*1e3, color=MUTED, ls=":", lw=1); axes[1].set_xlabel("q1 (deg)"); axes[1].set_ylabel("tendon 2 MCP moment arm (mm)")
    axes[1].set_title("dotted: P0 constant A21", fontsize=8)
    save(fig, "jacobian")


def cable_mechanics():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    elastic = CableMaterial(nonlinear_strain_coefficient=0., maxwell_ratio=0., viscous_Ns=0.)
    for k, (amp, color) in enumerate(((2e-5, C[0]), (1e-3, C[1]))):
        d = DistributedCablePlant(CableChainParams(elastic, [ContactParams("guide", "sliding", radius=.01, wrap=pi/2, mu=.1)], (.2, .2), 10.))
        xs, T = [], []
        for n in range(1, 20001):
            t = n*1e-4; x = amp*np.sin(2*pi*t); d.step(1e-4, x, amp*2*pi*np.cos(2*pi*t), 0, 0); xs.append(x); T.append(d.T.copy())
        T = np.array(T); xs = np.array(xs)
        axes[0].plot(xs[10000:]*1e3, T[10000:, 1], color=color, label=f"input {amp*1e3:g} mm amplitude")
    axes[0].set_xlabel("actuator displacement (mm)"); axes[0].set_ylabel("output tension (N)")
    axes[0].set_title("guide presliding / sliding hysteresis", fontsize=8); axes[0].legend(fontsize=7)
    m = CableMaterial(nonlinear_strain_coefficient=0., viscous_Ns=0.)
    r = DistributedCablePlant(CableChainParams(m, [], (.4,), 10.)); T = []
    for n in range(20000):
        r.step(1e-4, .001, 0., 0., 0.); T.append(r.T[0])
    t = np.arange(1, 20001)*1e-4; k = r.k[0]; km = m.maxwell_ratio*k
    axes[1].plot(t, T, color=C[0], label="simulated"); axes[1].plot(t, k*r.e[0] + km*.001*np.exp(-t/.5), color=C[1], ls=":", label="analytic SLS")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("tension (N)"); axes[1].set_title("relaxation after a 1 mm step", fontsize=8); axes[1].legend(fontsize=7)
    m = CableMaterial(nonlinear_strain_coefficient=0., viscous_Ns=0., creep_strain_rate=2e-4)
    cr = DistributedCablePlant(CableChainParams(m, [], (.4,), 10.)); xs = []; a = 1e-4/.5
    for n in range(30000):
        e = (20 + km*cr.v[0]/(1 + a))/(k + km/(1 + a)); x = e - (cr.L[0] - cr.M[0] - cr.creep[0]); cr.step(1e-4, x, 0, 0, 0); xs.append(x)
    t = np.arange(30000)*1e-4; tc = .5*(k + km)/k
    an = 10/k - (10/k - 10/(k + km))*np.exp(-t/tc) + .4*2e-4*(8/18)**2*t
    axes[2].plot(t, np.array(xs)*1e3, color=C[0], label="simulated"); axes[2].plot(t, an*1e3, color=C[1], ls=":", label="analytic SLS + Norton")
    axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("elongation (mm)"); axes[2].set_title("creep under a constant 20 N", fontsize=8); axes[2].legend(fontsize=7)
    save(fig, "cable_mechanics")


def episode(path):
    with h5py.File(path) as f:
        g = next(iter(f["episodes"].values()))
        return {k: g[k][()] for k in ("time", "simulation/joint_positions", "ground_truth1/segment_tensions", "ground_truth2/segment_tensions",
                                       "ground_truth1/cable_segment_strains", "transmission1/input_tension", "transmission1/output_tension",
                                       "transmission2/input_tension", "transmission2/output_tension", "simulation/contact_force") if k in g}


def tensions(data_root):
    files = sorted((data_root/"frozen/H2").glob("*coordinated_contact_repeated*.h5")) or sorted((data_root/"frozen/H2").glob("*contact*.h5"))
    if not files:
        return
    e = episode(files[0]); t = e["time"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    names = ["actuator span", "grounded idler - coax", "coax - P2 (link)", "P2 - coax (link)", "coax - grounded idler", "emulator span"]
    for i in range(6):
        axes[0].plot(t, e["ground_truth2/segment_tensions"][:, i], color=C[i], lw=1.2, label=names[i])
    with h5py.File(files[0]) as fh:
        g = next(iter(fh["episodes"].values())); strain = g["ground_truth2/cable_segment_strains"][()]
    for i in range(6):
        axes[1].plot(t, strain[:, i]*1e3, color=C[i], lw=1.2)
    axes[0].set_ylabel("segment tension, tendon 2 (N)"); axes[0].legend(fontsize=6)
    axes[1].set_ylabel("segment strain, tendon 2 (x1e-3)")
    for j, c in ((1, C[0]), (2, C[1])):
        axes[2].plot(t, e[f"transmission{j}/input_tension"], color=c, lw=1.2, label=f"tendon {j} input")
        axes[2].plot(t, e[f"transmission{j}/output_tension"], color=c, lw=1.2, ls=":", label=f"tendon {j} output")
    axes[2].set_ylabel("measured tension (N)"); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("time (s)")
    save(fig, "tensions")


def energy(gates):
    g = json.loads(gates.read_text())
    per = g["cable"]["isolated_energy_residual"]["per_dt"]
    dts = np.array([float(k) for k in per]); res = np.array([abs(v["residual_J"]) for v in per.values()])
    num = np.array([v["numerical_dissipation_J"] for v in per.values()])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].loglog(dts, res, "o-", color=C[0], label="|ledger residual|")
    axes[0].loglog(dts, num, "s-", color=C[1], label="implicit-Euler inertia loss (reported term)")
    axes[0].set_xlabel("plant step (s)"); axes[0].set_ylabel("energy over 2 cycles (J)"); axes[0].legend(fontsize=7)
    ref = g["coupled"]["timestep_refinement_vs_quarter_step"]
    labels = list(ref); vals = [max(ref[k]["joint_rmse_deg"]) for k in labels]
    axes[1].barh(range(len(labels)), vals, color=C[0], height=.6)
    axes[1].set_yticks(range(len(labels))); axes[1].set_yticklabels(labels, fontsize=7)
    axes[1].set_xlabel("joint RMSE vs quarter-step reference (deg); gates 0.25 free / 0.5 contact")
    save(fig, "energy_convergence")


def frozen_predictions(data_root, frozen_json):
    from til.phase4_estimators import frozen_estimates, load_frozen
    from til.two_tendon_observer import load_runs
    cal, models = load_frozen()
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.2), sharex=True)
    for col, level in enumerate(("P0", "H1", "H2", "H3")):
        files = sorted((data_root/f"frozen/{level}").glob("*coordinated_contact_0*.h5"))
        if not files:
            continue
        run = load_runs(str(files[0]))[0]; est = frozen_estimates(run, cal, models)
        for row in range(2):
            ax = axes[row, col]
            ax.plot(run["time"], np.degrees(run["y"][:, row]), color=INK, lw=1.5, label="true")
            for m, c in (("rigid", C[0]), ("compliance", C[1]), ("physics", C[2]), ("gru-seed0", C[6])):
                ax.plot(run["time"], np.degrees(est[m][:, row]), color=c, lw=1, label=m)
            ax.set_title(f"{level}  q{row+1}", fontsize=8)
            if col == 0:
                ax.set_ylabel("deg")
    axes[0, 0].legend(fontsize=6); [a.set_xlabel("time (s)") for a in axes[1]]
    save(fig, "frozen_predictions")
    s = json.loads(frozen_json.read_text())["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    levels = [l for l in ("P0", "H1", "H2", "H3") if l in s]
    methods = [("rigid", "rigid"), ("compliance", "compliance"), ("physics", "physics EKF")]
    for j in range(2):
        ax = axes[j]; w = .2
        for k, (m, label) in enumerate(methods):
            ax.bar(np.arange(len(levels)) + (k - 1.5)*w, [s[l]["methods"][m]["all"][f"q{j+1}_mae_deg"] for l in levels], w*.9, color=C[k], label=label)
        ax.bar(np.arange(len(levels)) + 1.5*w, [s[l]["gru_mean"]["all"][f"q{j+1}_mae_deg"] for l in levels], w*.9, color=C[6], label="GRU (3 seeds)")
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels); ax.set_ylabel(f"q{j+1} MAE (deg)")
    axes[0].legend(fontsize=7)
    save(fig, "frozen_by_level")


def robustness(rob_json):
    s = json.loads(rob_json.read_text())["summary"]
    names = sorted(s, key=lambda v: (v != "nominal", v))
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names)); h = .2
    for k, (m, label, c) in enumerate((("rigid", "rigid", C[0]), ("compliance", "compliance", C[1]), ("physics", "physics EKF", C[2]))):
        ax.barh(y + (k - 1.5)*h, [max(s[n]["methods"][m]["all"]["q1_mae_deg"], s[n]["methods"][m]["all"]["q2_mae_deg"]) for n in names], h*.9, color=c, label=label)
    ax.barh(y + 1.5*h, [max(s[n]["gru_mean"]["all"]["q1_mae_deg"], s[n]["gru_mean"]["all"]["q2_mae_deg"]) for n in names], h*.9, color=C[6], label="GRU (3 seeds)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis(); ax.set_xlabel("worse-joint MAE (deg), frozen estimators")
    ax.legend(fontsize=7)
    save(fig, "robustness")


def budget(test_json):
    r = json.loads(test_json.read_text())["results"]
    budgets = sorted(int(k) for k in r)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    methods = [("A-rigid+rehome", C[0]), ("B-compliance", C[1]), ("C-reduced", C[2]), ("D-ekf", C[3]), ("E-gru", C[6]), ("frozen-compliance", C[7])]
    for j in range(2):
        for m, c in methods:
            axes[j].plot(budgets, [r[str(b)]["aggregate"][m]["all"][f"q{j+1}_mae_deg"] for b in budgets], "o-", color=c, label=m, lw=1.5, ms=4)
        axes[j].set_xscale("log", base=2); axes[j].set_xlabel("calibration episodes (privileged joint labels)"); axes[j].set_ylabel(f"q{j+1} MAE (deg), sealed test")
    axes[0].legend(fontsize=7)
    save(fig, "budget")


def failures(frozen_json, data_root):
    from til.phase4_estimators import frozen_estimates, load_frozen
    from til.two_tendon_observer import load_runs
    rows = json.loads(frozen_json.read_text())["runs"]
    worst = max(rows, key=lambda r: max(r["scores"]["compliance"]["all"]["q1_mae_deg"], r["scores"]["compliance"]["all"]["q2_mae_deg"]))
    run = load_runs(str(ROOT/worst["file"]))[0]; cal, models = load_frozen(); est = frozen_estimates(run, cal, models)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for j in range(2):
        axes[j].plot(run["time"], np.degrees(run["y"][:, j]), color=INK, lw=1.5, label="true")
        for m, c in (("compliance", C[1]), ("physics", C[2]), ("gru-seed0", C[6])):
            axes[j].plot(run["time"], np.degrees(est[m][:, j] - run["y"][:, j]) + np.degrees(run["y"][:, j]), color=c, lw=1, label=m)
        axes[j].set_title(f"{worst['id']}  q{j+1}", fontsize=7); axes[j].set_xlabel("time (s)"); axes[j].set_ylabel("deg")
    axes[0].legend(fontsize=7)
    save(fig, "worst_frozen_run")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scale", default="full"); a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); data = ROOT/"data/phase4"/a.scale
    routing(); jacobian(); cable_mechanics(); tensions(data)
    if (FIG/"gates.json").exists(): energy(FIG/"gates.json")
    if (FIG/f"frozen-{a.scale}.json").exists():
        frozen_predictions(data, FIG/f"frozen-{a.scale}.json"); failures(FIG/f"frozen-{a.scale}.json", data)
    if (FIG/f"robustness-{a.scale}.json").exists(): robustness(FIG/f"robustness-{a.scale}.json")
    if (FIG/"recal/test.json").exists(): budget(FIG/"recal/test.json")


if __name__ == "__main__":
    main()
