"""Milestone F: Phase 4 acceptance gates, written to one machine-readable JSON file.

Every number is computed here. Tolerances are the ones declared in PHASE4_DESIGN.md section 7.
"""
import argparse, hashlib, json, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from math import pi, exp
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from til.finger import FingerParams
from til.phase4_cable import CableChainParams, CableMaterial, ContactParams, DistributedCablePlant, chain_from_route
from til.phase4_experiment import specification, simulate
from til.phase4_geometry import (P0_TOTAL_LENGTH, Phase4Finger, joint_grid, rig_route, va_route, vb_analytic, vb_route,
                                 vb_topology)
from til.phase4_hil import Fidelity
from til.transmission import TransmissionParams, TransmissionPlant
from til.two_tendon import ARM, HOME
from til import two_tendon_experiment as tte

SOURCES = ["til/phase4_geometry.py", "til/phase4_cable.py", "til/phase4_hil.py", "til/phase4_hardware.py", "til/phase4_experiment.py", "til/phase4_study.py", "scripts/phase4_verify.py"]


def source_hashes():
    return {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES}


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def gate(value, limit, kind="<"):
    ok = bool(value < limit) if kind == "<" else bool(value > limit) if kind == ">" else bool(value <= limit)
    return dict(value=float(value), limit=float(limit), comparison=kind, passed=ok)


# ------------------------------------------------------------------------------------------------- geometry
def geometry_gates():
    out = {}
    va, vb = Phase4Finger(route=va_route()), Phase4Finger(route=vb_route())
    grid = joint_grid()
    l0 = va.lengths_and_jacobian(HOME)[0]
    err_J = max(np.max(abs(va.lengths_and_jacobian(q)[1] + ARM)) for q in grid)
    err_l = max(np.max(abs(va.lengths_and_jacobian(q)[0] - l0 + ARM @ (q - HOME))) for q in grid)
    out["VA_jacobian_vs_P0_m_per_rad"] = gate(err_J, 1e-8)
    out["VA_length_vs_P0_m"] = gate(err_l, 1e-8)
    fd, an, home = 0., 0., np.max(abs(vb.lengths_and_jacobian(HOME)[1] + ARM))
    arms = []
    for q in grid:
        l, J = vb.lengths_and_jacobian(q)
        an = max(an, np.max(abs(J - vb_analytic(vb.route, q)[1])))
        cols = [(vb.lengths_and_jacobian(q + e)[0] - vb.lengths_and_jacobian(q - e)[0])/2e-6 for e in np.eye(2)*1e-6]
        fd = max(fd, np.max(abs(np.column_stack(cols) - J)))
        arms.append(vb_topology(vb.route, q)["arms"])
    arms = np.array(arms)
    out["VB_jacobian_fd_m_per_rad"] = gate(fd, 1e-7)
    out["VB_jacobian_mujoco_vs_closed_form"] = gate(an, 1e-10)
    out["VB_jacobian_at_home_vs_A"] = gate(home, 1e-9)
    variation = (arms.max(0) - arms.min(0))/np.array([ARM[0, 0], ARM[1, 0], ARM[1, 1]])
    out["VB_moment_arm_variation_fraction"] = dict(values=variation.tolist(), limit=.2, comparison=">", passed=bool(np.all(variation > .2)),
                                                   arm_ranges_m=[[float(a), float(b)] for a, b in zip(arms.min(0), arms.max(0))])
    out["VB_topology_valid_on_grid"] = dict(passed=bool(all(vb_topology(vb.route, q)["valid"] for q in grid)), points=len(grid))
    rng = np.random.default_rng(0)
    worst = 0.
    import mujoco
    for f in (va, vb):
        for q in grid[::3]:
            f.data.qpos[f.qadr] = q; f.data.qvel[:] = 0
            T = rng.uniform(1, 40, 2); f.apply_tension(T); mujoco.mj_forward(f.model, f.data)
            worst = max(worst, np.max(abs(f.data.qfrc_actuator[f.dof] + f.tendon_jacobian().T @ T)))
    out["virtual_work_tau_plus_JT_T_Nm"] = gate(worst, 1e-8)
    ra, rb = rig_route("R-A", total_length=P0_TOTAL_LENGTH), rig_route("R-B", total_length=P0_TOTAL_LENGTH)
    qs = np.linspace(-pi/6, pi/3, 181)
    la = np.array([ra.evaluate(q)["total_length"] for q in qs])
    lb = np.array([rb.evaluate(q)["total_length"] for q in qs])
    tang = 0.
    for route in (ra, rb):
        for q in qs:
            e = route.evaluate(q)
            for i, (pa, pb) in enumerate(e["tangent_points"]):
                d = (pb - pa)/np.hypot(*(pb - pa))
                for node, c, p in ((route.nodes[i], e["centers"][i], pa), (route.nodes[i+1], e["centers"][i+1], pb)):
                    if node.radius:
                        tang = max(tang, abs(np.hypot(*(p - c)) - node.radius), abs((p - c) @ d))
    out["RA_length_invariance_m"] = gate(np.ptp(la), 1e-9)
    out["RA_home_length_equals_P0_m"] = gate(abs(la[0] - P0_TOTAL_LENGTH), 1e-9)
    out["rig_tangency_error_m"] = gate(tang, 1e-12)
    out["rig_routes_valid_over_range"] = dict(passed=bool(all(ra.evaluate(q)["valid"] and rb.evaluate(q)["valid"] for q in qs)))
    out["RB_length_variation_m"] = dict(value=float(np.ptp(lb)), max_slope_m_per_rad=float(np.max(abs(np.gradient(lb, qs)))), passed=True,
                                        note="descriptive")
    out["RA_wraps_deg"] = {n.name: [float(np.degrees(ra.evaluate(q)["wraps"][k])) for q in (-pi/6, HOME[0], pi/3)]
                           for k, n in enumerate(ra.contacts)}
    return out


# ------------------------------------------------------------------------------------------------- cable
def drive(plant, n, x_in, x_out=lambda t: 0., dt=1e-4):
    out = []
    for k in range(1, n + 1):
        t = k*dt; h = 1e-7
        out.append(plant.step(dt, x_in(t), (x_in(t+h) - x_in(t-h))/(2*h), x_out(t), (x_out(t+h) - x_out(t-h))/(2*h)))
    return np.array(out)


def cable_gates():
    out = {}
    p2 = TransmissionPlant(TransmissionParams()); L = p2.lengths
    c = ContactParams("equivalent_pulley", "rolling", radius=.02, wrap=pi/2, inertia=2e-6, mu=.05, level_model="capstan",
                      bearing_coulomb=.001, bearing_viscous=1e-5, presliding_stiffness=2/.02**2, bending_coefficient=1e-4, bending_radius=.03)
    d = DistributedCablePlant(CableChainParams(CableMaterial(viscous_Ns=L[0], linear_density_kg_per_m=0., taut_regularization_m=0.), [c], L, 10.))
    xi = lambda t: .002*np.sin(2*pi*.5*t) + .0005*np.sin(2*pi*3*t); xo = lambda t: -.001*np.sin(2*pi*.7*t)
    out["one_contact_vs_phase2_N"] = gate(np.max(abs(drive(p2, 20000, xi, xo) - drive(d, 20000, xi, xo))), 1e-6)
    elastic = lambda **k: CableMaterial(nonlinear_strain_coefficient=0., maxwell_ratio=0., viscous_Ns=0., **k)
    ra = rig_route("R-A", total_length=P0_TOTAL_LENGTH)
    s = DistributedCablePlant(chain_from_route(ra, HOME[0], material=elastic(), friction="frictionless",
                                               contact_overrides={"rolling": dict(bearing_viscous=0., inertia=1e-9)}))
    drive(s, 4000, lambda t: min(t, .2)*.005)
    out["series_springs_N"] = gate(max(np.ptp(s.T), abs(s.T[0] - 10 - s.p.material.EA/np.sum(s.p.segment_lengths)*.001)), 1e-6)
    ratios = {}
    for beta in (pi/4, pi/2, pi):
        g = DistributedCablePlant(CableChainParams(elastic(), [ContactParams("guide", "sliding", radius=.01, wrap=beta, mu=.2,
                                                                                        presliding_stiffness=5e5)], (.2, .2), 10.))
        drive(g, 6000, lambda t: .01*t)
        ratios[f"{np.degrees(beta):.0f}deg"] = abs(g.T[0]/g.T[1] - exp(.2*beta))/exp(.2*beta)
    out["capstan_ratio_relative_error"] = dict(values=ratios, limit=.01, passed=bool(max(ratios.values()) < .01))
    m = CableMaterial(nonlinear_strain_coefficient=0., viscous_Ns=0.)
    r = DistributedCablePlant(CableChainParams(m, [], (.4,), 10.))
    T = drive(r, 20000, lambda t: .001); t = np.arange(1, 20001)*1e-4
    k = r.k[0]; km = m.maxwell_ratio*k
    analytic = k*r.e[0] + km*.001*np.exp(-t/m.relaxation_time_s)
    out["SLS_relaxation_relative_error"] = gate(np.max(abs(T[:, 0] - analytic)/analytic), .01)
    m = CableMaterial(nonlinear_strain_coefficient=0., viscous_Ns=0., creep_strain_rate=2e-4)
    cr = DistributedCablePlant(CableChainParams(m, [], (.4,), 10.)); F = 20.; a = 1e-4/m.relaxation_time_s; xs = []
    for n in range(30000):
        e = (F + km*cr.v[0]/(1 + a))/(k + km/(1 + a)); x = e - (cr.L[0] - cr.M[0] - cr.creep[0]); cr.step(1e-4, x, 0., 0., 0.); xs.append(x)
    t = np.arange(30000)*1e-4; tau_c = m.relaxation_time_s*(k + km)/k
    analytic = (F - 10)/k - ((F - 10)/k - (F - 10)/(k + km))*np.exp(-t/tau_c) + .4*m.creep_strain_rate*((F - 12)/18)**2*t
    out["SLS_plus_Norton_creep_relative_error"] = gate(np.max(abs(np.array(xs) - analytic)/analytic), .01)
    res = {}
    for dt in (4e-4, 2e-4, 1e-4, 5e-5):
        e = DistributedCablePlant(chain_from_route(ra, HOME[0]))
        drive(e, int(round(2/dt)), lambda t: .003*np.sin(2*pi*t), lambda t: -.001*np.sin(2*pi*t), dt=dt)
        res[str(dt)] = dict(residual_J=e.energy_residual(), numerical_dissipation_J=e.E_numerical, friction_J=e.E_friction,
                            viscous_J=e.E_viscous, maxwell_J=e.E_maxwell)
    vals = [abs(v["residual_J"]) for v in res.values()]
    out["isolated_energy_residual"] = dict(per_dt=res, residual_per_cycle_at_dt_1e4_J=vals[2]/2, limit=1e-6,
                                           decreasing=bool(vals[0] > vals[1] > vals[2]),
                                           passed=bool(vals[2]/2 < 1e-6 and vals[0] > vals[1] > vals[2]))
    # moving support with emulator compensation (R-A and R-B)
    for kind in ("R-A", "R-B"):
        route = rig_route(kind, total_length=P0_TOTAL_LENGTH); pl = DistributedCablePlant(chain_from_route(route, HOME[0]))
        q = lambda t: HOME[0] + .4*np.sin(pi*t); l0 = route.evaluate(HOME[0])["total_length"]
        for kk in range(4000):
            t0, t1 = kk*5e-4, (kk + 1)*5e-4
            L0, L1 = route.segment_lengths(q(t0)), route.segment_lengths(q(t1))
            pl.set_geometry(L0, (L1 - L0)/5e-4, route.evaluate(q(t0))["wraps"])
            x0, x1 = -(L0.sum() - l0), -(L1.sum() - l0)
            for j in range(5):
                pl.step(1e-4, 0., 0., x0 + (x1 - x0)*(j + 1)/5, (x1 - x0)/5e-4)
        g = pl.ground_truth()
        out[f"moving_support_{kind}"] = dict(energy_residual_J=g["energy_residual"], support_work_J=g["support_work"],
                                             min_tension_N=float(pl.T.min()), passed=bool(abs(g["energy_residual"]) < 1e-5 and pl.T.min() > 5))
    return out


# ------------------------------------------------------------------------------------------------- coupled
def _run(job):
    name, spec, fid, kw = job
    t = time.time()
    try:
        summary, a = simulate(spec, fid, **kw)
        return name, dict(summary=summary, trace=a, seconds=time.time() - t)
    except Exception as e:
        return name, dict(error=f"{type(e).__name__}: {e}", seconds=time.time() - t)


def coupled_jobs():
    c = specification(0, "test", "coordinated", contact=True, fidelity="P2", tag="_verify")
    f = dict(c, contact=False)
    p = dict(f, motion="proximal")
    jobs = []
    for level in ("P2", "P3"):
        for name, spec in (("contact", c), ("free", f)):
            for res, (outer, plant) in (("base", (5e-4, 1e-4)), ("half", (2.5e-4, 5e-5)), ("quarter", (1.25e-4, 2.5e-5))):
                jobs.append((f"refine/{level}/{name}/{res}", spec, level, dict(outer=outer, plant=plant, duration=.6 + c["period_s"])))
    for solref in (.002, .004, .008):
        jobs.append((f"solref/{solref}", c, "P2", dict(solref=solref, duration=.6 + c["period_s"])))
    for label, route in (("slope6mm", dict(offset_guide=(0., .006), offset_radius=.012)),
                         ("slope17mm", dict(offset_guide=(.012, .012), offset_radius=.012)),
                         ("slope45mm", dict(offset_guide=(-.025, .065), offset_radius=.006))):
        jobs.append((f"rb/{label}", p, Fidelity("RB", rig="R-B", route=route), dict(duration=.6 + c["period_s"])))
    jobs.append(("parity/P0_existing", c, "P0", dict(duration=1.5)))
    jobs.append(("parity/VA_lumped", c, Fidelity("VA", virtual="V-A", rig="none", cable="lumped"), dict(duration=1.5)))
    jobs.append(("sensing/clean", f, "P2", dict(duration=1.)))
    jobs.append(("sensing/telemetry_only", f, Fidelity("P2s", sensing="imperfect"), dict(duration=1.)))
    return jobs


def interp_trace(a, b):
    """Resample trace b onto trace a's log times."""
    t = a[:, 0]
    return np.column_stack([np.interp(t, b[:, 0], b[:, j]) for j in range(b.shape[1])])


def coupled_gates(workers):
    jobs = coupled_jobs()
    with ProcessPoolExecutor(workers) as ex:
        results = dict(ex.map(_run, jobs))
    out = {"runs": {k: dict(seconds=v["seconds"], error=v.get("error"), flags=v.get("summary", {}).get("flags"),
                            solver=v.get("summary", {}).get("solver"), rig=v.get("summary", {}).get("rig"))
                    for k, v in results.items()}}
    refine = {}
    for level in ("P2", "P3"):
        for name in ("contact", "free"):
            ref = results[f"refine/{level}/{name}/quarter"]
            for res in ("base", "half"):
                cur = results[f"refine/{level}/{name}/{res}"]
                if "error" in cur or "error" in ref:
                    refine[f"{level}/{name}/{res}"] = dict(error=cur.get("error") or ref.get("error"), passed=False); continue
                a, b = cur["trace"], interp_trace(cur["trace"], ref["trace"])
                q = np.degrees(np.sqrt(np.mean((a[:, 1:3] - b[:, 1:3])**2, 0)))
                T = np.sqrt(np.mean((a[:, 13:17] - b[:, 13:17])**2, 0))
                limit = .5 if name == "contact" else .25
                refine[f"{level}/{name}/{res}"] = dict(joint_rmse_deg=q.tolist(), tension_rmse_N=T.tolist(), joint_limit_deg=limit,
                                                       tension_limit_N=1., passed=bool(q.max() < limit and T.max() < 1.))
    out["timestep_refinement_vs_quarter_step"] = refine
    sol = {}
    base = results["solref/0.004"]["trace"]
    for s in (.002, .008):
        r = results[f"solref/{s}"]
        if "error" in r:
            sol[str(s)] = dict(error=r["error"]); continue
        b = interp_trace(base, r["trace"])
        sol[str(s)] = dict(joint_rmse_deg=np.degrees(np.sqrt(np.mean((base[:, 1:3] - b[:, 1:3])**2, 0))).tolist(),
                           contact_force_peak_N=float(r["trace"][:, 19].max()), finite=True)
    out["contact_solref_sensitivity"] = sol
    rb = {}
    for label in ("slope6mm", "slope17mm", "slope45mm"):
        r = results[f"rb/{label}"]
        rb[label] = dict(stable=("error" not in r and "large_tracking_error" not in r["summary"]["flags"]),
                         error=r.get("error"), flags=r.get("summary", {}).get("flags"))
    out["rig_length_slope_stability"] = rb
    a0, a1 = results["parity/P0_existing"]["trace"], tte.simulate(specification(0, "test", "coordinated", contact=True, fidelity="P2", tag="_verify"),
                                                                  duration=1.5)[1]
    out["P0_preset_equals_existing_simulator"] = gate(np.max(abs(a0 - a1)), 1e-15, "<=")
    va = results["parity/VA_lumped"]["trace"]
    out["VA_lumped_equals_P0_joint_rad"] = gate(np.max(abs(va[:, 1:3] - a0[:, 1:3])), 1e-8)
    cl, no = results["sensing/clean"]["trace"], results["sensing/telemetry_only"]["trace"]
    out["telemetry_only_corruption_changes_no_mechanics"] = dict(joint_diff=float(np.max(abs(cl[:, 1:3] - no[:, 1:3]))),
        tension_diff=float(np.max(abs(cl[:, 13:17] - no[:, 13:17]))), motor_channel_diff=float(np.max(abs(cl[:, 5:13] - no[:, 5:13]))),
        passed=bool(np.max(abs(cl[:, 1:3] - no[:, 1:3])) == 0 and np.max(abs(cl[:, 5:13] - no[:, 5:13])) > 0))
    for level in ("P2", "P3"):
        for name in ("contact", "free"):
            s = results[f"refine/{level}/{name}/base"]
            if "error" in s:
                continue
            a = s["trace"]; hold = a[:, 0] < .6
            out[f"coupled/{level}/{name}"] = dict(flags=s["summary"]["flags"], energy_residual_J=s["summary"]["energy_residual_J"],
                numerical_dissipation_J=s["summary"]["numerical_dissipation_J"], support_work_J=s["summary"]["support_work_J"],
                pretension_hold_joint_dev_deg=float(np.degrees(np.max(abs(a[hold, 1:3] - HOME)))),
                pretension_hold_tension_dev_N=float(np.max(abs(a[hold, 13:17] - 10))), contact_force_peak_N=s["summary"]["contact_force_peak_N"],
                peak_tensions_N=s["summary"]["peak_input_output_tension_N"], rig=s["summary"]["rig"], solver=s["summary"]["solver"])
    return out


def delay_study(workers):
    """Emulator interface delay sensitivity (inner-loop configuration, separate from telemetry)."""
    import copy
    from til import phase4_hil
    c = specification(0, "test", "coordinated", contact=True, fidelity="P2", tag="_verify")
    jobs = [(f"delay/{d}", c, Fidelity("P2d", hardware="nominal"), dict(duration=.6 + c["period_s"], emulator_delay=d))
            for d in (0., 5e-4, 1e-3, 2e-3)]
    with ProcessPoolExecutor(workers) as ex:
        results = dict(ex.map(_run_delay, jobs))
    base = results["delay/0.0"]["trace"]
    out = {}
    for k, r in results.items():
        if "error" in r:
            out[k] = dict(error=r["error"], stable=False); continue
        b = interp_trace(base, r["trace"])
        out[k] = dict(stable="large_tracking_error" not in r["summary"]["flags"], flags=r["summary"]["flags"],
                      joint_rmse_vs_zero_delay_deg=np.degrees(np.sqrt(np.mean((base[:, 1:3] - b[:, 1:3])**2, 0))).tolist(),
                      tension_rmse_vs_zero_delay_N=np.sqrt(np.mean((base[:, 13:17] - b[:, 13:17])**2, 0)).tolist())
    return out


def _run_delay(job):
    name, spec, fid, kw = job
    delay = kw.pop("emulator_delay")
    import til.phase4_experiment as pe
    original = pe.resolved_configuration
    def patched(*args, **kwargs):
        configs, fp = original(*args, **kwargs)
        for c in configs:
            c.emulator.delay = delay
        return configs, fp
    pe.resolved_configuration = patched
    try:
        return _run((name, spec, fid, kw))
    finally:
        pe.resolved_configuration = original


# ------------------------------------------------------------------------------------------------- hardware topology
def _run_hw(job):
    name, spec, fid, kw = job
    from til.phase4_experiment import simulate_hardware
    t = time.time()
    try:
        summary, a = simulate_hardware(spec, fid, **kw)
        return name, dict(summary=summary, trace=a, seconds=time.time() - t)
    except Exception as e:
        return name, dict(error=f"{type(e).__name__}: {e}", seconds=time.time() - t)


def hardware_gates(workers):
    from til.phase4_geometry import hardware_routes
    from til.phase4_hardware import HardwareFidelity
    out = {}
    t1, t2 = hardware_routes()
    qs = np.linspace(-pi/6, pi/3, 91)
    out["route1_length_independent_of_q"] = gate(max(abs(t1.segment_jacobian(q).sum()) for q in qs), 1e-12)
    out["route2_axial_passage_no_length_change_m"] = gate(max(np.max(abs(t2.segment_jacobian(q))) for q in qs), 1e-15, "<=")
    from til.phase4_hardware import nominal_A
    out["nominal_winding_matrix"] = dict(axial=nominal_A((t1, t2)).tolist(), coaxial_variant=nominal_A(hardware_routes(mcp_passage="coaxial")).tolist(),
                                         passed=bool(np.allclose(nominal_A((t1, t2)), [[.015, 0], [0, .010]])))
    out["routes_valid_over_range"] = dict(passed=bool(all(t1.evaluate(q)["valid"] and t2.evaluate(q)["valid"] for q in qs)))
    c = specification(0, "test", "coordinated", contact=True, fidelity="H2", tag="_verify")
    f = dict(c, contact=False)
    dur = .6 + c["period_s"]
    jobs = []
    for level in ("H2", "H3"):
        for name, spec in (("contact", c), ("free", f)):
            for res, (outer, plant) in (("base", (5e-4, 1e-4)), ("half", (2.5e-4, 5e-5)), ("quarter", (1.25e-4, 2.5e-5))):
                jobs.append((f"refine/{level}/{name}/{res}", spec, level, dict(outer=outer, plant=plant, duration=dur)))
    for steps in (0, 1, 2, 4, 8):
        jobs.append((f"render_delay/{steps}", c, HardwareFidelity("H2d", render_delay_steps=steps), dict(duration=dur)))
    for k in (2e3, 5e3, 2e4, 1e5, 5e5):
        jobs.append((f"render_stiffness/{k:g}", c, HardwareFidelity("H2k", render_contact=dict(contact_stiffness=k)), dict(duration=dur)))
    with ProcessPoolExecutor(workers) as ex:
        results = dict(ex.map(_run_hw, jobs))
    out["runs"] = {k: dict(seconds=v["seconds"], error=v.get("error"), flags=v.get("summary", {}).get("flags")) for k, v in results.items()}
    refine = {}
    for level in ("H2", "H3"):
        for name in ("contact", "free"):
            ref = results[f"refine/{level}/{name}/quarter"]
            for res in ("base", "half"):
                cur = results[f"refine/{level}/{name}/{res}"]
                if "error" in cur or "error" in ref:
                    refine[f"{level}/{name}/{res}"] = dict(error=cur.get("error") or ref.get("error"), passed=False); continue
                a, b = cur["trace"], interp_trace(cur["trace"], ref["trace"])
                q = np.degrees(np.sqrt(np.mean((a[:, 1:3] - b[:, 1:3])**2, 0))); T = np.sqrt(np.mean((a[:, 13:17] - b[:, 13:17])**2, 0))
                limit = .5 if name == "contact" else .25
                refine[f"{level}/{name}/{res}"] = dict(joint_rmse_deg=q.tolist(), tension_rmse_N=T.tolist(), joint_limit_deg=limit,
                                                       tension_limit_N=1., passed=bool(q.max() < limit and T.max() < 1.))
    out["timestep_refinement_vs_quarter_step"] = refine
    def stability(prefix, keys, base_key):
        res = {}
        base = results[f"{prefix}/{base_key}"]
        for k in keys:
            r = results[f"{prefix}/{k}"]
            if "error" in r:
                res[str(k)] = dict(stable=False, error=r["error"]); continue
            s = r["summary"]; b = interp_trace(base["trace"], r["trace"]) if "trace" in base else None
            res[str(k)] = dict(stable=not any(x in s["flags"] for x in ("large_tracking_error", "render_command_clipped", "rig_hard_stop")),
                               flags=s["flags"], contact_force_peak_N=s["contact_force_peak_N"],
                               emulator_force_tracking_peak_N=s["emulator_force_tracking_peak_N"],
                               joint_rmse_vs_reference_deg=None if b is None else np.degrees(np.sqrt(np.mean((base["trace"][:, 1:3] - b[:, 1:3])**2, 0))).tolist())
        return res
    out["render_delay_stability"] = stability("render_delay", (0, 1, 2, 4, 8), 0)
    out["render_contact_stiffness_stability"] = stability("render_stiffness", ("2000", "5000", "20000", "100000", "500000"), "5000")
    for level in ("H2", "H3"):
        for name in ("contact", "free"):
            s = results[f"refine/{level}/{name}/base"]
            if "error" in s:
                out[f"coupled/{level}/{name}"] = dict(error=s["error"], passed=False); continue
            a = s["trace"]; hold = a[:, 0] < .6; sm = s["summary"]
            out[f"coupled/{level}/{name}"] = dict(flags=sm["flags"], energy_residual_J=sm["energy_residual_J"],
                numerical_dissipation_J=sm["numerical_dissipation_J"], joint_work_J=sm["joint_work_J"], support_work_J=sm["support_work_J"],
                rendering_work_J=sm["rendering_work_J"], mechanism=sm["mechanism"],
                pretension_hold_joint_dev_deg=float(np.degrees(np.max(abs(a[hold, 1:3] - HOME)))),
                pretension_hold_tension_dev_N=float(np.max(abs(a[hold, 13:17] - 10))), contact_force_peak_N=sm["contact_force_peak_N"],
                free_tracking_mae_deg=sm["mae_deg"], emulator_force_tracking_peak_N=sm["emulator_force_tracking_peak_N"], solver=sm["solver"],
                passed=bool(not sm["flags"] and max(abs(x) for x in sm["energy_residual_J"]) < 1e-4 and abs(sm["mechanism"]["residual_J"]) < 1e-3))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT/"figures/phase4/gates.json"))
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--skip-coupled", action="store_true")
    ap.add_argument("--only-hardware", action="store_true", help="recompute the hardware-topology gates and merge into --out"); a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if a.only_hardware and out.exists():
        report = json.loads(out.read_text())
        report.update(source_commit=git_head(), source_sha256=source_hashes(), updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        report["hardware"] = hardware_gates(a.workers); print("hardware done", flush=True)
    else:
        report = dict(source_commit=git_head(), source_sha256=source_hashes(), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                      scope="Synthetic reference-plant verification. Category C (hypothetical) parameters; no hardware data.")
        report["geometry"] = geometry_gates(); print("geometry done", flush=True)
        report["cable"] = cable_gates(); print("cable done", flush=True)
        if not a.skip_coupled:
            report["coupled"] = coupled_gates(a.workers); print("coupled done", flush=True)
            report["emulator_delay"] = delay_study(a.workers); print("delay done", flush=True)
            report["hardware"] = hardware_gates(a.workers); print("hardware done", flush=True)
    report["note"] = ("'coupled' and 'emulator_delay' verify the first Phase 4 architecture (virtual joints, rig-link servo), kept as an "
                      "alternative; the study uses the hardware topology verified under 'hardware' (user decision 2026-11-16).")
    for k in ("summary", "all_passed", "seconds"):
        report.pop(k, None)
    def collect(d, path=""):
        for k, v in d.items():
            if isinstance(v, dict):
                if "passed" in v:
                    yield f"{path}{k}", v["passed"]
                else:
                    yield from collect(v, f"{path}{k}/")
    report["summary"] = {k: v for k, v in collect({kk: vv for kk, vv in report.items() if isinstance(vv, dict)})}
    report["all_passed"] = bool(all(report["summary"].values()))
    report["seconds"] = time.time() - start
    out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf8")
    print(json.dumps(report["summary"], indent=1)); print("all passed:", report["all_passed"])


if __name__ == "__main__":
    main()
