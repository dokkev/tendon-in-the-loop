"""Phase 4 matched-task experiments, recorded in the two-channel schema the frozen estimators read.

The controller, the motor command protocol and the task generator are the P0 ones (constant-A command map,
nominal compliance feedforward, quasistatic tension feedforward). Only the reference plant changes, so a run
with the same specification is a matched task across P0-P3.
"""
from dataclasses import asdict
import copy
import json

import numpy as np

from .phase4_hil import IMPERFECT_HARDWARE, Fidelity, Phase4Simulator, preset
from .transmission import FIXTURES
from .two_tendon import ARM, HOME
from .two_tendon_experiment import gravity, resolved_configuration, static_tension, target

SEED_BASE = 404000
MOTIONS = ("proximal", "distal", "coordinated", "reversal")


def specification(index=0, split="test", motion="coordinated", contact=False, cycles=2, condition="nominal",
                  fidelity="P2", specimen=None, session=0, tag="", seed_offset=0):
    """Same amplitude / period distributions as the P0 study, Phase 4 seed range (disjoint from P0)."""
    seed = SEED_BASE + {"train": 0, "validation": 10000, "test": 20000, "calibration": 30000}[split] + index + seed_offset
    rng = np.random.default_rng(seed)
    level = fidelity if isinstance(fidelity, str) else fidelity.level
    return dict(id=f"p4_{level}_{split}{tag}_{motion}{'_contact' if contact else ''}_{index:03d}", scenario_id=split, seed=seed,
                condition=condition, motion=motion, amplitude=[float(rng.uniform(.18, .32)), float(rng.uniform(.22, .40))],
                period_s=float(rng.uniform(2.4, 3.2)), cycles=cycles, contact=bool(contact), object_x=.055,
                object_stiffness=1000., controller="elastic_feedforward", fidelity_level=level,
                specimen_id=specimen or f"specimen_{split}_{index:03d}", session_index=session)


def simulate(spec, fidelity=None, writer=None, outer=.0005, plant=.0001, controller_period=None, solref=.004,
             noise=True, duration=None, trace_extra=False):
    fid = copy.deepcopy(fidelity) if isinstance(fidelity, Fidelity) else preset(fidelity or spec.get("fidelity_level", "P2"))
    configs, fp = resolved_configuration(spec, outer, plant, 0., noise)
    if spec["condition"].startswith("noise"):  # sensor noise only: scale the stochastic telemetry terms
        scale = float(spec["condition"][5:])
        for c in configs:
            c.motor.velocity_noise *= scale; c.motor.current_noise *= scale; c.motor.current_colored_noise *= scale
            c.sensors.tension_noise *= scale
    if controller_period is None:
        controller_period = IMPERFECT_HARDWARE["controller_period"] if fid.hardware == "imperfect" else .0001
    sim = Phase4Simulator(configs, fp, fid)
    if writer:
        writer.g.attrs["resolved_configuration"] = json.dumps(dict(channels=[c.to_dict() for c in sim.configs], finger=asdict(fp),
            fidelity=fid.to_dict(), rig_route=sim.rig_route.to_dict() if sim.rig_route else None,
            virtual_route=sim.finger.route.to_dict() if hasattr(sim.finger, "route") else "P0 fixed tendons",
            cable=[ch.plant.p.to_dict() if hasattr(ch.plant.p, "to_dict") else asdict(ch.plant.p) for ch in sim.chains]), default=str)
    sim.finger.model.geom_solref[:, 0] = solref
    nominal_reference = HOME + (-gravity(HOME) - ARM.T @ np.array([10., 10.]))/.08
    for c in sim.chains:
        c.motor.command_period = controller_period
        c.motor.s.i_ff = .02*10/.1
    lengths = FIXTURES["large_guide"].lengths
    compliance = sum(lengths)/(1e10*np.pi*.0005**2/4)
    length = duration or (.6 + spec["cycles"]*spec["period_s"])
    data, flags, extra = [], [], []
    limits, slack, infeasible, peaktracking = 0, np.zeros(2), 0, np.zeros(2)
    rig_peak, rig_invalid, rig_saturated = 0., 0, 0
    for _ in range(round(length/sim.dt)):
        qref, vref = target(sim.t, spec)
        Tref = static_tension(qref, nominal_reference)
        infeasible += int(np.any(Tref < 0) | np.any(Tref > 40))
        commands = ARM @ (qref - HOME)
        if spec["controller"] == "elastic_feedforward":
            commands += compliance*(Tref - 10.)
        for i, c in enumerate(sim.chains):
            c.motor.s.i_ff = float(np.clip(.02*Tref[i]/.1, -8., 8.))
        sim.step(commands, ARM @ vref)
        if sim.log_due:
            rec = sim.record()
            rec["protocol"] = dict(desired_joints=qref, motor_displacements=commands, static_required_tensions=Tref,
                                   reversal=float(np.any(vref < -.01)))
            if writer:
                writer.append(sim.t, max(0., (sim.t - .6)/spec["period_s"]), rec)
            obs = [rec[f"motor{i}"][k] for i in (1, 2) for k in ("position", "velocity", "current", "applied_command")]
            tin = [c.plant.s.T1 for c in sim.chains]
            tout = [c.plant.s.T2 for c in sim.chains]
            deformation = [c.motor.x_in + c.emulator.s.x for c in sim.chains]
            tracking = [c.emulator.tracking_error for c in sim.chains]
            residual = [c.plant.ground_truth()["energy_residual"] for c in sim.chains]
            data.append([sim.t, *sim.finger.q, *qref, *obs, *tin, *tout, *deformation, rec["simulation"]["contact_force"],
                         *tracking, *residual, *commands])
            if trace_extra:
                extra.append(dict(segment_tensions=[np.array(c.plant.T) if sim.distributed else np.array([c.plant.s.T1, c.plant.s.T2])
                                                    for c in sim.chains],
                                  jacobian=sim.finger.tendon_jacobian().copy(), rig=rec.get("rig", {}).copy()))
            limits += int(any(c.motor.limit_hit or c.emulator.s.force_saturated or abs(c.motor.s.u) >= 7.999 for c in sim.chains))
            slack += np.array(tout) < .05
            peaktracking = np.maximum(peaktracking, np.abs(tracking))
            if sim.rig is not None:
                rig_peak = max(rig_peak, abs(rec["rig"]["tracking_error"]))
                rig_invalid += int(not rec["rig"]["valid"])
                rig_saturated += int(rec["rig"]["servo_saturated"])
    a = np.array(data)
    if not np.isfinite(a).all():
        raise FloatingPointError("nonfinite phase 4 trial")
    e = np.degrees(a[:, 1:3] - a[:, 3:5])
    moving = a[:, 0] >= .6
    if limits: flags.append("actuator_limit")
    if max(peaktracking) > .002: flags.append("emulator_tracking")
    if max(a[:, 13:17].max(0)) > 40.: flags.append("above_nominal_tension_envelope")
    if np.max(np.abs(e)) > 30.: flags.append("large_tracking_error")
    if rig_invalid: flags.append("rig_route_invalid")
    if rig_saturated: flags.append("rig_servo_saturated")
    solver = [dict(c.plant.solver_stats) for c in sim.chains] if sim.distributed else []
    if any(s["fallback"] for s in solver): flags.append("cable_solver_fallback")
    gts = [c.plant.ground_truth() for c in sim.chains]
    summary = dict(config=spec, fidelity=fid.to_dict(), flags=flags, finite=True,
        mae_deg=np.mean(abs(e[moving]), 0).tolist(), rmse_deg=np.sqrt(np.mean(e[moving]**2, 0)).tolist(),
        peak_error_deg=np.max(abs(e), 0).tolist(), limit_fraction=limits/len(a), slack_fraction=(slack/len(a)).tolist(),
        static_free_torque_infeasible_fraction=infeasible/max(1, round(length/sim.dt)),
        peak_input_output_tension_N=a[:, 13:17].max(0).tolist(), motor_travel_mm=(np.ptp(a[:, [5, 9]], axis=0)*20).tolist(),
        motor_current_peak_A=np.max(abs(a[:, [7, 11]]), axis=0).tolist(), contact_force_peak_N=float(a[:, 19].max()),
        emulator_tracking_peak_mm=(peaktracking*1000).tolist(), energy_residual_J=a[-1, 22:24].tolist(),
        numerical_dissipation_J=[g.get("numerical_dissipation", 0.) for g in gts],
        support_work_J=[g.get("support_work", 0.) for g in gts],
        port_work_mismatch_J=sim.record()["coupling"]["port_work_mismatch"].tolist(),
        rig=dict(peak_tracking_deg=float(np.degrees(rig_peak)), invalid_samples=rig_invalid, saturated_samples=rig_saturated,
                 servo_work_J=sim.rig.work if sim.rig else 0., cable_work_on_link_J=sim.rig.cable_work if sim.rig else 0.),
        solver=solver, history_out=[c.plant.export_history() if sim.distributed else {} for c in sim.chains],
        calibration=dict(home=HOME.tolist(), nominal_tension_N=[10, 10], spring_reference_rad=nominal_reference.tolist(),
                         source="One-time joint-reference and input/output load-cell setup; not motor-only initialization"),
        simulation=dict(outer_dt=outer, plant_dt=plant, controller_period=controller_period, contact_solref=solref))
    if writer:
        writer.close(dict(quality=json.dumps(summary, default=float)))
    return (summary, a, extra) if trace_extra else (summary, a)


def simulate_hardware(spec, fidelity=None, writer=None, outer=.0005, plant=.0001, controller_period=None, solref=None,
                      noise=True, duration=None, trace_extra=False):
    """Matched task on the hardware-topology HIL (physical joints, force-rendered virtual load)."""
    from .phase4_hardware import HARDWARE, HardwareFidelity, HardwareHIL, hardware_preset
    fid = copy.deepcopy(fidelity) if isinstance(fidelity, HardwareFidelity) else hardware_preset(fidelity or spec.get("fidelity_level", "H2"))
    configs, fp = resolved_configuration(spec, outer, plant, 0., noise)
    if spec["condition"].startswith("noise"):
        scale = float(spec["condition"][5:])
        for c in configs:
            c.motor.velocity_noise *= scale; c.motor.current_noise *= scale; c.motor.current_colored_noise *= scale
            c.sensors.tension_noise *= scale
    if controller_period is None:
        controller_period = HARDWARE["controller_period"] if fid.hardware == "imperfect" else .0001
    sim = HardwareHIL(configs, fp, fid)
    # solref is unused here: the hardware virtual environment renders a penalty contact (see VirtualEnvironment).
    if writer:
        writer.g.attrs["resolved_configuration"] = json.dumps(dict(channels=[c.to_dict() for c in sim.configs], finger=asdict(fp),
            fidelity=fid.to_dict(), routes=[r.to_dict() for r in sim.routes], cable=[ch["plant"].p.to_dict() for ch in sim.ch],
            architecture="physical joints + force-rendered virtual environment"), default=str)
    A = sim.A  # the rig's nominal winding matrix (design geometry); the P0 controller structure with this A
    nominal_reference = HOME + (-gravity(HOME) - A.T @ np.array([10., 10.]))/.08
    for ch in sim.ch:
        ch["motor"].command_period = controller_period
        ch["motor"].s.i_ff = .02*10/.1
    lengths = FIXTURES["large_guide"].lengths
    compliance = sum(lengths)/(1e10*np.pi*.0005**2/4)
    length = duration or (.6 + spec["cycles"]*spec["period_s"])
    data, flags, extra = [], [], []
    limits, slack, infeasible, peaktracking = 0, np.zeros(2), 0, np.zeros(2)
    for _ in range(round(length/sim.dt)):
        qref, vref = target(sim.t, spec)
        Tref = np.linalg.solve(A.T, .08*(qref - nominal_reference) - gravity(qref))
        infeasible += int(np.any(Tref < 0) | np.any(Tref > 40))
        commands = A @ (qref - HOME)
        if spec["controller"] == "elastic_feedforward":
            commands += compliance*(Tref - 10.)
        for i, ch in enumerate(sim.ch):
            ch["motor"].s.i_ff = float(np.clip(.02*Tref[i]/.1, -8., 8.))
        sim.step(commands, A @ vref)
        if sim.log_due:
            rec = sim.record()
            rec["protocol"] = dict(desired_joints=qref, motor_displacements=commands, static_required_tensions=Tref,
                                   reversal=float(np.any(vref < -.01)))
            if writer:
                writer.append(sim.t, max(0., (sim.t - .6)/spec["period_s"]), rec)
            obs = [rec[f"motor{i}"][k] for i in (1, 2) for k in ("position", "velocity", "current", "applied_command")]
            tin = [float(ch["plant"].T[0]) for ch in sim.ch]
            tout = [float(ch["plant"].T[-1]) for ch in sim.ch]
            deformation = [ch["motor"].x_in + ch["emulator"].x for ch in sim.ch]
            tracking = [rec[f"emulator{i}"]["force_tracking_error"] for i in (1, 2)]
            residual = [ch["plant"].energy_residual() for ch in sim.ch]
            data.append([sim.t, *sim.mech.q, *qref, *obs, *tin, *tout, *deformation, rec["simulation"]["contact_force"],
                         *tracking, *residual, *commands])
            if trace_extra:
                extra.append(dict(segment_tensions=[ch["plant"].T.copy() for ch in sim.ch], commands=sim.T_cmd.copy(),
                                  env_torque=sim.env.tau.copy(), cable_torque=sim.cable_torque.copy()))
            limits += int(any(ch["motor"].limit_hit or ch["emulator"].saturated or ch["emulator"].stop_hit or abs(ch["motor"].s.u) >= 7.999
                              for ch in sim.ch))
            slack += np.array(tout) < .05
            peaktracking = np.maximum(peaktracking, np.abs(tracking))
    a = np.array(data)
    if not np.isfinite(a).all():
        raise FloatingPointError("nonfinite hardware trial")
    e = np.degrees(a[:, 1:3] - a[:, 3:5]); moving = a[:, 0] >= .6
    if limits: flags.append("actuator_or_emulator_limit")
    if max(a[:, 13:17].max(0)) > 40.: flags.append("above_nominal_tension_envelope")
    if np.max(np.abs(e)) > 30.: flags.append("large_tracking_error")
    if sim.mech.stop_hits: flags.append("rig_hard_stop")
    if sim.render_clipped: flags.append("render_command_clipped")
    solver = [dict(ch["plant"].solver_stats) for ch in sim.ch]
    if any(s["fallback"] for s in solver): flags.append("cable_solver_fallback")
    gts = [ch["plant"].ground_truth() for ch in sim.ch]
    mech = sim.mech
    summary = dict(config=spec, fidelity=fid.to_dict(), architecture="hardware", nominal_winding_matrix=A.tolist(), flags=flags, finite=True,
        mae_deg=np.mean(abs(e[moving]), 0).tolist(), rmse_deg=np.sqrt(np.mean(e[moving]**2, 0)).tolist(),
        peak_error_deg=np.max(abs(e), 0).tolist(), limit_fraction=limits/len(a), slack_fraction=(slack/len(a)).tolist(),
        static_free_torque_infeasible_fraction=infeasible/max(1, round(length/sim.dt)),
        peak_input_output_tension_N=a[:, 13:17].max(0).tolist(), motor_travel_mm=(np.ptp(a[:, [5, 9]], axis=0)*20).tolist(),
        motor_current_peak_A=np.max(abs(a[:, [7, 11]]), axis=0).tolist(), contact_force_peak_N=float(a[:, 19].max()),
        emulator_force_tracking_peak_N=peaktracking.tolist(), energy_residual_J=a[-1, 22:24].tolist(),
        numerical_dissipation_J=[g["numerical_dissipation"] for g in gts], joint_work_J=[g["joint_work"] for g in gts],
        support_work_J=[g["support_work"] for g in gts], rendering_work_J=sim.W_render.tolist(),
        mechanism=dict(cable_work_J=mech.KE_work, kinetic_J=mech.kinetic, bearing_dissipation_J=mech.E_bearing,
                       residual_J=mech.KE_work - mech.kinetic - mech.E_bearing, stop_hits=mech.stop_hits),
        render_clipped_steps=sim.render_clipped, solver=solver, history_out=[ch["plant"].export_history() for ch in sim.ch],
        calibration=dict(home=HOME.tolist(), nominal_tension_N=[10, 10], spring_reference_rad=nominal_reference.tolist(),
                         source="One-time rig-encoder reference and known pretension; not motor-only initialization"),
        simulation=dict(outer_dt=outer, plant_dt=plant, controller_period=controller_period, contact_solref=solref))
    if writer:
        writer.close(dict(quality=json.dumps(summary, default=float)))
    return (summary, a, extra) if trace_extra else (summary, a)
