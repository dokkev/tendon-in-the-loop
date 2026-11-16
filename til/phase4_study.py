"""Phase 4 study definitions: matched tasks, one-mechanism ablations, one-factor robustness sweeps,
specimen families with persistent sessions, and the HDF5 writers/readers for them.

The study runs on the intended hardware topology (grounded motors, physical MCP link and PIP pulley, force-rendered
virtual load; user decision 2026-11-16). P0 is the frozen historical reference (virtual joints, constant-A fixed
tendons, Phase 2 lumped cable). All material, friction and history values are category C (hypothetical).
"""
import copy
import json
from pathlib import Path

import numpy as np

from .phase4_experiment import simulate, simulate_hardware, specification
from .phase4_geometry import hardware_routes
from .phase4_hardware import HardwareFidelity, hardware_preset, nominal_A
from .two_tendon import ARM
from .phase4_hil import preset
from .two_tendon_experiment import dataset_file, episode_writer, normalize_units
from .two_tendon_observer import load_runs

# (motion, contact, cycles): free single-joint, coordinated, reversal, contact/release/sustained, repeated cycles
TASKS = [("proximal", False, 2), ("distal", False, 2), ("coordinated", False, 2), ("reversal", False, 2),
         ("coordinated", True, 2), ("coordinated", True, 5)]


def task_name(motion, contact, cycles):
    return f"{motion}{'_contact' if contact else ''}{'_repeated' if cycles > 2 else ''}"


HF = HardwareFidelity

# One mechanism changed at a time (instruction section 17).
ABLATIONS = {
    "P0": preset("P0"),
    "H1": hardware_preset("H1"),
    "H1_coaxial": HF("H1_coaxial", friction=False, routes=dict(mcp_passage="coaxial")),
    "H1_coax_offset": HF("H1_coax_offset", friction=False, routes=dict(mcp_passage="coaxial", coax_offset=(-.003, 0.))),
    "H2": hardware_preset("H2"),
    "H2_sliding_guides": HF("H2_sliding_guides", routes=dict(guide_kind="sliding")),
    "H2_sliding_lumped": HF("H2_sliding_lumped", routes=dict(guide_kind="sliding"),
                            contacts=dict(entry_guide=dict(mu=.10), exit_guide=dict(mu=0.))),
    "H2_elastic": HF("H2_elastic", viscoelastic=False),
    "H2_evolving": HF("H2_evolving", history="evolving"),
    "H2_sensing": HF("H2_sensing", sensing="imperfect"),
    "H2_hardware": HF("H2_hardware", hardware="imperfect"),
    "H3": hardware_preset("H3"),
}

ABLATION_PAIRS = [
    ("P0 reference -> hardware topology (physical joints, resolved routing)", "P0", "H1"),
    ("decoupled (MCP axial passage) vs coupled (MCP coaxial idler) routing", "H1", "H1_coaxial"),
    ("constant vs variable Jacobian (offset coaxial idler)", "H1_coaxial", "H1_coax_offset"),
    ("frictionless vs stick-slip bearings", "H1", "H2"),
    ("rolling idlers vs sliding guides", "H2", "H2_sliding_guides"),
    ("lumped vs distributed guide friction (same total capstan exponent)", "H2_sliding_lumped", "H2_sliding_guides"),
    ("elastic vs viscoelastic", "H2_elastic", "H2"),
    ("fixed vs evolving mechanics", "H2", "H2_evolving"),
    ("clean vs imperfect sensing (telemetry only)", "H2", "H2_sensing"),
    ("nominal vs imperfect inner-loop hardware", "H2", "H2_hardware"),
    ("H2 -> H3 (all P3 mechanisms)", "H2", "H3"),
]


def _maxwell_history(fraction):
    return dict(maxwell_fraction=fraction)


# One factor at a time around H2 (section 14): name -> (fidelity, spec condition)
ROBUSTNESS = {
    "nominal": (HF("nominal"), "nominal"),
    "sensor_noise_x0": (HF("sensor_noise_x0"), "noise0"),
    "sensor_noise_x3": (HF("sensor_noise_x3"), "noise3"),
    "sensor_bias_gain": (HF("sensor_bias_gain", sensing="imperfect"), "nominal"),
    "geometry_P1_radius_13mm": (HF("geometry_P1_radius_13mm", routes=dict(R1=.013)), "nominal"),
    "geometry_P1_radius_17mm": (HF("geometry_P1_radius_17mm", routes=dict(R1=.017)), "nominal"),
    "geometry_P2_radius_9mm": (HF("geometry_P2_radius_9mm", routes=dict(R2=.009)), "nominal"),
    "geometry_P2_radius_11mm": (HF("geometry_P2_radius_11mm", routes=dict(R2=.011)), "nominal"),
    "friction_sliding_guides_mu0.05": (HF("friction_sliding_guides_mu0.05", routes=dict(guide_kind="sliding")), "nominal"),
    "friction_sliding_guides_mu0.15": (HF("friction_sliding_guides_mu0.15", routes=dict(guide_kind="sliding"),
                                          contacts=dict(sliding=dict(mu=.15))), "nominal"),
    "friction_bearings_x10": (HF("friction_bearings_x10", contacts=dict(rolling=dict(bearing_coulomb=1e-3, bearing_mu=.02)),
                                 joint_bearing=dict(coulomb=2e-3, viscous=1e-4, v_eps=.01)), "nominal"),
    "friction_presliding_x0.3": (HF("friction_presliding_x0.3", routes=dict(guide_kind="sliding"),
                                    contacts=dict(sliding=dict(presliding_stiffness=1500.))), "nominal"),
    "compliance_modulus_x0.8": (HF("compliance_modulus_x0.8", material=dict(modulus_pa=.8e10)), "nominal"),
    "compliance_modulus_x1.25": (HF("compliance_modulus_x1.25", material=dict(modulus_pa=1.25e10)), "nominal"),
    "compliance_diameter_0.4mm": (HF("compliance_diameter_0.4mm", material=dict(diameter_m=.0004)), "nominal"),
    "history_conditioned_d0.3": (HF("history_conditioned_d0.3", initial_history=[dict(d=.3), dict(d=.3)]), "nominal"),
    "history_maxwell_preloaded": (HF("history_maxwell_preloaded", initial_history=[_maxwell_history(.6), _maxwell_history(.6)]), "nominal"),
    "history_relaxation_x3": (HF("history_relaxation_x3", material=dict(relaxation_time_s=1.5)), "nominal"),
    "calibration_encoder_zero": (HF("calibration_encoder_zero"), "zero"),
    "calibration_pretension_12_8": (HF("calibration_pretension_12_8"), "pretension"),
    "combined_aged_specimen": (HF("combined_aged_specimen", history="evolving", material=dict(modulus_pa=.9e10),
                                  routes=dict(guide_kind="sliding"), contacts=dict(sliding=dict(mu=.065)),
                                  initial_history=[dict(d=.3), dict(d=.3)]), "pretension"),
}


def specimen_draw(name, rng):
    """Persistent specimen identity: a physically consistent family (category C).

    One cable batch fixes modulus and relaxation time; the guide surface condition moves friction coefficient and
    presliding stiffness together; the prior conditioning d sets stiffness loss and friction growth through the
    plant's own law; the Maxwell state records how recently the cable was loaded.
    """
    surface = rng.uniform(.75, 1.35)
    return dict(id=name, modulus_factor=float(rng.uniform(.88, 1.12)), relaxation_factor=float(rng.uniform(.7, 1.4)),
                surface_factor=float(surface), prior_d=float(rng.uniform(0., .3)), maxwell_fraction=float(rng.uniform(.7, 1.)))


def specimen_fidelity(specimen, level="H3", history=None):
    base = hardware_preset(level).to_dict()
    material = dict(modulus_pa=1e10*specimen["modulus_factor"], relaxation_time_s=.5*specimen["relaxation_factor"])
    contacts = dict(sliding=dict(mu=.05*specimen["surface_factor"], presliding_stiffness=5000.*specimen["surface_factor"]),
                    rolling=dict(presliding_stiffness=5000.*specimen["surface_factor"], bearing_mu=.002*specimen["surface_factor"]))
    hist = copy.deepcopy(history) if history else [dict(d=specimen["prior_d"]), dict(d=specimen["prior_d"])]
    for h in hist:
        h.setdefault("maxwell_fraction", specimen["maxwell_fraction"])
    base.update(level=f"{level}_{specimen['id']}", material=material, contacts=contacts, initial_history=hist,
                routes=dict(guide_kind="sliding"))
    return HardwareFidelity(**base)


def rig_A(fid):
    """Design winding matrix of the rig the run uses (recorded in the run configuration)."""
    return nominal_A(hardware_routes(**fid.routes)).tolist() if isinstance(fid, HardwareFidelity) else ARM.tolist()


def run_one(spec, fid, writer=None, **kw):
    if isinstance(fid, HardwareFidelity):
        return simulate_hardware(spec, fid, writer=writer, **kw)
    return simulate(spec, fid, writer=writer, **kw)


# ------------------------------------------------------------------------------------------------- I/O
def run_group(path, items):
    """Run a list of (spec, fidelity, kwargs) sequentially into one HDF5 file. Returns the manifest."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ds = dataset_file(str(path))
    ds.f.attrs["phase4"] = json.dumps(dict(scope="Phase 4 synthetic reference plant; category C parameters"))
    manifest = []
    try:
        for spec, fid, kw in items:
            spec["rig_A"] = rig_A(fid)
            w = episode_writer(ds, spec)
            try:
                summary, _ = run_one(spec, fid, writer=w, **kw)
            except (FloatingPointError, ValueError) as e:
                w.close(dict(failed=f"{type(e).__name__}: {e}"))
                manifest.append(dict(id=spec["id"], failed=str(e)))
                continue
            manifest.append(dict(id=spec["id"], flags=summary["flags"], fidelity=summary["fidelity"]["level"],
                                 history_out=summary["history_out"], energy_residual_J=summary["energy_residual_J"],
                                 solver=summary["solver"], mechanism=summary.get("mechanism"), rig=summary.get("rig")))
        normalize_units(ds.f)
    finally:
        ds.close()
    return manifest


def run_specimen_sessions(path, specimen, split, sessions, start_index, level="H3"):
    """Sessions of one specimen run in order; creep, settling and conditioning carry over (re-tensioned each start)."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    ds = dataset_file(str(path))
    manifest, history = [], None
    try:
        for s in range(sessions):
            motion, contact, cyc = TASKS[(start_index + s) % len(TASKS)]
            spec = specification(start_index + s, split, motion, contact, cyc, fidelity=level, specimen=specimen["id"], session=s,
                                 tag=f"_{specimen['id']}_s{s}")
            spec["specimen"] = specimen
            fid = specimen_fidelity(specimen, level, history)
            spec["rig_A"] = rig_A(fid)
            w = episode_writer(ds, spec)
            summary, _ = run_one(spec, fid, writer=w)
            history = [dict(h, maxwell_fraction=1.) for h in summary["history_out"]]  # rested between sessions
            manifest.append(dict(id=spec["id"], specimen=specimen["id"], session=s, flags=summary["flags"], history_out=summary["history_out"]))
        normalize_units(ds.f)
    finally:
        ds.close()
    return manifest


def load_many(paths, splits=None):
    runs = []
    for p in paths:
        runs += load_runs(str(p), splits)
    return runs
