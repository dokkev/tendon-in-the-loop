"""Hardware-topology HIL gates: physical joints, distributed tendons on the intended routes, force-rendered load."""
import numpy as np
import pytest

from til.phase4_experiment import resolved_configuration, simulate_hardware, specification
from til.phase4_geometry import hardware_routes
from til.phase4_hardware import HardwareFidelity, HardwareHIL, hardware_preset, nominal_A
from til.two_tendon import ARM, HOME


def test_axial_mcp_passage_decouples_the_pip_tendon():
    t1, t2 = hardware_routes()
    for q in np.linspace(-np.pi/6, np.pi/3, 19):
        assert abs(t1.segment_jacobian(q).sum()) < 1e-12  # tendon 1 path does not depend on q
        assert np.max(abs(t2.segment_jacobian(q))) == 0.  # no strand of tendon 2 changes length with q1
    assert np.allclose(nominal_A(hardware_routes()), [[.015, 0.], [0., .010]])
    p1 = t1.contacts[0]; idlers = [n for n in t2.contacts if n.name.startswith("mcp_")]
    assert p1.kind == "joint" and p1.radius == .015  # MCP pulley: driven, fixed to the link
    assert len(idlers) == 4 and all(n.kind == "rolling" for n in idlers)  # MCP idler pulleys: free, distinct from P1


def test_coaxial_variant_realises_the_coupled_matrix():
    t1, t2 = hardware_routes(mcp_passage="coaxial")
    for q in np.linspace(-np.pi/6, np.pi/3, 19):
        G = t2.segment_jacobian(q)
        assert abs(G[:3].sum() + .010) < 1e-9 and abs(G.sum()) < 1e-9
    assert np.allclose(nominal_A((t1, t2)), ARM)


def test_static_equilibrium_at_home():
    spec = specification(0, "test", "proximal", fidelity="H2")
    configs, fp = resolved_configuration(spec, .0005, .0001, 0., False)
    sim = HardwareHIL(configs, fp, hardware_preset("H2"))
    for ch in sim.ch:
        ch["motor"].s.i_ff = 2.
    for _ in range(600):
        sim.step(np.zeros(2))
    assert np.max(abs(sim.mech.q - HOME)) < 1e-6
    for ch in sim.ch:
        assert np.max(abs(ch["plant"].T - 10.)) < 1e-4


def test_pip_tendon_puts_no_torque_on_the_mcp():
    spec = specification(0, "test", "proximal", fidelity="H1")
    configs, fp = resolved_configuration(spec, .0005, .0001, 0., False)
    sim = HardwareHIL(configs, fp, hardware_preset("H1"))
    for ch in sim.ch:
        ch["motor"].s.i_ff = 2.
    sim.ch[1]["motor"].s.i_ff = 3.  # extra PIP motor torque only
    ch2 = sim.ch[1]
    for _ in range(400):
        sim.step(np.zeros(2))
        # tendon 2's generalized force on the MCP: joint pulleys on dof 0 (none) minus the routing term (zero)
        on_mcp = sum(r*ch2["plant"].joint_forces()[k] for k, dof, r in ch2["joints"] if dof == 0) - ch2["plant"].T @ ch2["route"].segment_jacobian(sim.mech.q[0])
        assert on_mcp == 0.
    assert sim.mech.q[1] - HOME[1] > .01  # the PIP moved
    assert abs(sim.mech.q[0] - HOME[0]) < 1e-4  # the MCP moves only through the inertial (mass-matrix) coupling


@pytest.fixture(scope="module")
def free_h1():
    return simulate_hardware(specification(0, "test", "coordinated", fidelity="H1"), "H1", duration=1.6, trace_extra=True)


def test_frictionless_generalized_force_is_A_transpose(free_h1):
    s, a, extra = free_h1
    A = np.array(s["nominal_winding_matrix"]); assert A[1, 0] == 0.
    worst = 0.
    for e in extra[50:]:
        Ta = np.array([t[0] for t in e["segment_tensions"]]); Te = np.array([t[-1] for t in e["segment_tensions"]])
        worst = max(worst, np.max(abs(e["cable_torque"] - A.T @ (Ta - Te))))
    assert worst < 5e-4  # N m; residual = idler and cable-mass inertia


def test_motor_kinematics_match_the_equivalent_model(free_h1):
    s, a, _ = free_h1
    x = .02*a[:, [5, 9]]
    model = (a[:, 1:3] - HOME) @ np.array(s["nominal_winding_matrix"]).T
    assert np.mean(abs(x - model)) < 5e-4  # m; the difference is cable stretch under the tension changes
    assert s["flags"] == [] and abs(s["mechanism"]["residual_J"]) < 1e-6
    assert max(abs(r) for r in s["energy_residual_J"]) < 1e-5


def test_contact_is_rendered_through_the_emulators():
    s, a, extra = simulate_hardware(specification(0, "test", "coordinated", contact=True, fidelity="H2"), "H2", duration=2.4, trace_extra=True)
    assert s["contact_force_peak_N"] > 1.
    cmd = np.array([e["commands"] for e in extra])
    assert cmd[:, 0].max() > 12.  # emulator 1 pulls back against MCP flexion while the pad is pressed
    assert "render_command_clipped" not in s["flags"]


def test_telemetry_only_corruption_leaves_mechanics_unchanged():
    spec = specification(0, "test", "distal", fidelity="H2")
    clean = simulate_hardware(spec, "H2", duration=.6)[1]
    noisy = simulate_hardware(spec, HardwareFidelity("H2s", sensing="imperfect"), duration=.6)[1]
    assert np.max(abs(clean[:, 1:3] - noisy[:, 1:3])) == 0.
    assert np.max(abs(clean[:, 5:13] - noisy[:, 5:13])) > 0.
