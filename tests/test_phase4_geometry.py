"""Milestone B gates: virtual (V-A, V-B) and rig (R-A, R-B) geometry."""
from math import pi

import mujoco
import numpy as np
import pytest

from til.finger import FingerParams
from til.phase4_geometry import (P0_TOTAL_LENGTH, Phase4Finger, VirtualRoute, joint_grid, rig_route, va_route,
                                 vb_analytic, vb_route, vb_topology)
from til.two_tendon import ARM, HOME, TwoTendonFinger

GRID = joint_grid()  # 11 x 13 = 143 points over q1 = -30..60 deg, q2 = 0..150 deg


@pytest.fixture(scope="module")
def va():
    return Phase4Finger(route=va_route())


@pytest.fixture(scope="module")
def vb():
    return Phase4Finger(route=vb_route())


def fd_jacobian(finger, q, h=1e-6):
    cols = []
    for j in range(2):
        e = np.eye(2)[j]*h
        cols.append((finger.lengths_and_jacobian(q + e)[0] - finger.lengths_and_jacobian(q - e)[0])/(2*h))
    return np.column_stack(cols)


def test_va_reproduces_constant_arm_reference_on_full_grid(va):
    l0 = va.lengths_and_jacobian(HOME)[0]
    assert len(GRID) == 143
    for q in GRID:
        l, J = va.lengths_and_jacobian(q)
        assert np.max(abs(J + ARM)) < 1e-8
        assert np.max(abs((l - l0) + ARM @ (q - HOME))) < 1e-8


def test_va_dynamics_match_p0_fixed_tendons(va):
    p0 = TwoTendonFinger()
    finger = Phase4Finger(route=va_route())
    for f in (p0, finger):
        f.reset(HOME, (10., 10.))
    rng = np.random.default_rng(3)
    for k in range(400):
        T = 10 + 6*np.sin(np.array([.01, .017])*k) + rng.uniform(0, .5, 2)
        for f in (p0, finger):
            f.apply_tension(T); f.step()
    assert np.max(abs(p0.q - finger.q)) < 1e-8
    assert np.max(abs(p0.qd - finger.qd)) < 1e-6


def test_vb_matches_jacobian_at_home_and_varies(vb):
    l, J = vb.lengths_and_jacobian(HOME)
    assert np.max(abs(J + ARM)) < 1e-9
    arms = np.array([vb_topology(vb.route, q)["arms"] for q in GRID])
    variation = (arms.max(0) - arms.min(0))/abs(np.array([ARM[0, 0], ARM[1, 0], ARM[1, 1]]))
    assert np.all(variation >= .2)


def test_vb_mujoco_analytic_and_finite_difference_agree(vb):
    for q in GRID:
        l, J = vb.lengths_and_jacobian(q)
        la, Ja = vb_analytic(vb.route, q)
        assert np.max(abs(J - Ja)) < 1e-10
        assert np.max(abs(fd_jacobian(vb, q) - J)) < 1e-7
        assert vb_topology(vb.route, q)["valid"]


def test_coordinated_perturbation_is_first_order_consistent(vb):
    direction = np.array([1., 1.])/np.sqrt(2)
    for q in GRID[::7]:
        h = 1e-6
        dl = (vb.lengths_and_jacobian(q + h*direction)[0] - vb.lengths_and_jacobian(q - h*direction)[0])/(2*h)
        assert np.max(abs(dl - vb.lengths_and_jacobian(q)[1] @ direction)) < 1e-7


@pytest.mark.parametrize("route", ["V-A", "V-B"])
def test_virtual_work_and_generalized_force(route):
    finger = Phase4Finger(route=va_route() if route == "V-A" else vb_route())
    rng = np.random.default_rng(1)
    for q in GRID[::5]:
        finger.data.qpos[finger.qadr] = q; finger.data.qvel[:] = 0
        T = rng.uniform(1, 40, 2)
        finger.apply_tension(T)
        mujoco.mj_forward(finger.model, finger.data)
        J = finger.tendon_jacobian()
        tau = finger.data.qfrc_actuator[finger.dof]
        assert np.max(abs(tau + J.T @ T)) < 1e-8
        dq = rng.normal(0, 1e-7, 2)
        dl = finger.lengths_and_jacobian(q + dq)[0] - finger.lengths_and_jacobian(q)[0]
        assert abs(tau @ dq + T @ dl) < 1e-8*np.linalg.norm(T)*np.linalg.norm(dq)  # second order in dq


def test_pulley_radius_variation_scales_constant_arms():
    route = VirtualRoute("V-A", radii=(.018, .012, .008))
    finger = Phase4Finger(route=route)
    for q in GRID[::11]:
        J = finger.lengths_and_jacobian(q)[1]
        assert np.max(abs(J + np.array([[.018, 0], [.012, .008]]))) < 1e-8


def test_ra_length_is_invariant_while_wraps_change():
    route = rig_route("R-A", total_length=P0_TOTAL_LENGTH)
    qs = np.linspace(-pi/6, pi/3, 91)
    lengths = np.array([route.evaluate(q)["total_length"] for q in qs])
    wraps = np.array([route.evaluate(q)["wraps"] for q in qs])
    assert np.ptp(lengths) < 1e-9
    assert abs(lengths[0] - P0_TOTAL_LENGTH) < 1e-9
    assert np.allclose(wraps[:, 1] + qs, pi/2, atol=1e-9) and np.allclose(wraps[:, 3] - qs, pi/2, atol=1e-9)
    assert np.allclose(wraps[:, 2], pi, atol=1e-9)
    assert abs(route.segment_jacobian(.4).sum()) < 1e-8
    assert all(route.evaluate(q)["valid"] for q in qs)


@pytest.mark.parametrize("kind", ["R-A", "R-B"])
def test_rig_tangency_and_continuity(kind):
    route = rig_route(kind, total_length=P0_TOTAL_LENGTH)
    previous = None
    for q in np.linspace(-pi/6, pi/3, 181):
        e = route.evaluate(q)
        for i, (pa, pb) in enumerate(e["tangent_points"]):
            d = (pb - pa)/np.hypot(*(pb - pa))
            for node, center, p in ((route.nodes[i], e["centers"][i], pa), (route.nodes[i+1], e["centers"][i+1], pb)):
                if node.radius:
                    assert abs(np.hypot(*(p - center)) - node.radius) < 1e-12
                    assert abs((p - center) @ d) < 1e-12  # radius perpendicular to the free span
        if previous is not None:
            assert np.max(abs(e["segment_lengths"] - previous)) < 1e-3  # 0.5 deg steps, no jumps
        previous = e["segment_lengths"]


def test_rb_length_varies_and_radius_variation_keeps_ra_invariant():
    rb = rig_route("R-B", total_length=P0_TOTAL_LENGTH)
    lengths = [rb.evaluate(q)["total_length"] for q in np.linspace(-pi/6, pi/3, 31)]
    assert np.ptp(lengths) > .005
    for radius in (.008, .012, .016):
        ra = rig_route("R-A", idler_radius=radius)
        assert np.ptp([ra.evaluate(q)["total_length"] for q in np.linspace(-pi/6, pi/3, 31)]) < 1e-9


def test_topology_transitions_are_detected():
    # Wrap loss on joint idler 1 (wrap = pi/2 - q) when the rig link is driven past the designed range.
    route = rig_route("R-A", total_length=P0_TOTAL_LENGTH)
    route.q_range = (-pi/6, 1.7)
    qs = np.linspace(-pi/6, 1.7, 200)
    flags = np.array([route.evaluate(q)["valid"] for q in qs])
    assert flags[qs < pi/2 - np.deg2rad(2.1)].all() and not flags[qs > pi/2 - np.deg2rad(1.9)].any()
    # Crossing (opposite-direction) overlapping contacts have no common tangent: rejected, not silently routed.
    from til.phase4_geometry import _common_tangent
    with pytest.raises(ValueError):
        _common_tangent(np.zeros(2), .012, np.array([.02, 0.]), -.012)
    # Out of the declared rig range.
    assert not rig_route("R-A").evaluate(1.2)["valid"]
    # V-B site pushed through the joint: the moment arm collapses and the flag is raised.
    route = vb_route()
    route.sites["t1_link1"] = (.02, .0)
    route.sites["t1_base"] = (-.02, .0)
    assert not vb_topology(route, np.array([0., .2]))["valid"]
