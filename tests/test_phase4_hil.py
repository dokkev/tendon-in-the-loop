"""Milestone E gates: coupled two-channel HIL with the Phase 4 reference plants."""
import numpy as np
import pytest

from til import two_tendon_experiment as tte
from til.phase4_experiment import specification, simulate
from til.phase4_hil import Fidelity, Phase4Simulator, preset
from til.two_tendon import HOME


@pytest.fixture(scope="module")
def spec():
    return specification(0, "test", "coordinated", contact=True, fidelity="P0")


def test_p0_preset_is_the_existing_simulator(spec):
    _, a0 = tte.simulate(spec, duration=.8)
    _, a1 = simulate(spec, "P0", duration=.8)
    assert np.max(abs(a0 - a1)) == 0.


def test_va_route_with_lumped_cable_matches_p0(spec):
    _, a0 = simulate(spec, "P0", duration=.8)
    _, a1 = simulate(spec, Fidelity("VA", virtual="V-A", rig="none", cable="lumped"), duration=.8)
    assert np.max(abs(a0[:, 1:3] - a1[:, 1:3])) < 1e-8
    assert np.max(abs(a0[:, 13:17] - a1[:, 13:17])) < 1e-6


@pytest.fixture(scope="module")
def p2_runs(spec):
    free = dict(spec, contact=False)
    return {"contact": simulate(spec, "P2", duration=1.4, trace_extra=True),
            "free": simulate(free, "P2", duration=1.4, trace_extra=True)}


def test_pretension_equilibrium_and_single_integration(p2_runs):
    summary, a, extra = p2_runs["free"]
    hold = a[:, 0] < .6
    assert np.max(abs(np.degrees(a[hold, 1:3] - HOME))) < .2
    assert np.max(abs(a[hold, 13:17] - 10.)) < .5
    assert summary["flags"] == []
    assert all(s["fallback"] == 0 for s in summary["solver"])


def test_contact_raises_tension_and_rig_follows(p2_runs):
    free, contact = p2_runs["free"][0], p2_runs["contact"][0]
    assert contact["contact_force_peak_N"] > .1 and free["contact_force_peak_N"] == 0.
    assert max(contact["peak_input_output_tension_N"]) > max(free["peak_input_output_tension_N"])
    assert contact["rig"]["peak_tracking_deg"] < .5 and contact["rig"]["invalid_samples"] == 0


def test_distributed_tensions_and_variable_jacobian_are_recorded(p2_runs):
    _, a, extra = p2_runs["contact"]
    T = np.array([e["segment_tensions"][0] for e in extra])
    assert T.shape[1] == 6 and np.all(T >= 0)
    J = np.array([e["jacobian"] for e in extra])
    assert np.ptp(J[:, 0, 0]) > 1e-4  # configuration-dependent virtual Jacobian under V-B


def test_telemetry_only_corruption_leaves_mechanics_unchanged(spec):
    clean = simulate(dict(spec, contact=False), Fidelity("P2"), duration=.5)[1]
    noisy = simulate(dict(spec, contact=False), Fidelity("P2s", sensing="imperfect"), duration=.5)[1]
    assert np.max(abs(clean[:, 1:3] - noisy[:, 1:3])) == 0.
    assert np.max(abs(clean[:, 13:17] - noisy[:, 13:17])) == 0.
    assert np.max(abs(clean[:, 5:13] - noisy[:, 5:13])) > 0.  # the motor channels differ


def test_rb_route_is_compensated_by_the_emulator(spec):
    summary, a = simulate(dict(spec, contact=False, motion="proximal"), Fidelity("RB", rig="R-B"), duration=1.4)
    assert min(a[:, 13:17].min(0)) > 5.
    assert summary["rig"]["invalid_samples"] == 0
