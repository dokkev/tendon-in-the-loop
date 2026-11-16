"""Milestones C and D gates: distributed cable limits, friction, viscoelasticity, history, energy."""
from math import pi, exp

import numpy as np
import pytest

from til.phase4_cable import CableChainParams, CableMaterial, ContactParams, DistributedCablePlant, chain_from_route
from til.phase4_geometry import P0_TOTAL_LENGTH, rig_route
from til.transmission import TransmissionParams, TransmissionPlant
from til.two_tendon import HOME

DT = 1e-4


def drive(plant, n, x_in, x_out=lambda t: 0., dt=DT):
    out = []
    for k in range(1, n + 1):
        t = k*dt
        h = 1e-7
        xi, xo = x_in(t), x_out(t)
        out.append(plant.step(dt, xi, (x_in(t + h) - x_in(t - h))/(2*h), xo, (x_out(t + h) - x_out(t - h))/(2*h)))
    return np.array(out)


def elastic_material(**kw):
    return CableMaterial(nonlinear_strain_coefficient=0., maxwell_ratio=0., viscous_Ns=0., **kw)


def phase2_equivalent():
    p2 = TransmissionPlant(TransmissionParams())
    L = p2.lengths
    c = ContactParams("equivalent_pulley", "rolling", radius=.02, wrap=pi/2, inertia=2e-6, mu=.05, level_model="capstan",
                      bearing_coulomb=.001, bearing_viscous=1e-5, presliding_stiffness=2/.02**2, bending_coefficient=1e-4,
                      bending_radius=.03)
    return p2, DistributedCablePlant(CableChainParams(CableMaterial(viscous_Ns=L[0], linear_density_kg_per_m=0., taut_regularization_m=0.), [c], L, 10.))


def test_one_contact_limit_reproduces_phase2_plant():
    p2, d = phase2_equivalent()
    xi = lambda t: .002*np.sin(2*pi*.5*t) + .0005*np.sin(2*pi*3*t)
    xo = lambda t: -.001*np.sin(2*pi*.7*t)
    a, b = drive(p2, 15000, xi, xo), drive(d, 15000, xi, xo)
    assert np.max(abs(a - b)) < 1e-6
    # Phase 2 does not separate the implicit-Euler inertia loss; compare the same (uncorrected) ledger.
    assert abs(p2.ground_truth()["energy_residual"] - (d.energy_residual() + d.E_numerical)) < 1e-9


def test_frictionless_contacts_are_springs_in_series():
    route = rig_route("R-A", total_length=P0_TOTAL_LENGTH)
    params = chain_from_route(route, HOME[0], material=elastic_material(), friction="frictionless",
                              contact_overrides={"rolling": dict(bearing_viscous=0., inertia=1e-9)})
    d = DistributedCablePlant(params)
    drive(d, 4000, lambda t: min(t, .2)*.005)  # ramp to 1 mm, then hold
    k_series = d.p.material.EA/np.sum(params.segment_lengths)
    assert np.ptp(d.T) < 1e-6
    assert abs(d.T[0] - (10 + k_series*.001)) < 1e-6


@pytest.mark.parametrize("beta", [pi/4, pi/2, pi])
def test_ideal_sliding_capstan_ratio(beta):
    mu = .2
    c = ContactParams("guide", "sliding", radius=.01, wrap=beta, mu=mu, presliding_stiffness=5e5)
    d = DistributedCablePlant(CableChainParams(elastic_material(), [c], (.2, .2), 10.))
    drive(d, 6000, lambda t: .01*t)  # steady pull; the far end is fixed
    assert d.w[0] > 0  # material slides toward the actuator
    assert abs(abs(d.f[0]) - d._levels(d.T)[0]) < 1e-9  # on the sliding bound
    assert abs(d.T[0]/d.T[1] - exp(mu*beta))/exp(mu*beta) < .01
    # radius does not enter the exponent
    c2 = ContactParams("guide", "sliding", radius=.03, wrap=beta, mu=mu, presliding_stiffness=5e5)
    d2 = DistributedCablePlant(CableChainParams(elastic_material(), [c2], (.2, .2), 10.))
    drive(d2, 6000, lambda t: .01*t)
    assert abs(d2.T[0]/d2.T[1] - d.T[0]/d.T[1]) < 1e-6


def test_presliding_reversal_and_hysteresis():
    c = ContactParams("guide", "sliding", radius=.01, wrap=pi/2, mu=.1, presliding_stiffness=5000.)
    small = DistributedCablePlant(CableChainParams(elastic_material(), [c], (.2, .2), 10.))
    drive(small, 20000, lambda t: 2e-5*np.sin(2*pi*t))  # below the sliding level: recoverable presliding
    assert small.E_friction < 1e-9
    large = DistributedCablePlant(CableChainParams(elastic_material(), [c], (.2, .2), 10.))
    T = drive(large, 20000, lambda t: 1e-3*np.sin(2*pi*t))
    assert large.E_friction > 1e-5
    # Direction-dependent transmission: at the same input displacement the output differs on load/unload.
    t = np.arange(1, 20001)*DT
    up = (t > 1.) & (np.cos(2*pi*t) > 0) & (abs(np.sin(2*pi*t)) < .05)
    down = (t > 1.) & (np.cos(2*pi*t) < 0) & (abs(np.sin(2*pi*t)) < .05)
    assert T[up, 1].mean() < T[down, 1].mean() - .05


def test_slack_and_retensioning():
    c = ContactParams("guide", "sliding", radius=.01, wrap=pi/2, mu=.05)
    d = DistributedCablePlant(CableChainParams(CableMaterial(), [c], (.2, .2), 10.))
    T = drive(d, 20000, lambda t: -.004*np.sin(pi*t/2))  # pays out beyond the pretension stretch, then returns
    assert T.min() == 0. and np.all(T >= 0)
    assert d.T[0] > 9.  # re-engaged near pretension at the end
    assert abs(d.energy_residual()) < 1e-5


def test_multiple_guides_tension_distribution_and_inertia():
    route = rig_route("R-A", total_length=P0_TOTAL_LENGTH)
    d = DistributedCablePlant(chain_from_route(route, HOME[0]))
    drive(d, 8000, lambda t: .004*min(t, .5))
    ratios = d.T[:-1]/d.T[1:]
    assert np.all(np.diff(d.T) < 0)  # friction opposes the pull: tension falls from actuator to emulator
    bound = np.array([exp(c.mu*b) if c.kind == "sliding" else np.inf for c, b in zip(d.p.contacts, d.wraps)])
    assert np.all(ratios <= bound*(1 + 1e-6))
    # nonzero idler inertia: a fast pull produces an inertial tension difference across a rolling idler
    fast = DistributedCablePlant(chain_from_route(route, HOME[0], friction="frictionless",
                                                  contact_overrides={"rolling": dict(inertia=2e-5, bearing_viscous=0.)}))
    i = 2  # link idler is contact index 2; segments 2 and 3 straddle it
    gaps = []
    for k in range(1, 400):
        t = k*DT
        w = fast.w[i]
        fast.step(DT, .5*t*t, t, 0., 0.)
        inertial = fast.c[i]*(fast.w[i] - w)/DT
        assert abs((fast.T[i] - fast.T[i + 1]) - inertial) < 1e-8
        gaps.append(fast.T[i] - fast.T[i + 1])
    assert max(gaps) > 1e-2


@pytest.mark.parametrize("kind", ["R-A", "R-B"])
def test_moving_routing_support_energy_and_length(kind):
    route = rig_route(kind, total_length=P0_TOTAL_LENGTH)
    d = DistributedCablePlant(chain_from_route(route, HOME[0]))
    q = lambda t: HOME[0] + .4*np.sin(2*pi*.5*t)
    l0 = route.evaluate(HOME[0])["total_length"]
    outer = 5
    for k in range(4000):
        t0, t1 = k*outer*DT, (k + 1)*outer*DT
        L0, L1 = route.segment_lengths(q(t0)), route.segment_lengths(q(t1))
        d.set_geometry(L0, (L1 - L0)/(outer*DT), route.evaluate(q(t0))["wraps"])
        # emulator compensation of the rig path change (as in the HIL): x_out = -(l_R(q) - l_R(q0))
        x0, x1 = -(L0.sum() - l0), -(L1.sum() - l0)
        for j in range(outer):
            d.step(DT, 0., 0., x0 + (x1 - x0)*(j + 1)/outer, (x1 - x0)/(outer*DT))
        assert np.max(abs(d.L - L1)) < 1e-12
    gt = d.ground_truth()
    assert abs(gt["energy_residual"]) < 1e-5
    assert gt["support_work"] != 0.
    assert d.T.min() > 5.  # compensated: stays taut
    if kind == "R-A":
        assert abs(np.sum(d.L) - np.sum(route.segment_lengths(HOME[0]))) < 1e-9


def test_sls_relaxation_matches_analytic():
    m = CableMaterial(nonlinear_strain_coefficient=0., viscous_Ns=0.)
    d = DistributedCablePlant(CableChainParams(m, [], (.4,), 10.))
    step = .001
    k = d.k[0]; km = m.maxwell_ratio*k
    T = drive(d, 20000, lambda t: step)  # instantaneous stretch, then held (constant displacement)
    t = np.arange(1, 20001)*DT
    e = d.e[0]; e0 = e - step
    analytic = k*e + km*step*np.exp(-t/m.relaxation_time_s)
    assert np.max(abs(T[:, 0] - analytic)/analytic) < .01


def test_constant_force_creep_matches_sls_plus_norton():
    m = CableMaterial(nonlinear_strain_coefficient=0., viscous_Ns=0., creep_strain_rate=2e-4, creep_threshold_N=12.,
                      creep_reference_N=18., creep_exponent=2.)
    d = DistributedCablePlant(CableChainParams(m, [], (.4,), 10.))
    F = 20.
    k = d.k[0]; km = m.maxwell_ratio*k; a = DT/m.relaxation_time_s
    xs = []
    for n in range(30000):
        # Dead-weight boundary: the end displacement that gives T = F under the implicit Maxwell update.
        e = (F + km*d.v[0]/(1 + a))/(k + km/(1 + a))
        x = e - (d.L[0] - d.M[0] - d.creep[0])
        d.step(DT, x, 0., 0., 0.)
        assert abs(d.T[0] - F) < 1e-4  # the step's creep increment is applied after the solve
        xs.append(x)
    t = np.arange(30000)*DT
    e_inf, e_start = (F - 10)/k, (F - 10)/(k + km)
    tau_c = m.relaxation_time_s*(k + km)/k
    norton = .4*m.creep_strain_rate*((F - 12.)/18.)**2*t
    analytic = e_inf - (e_inf - e_start)*np.exp(-t/tau_c) + norton
    assert abs(d.creep[0] - norton[-1] - .4*m.creep_strain_rate*((F - 12.)/18.)**2*DT)/norton[-1] < .01
    assert np.max(abs(np.array(xs) - analytic)/analytic) < .01


def test_isolated_energy_residual_converges_with_step_refinement():
    route = rig_route("R-A", total_length=P0_TOTAL_LENGTH)
    residuals = []
    for dt in (4e-4, 2e-4, 1e-4):
        d = DistributedCablePlant(chain_from_route(route, HOME[0]))
        drive(d, int(round(2./dt)), lambda t: .003*np.sin(2*pi*t), lambda t: -.001*np.sin(2*pi*t), dt=dt)
        residuals.append(abs(d.energy_residual()))
        assert d.E_numerical < 1e-6
    assert residuals[2] < residuals[1] < residuals[0]
    assert residuals[2]/2 < 1e-6  # per 1 s cycle


def test_history_persists_across_sessions_and_retension_preserves_record():
    m = CableMaterial(creep_strain_rate=5e-4, conditioning_per_m=2.)
    route = rig_route("R-A", total_length=P0_TOTAL_LENGTH)
    p = chain_from_route(route, HOME[0], material=m)
    d = DistributedCablePlant(p)
    drive(d, 5000, lambda t: .004*np.sin(2*pi*t))
    history = d.export_history()
    assert history["d"] > 0 and max(history["creep"]) > 0
    p.history = history
    d2 = DistributedCablePlant(p)
    assert d2.d == history["d"] and np.allclose(d2.creep, history["creep"])
    assert abs(d2.T[0] - 10.) < 1e-6  # session start re-tensioned; history kept
    p.retension_at_reset = False
    d3 = DistributedCablePlant(p)
    assert d3.T[0] < 10.  # without re-tensioning the accumulated creep lowers the pretension
