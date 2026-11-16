"""Phase 4 coupled two-channel HIL: P0-P3 reference plants behind one configuration object.

The loop is the P0 loop (two inherited causal chains, one shared MuJoCo finger integrated once per step).
What changes with the fidelity level:

* virtual geometry: P0 fixed tendons, or a Phase4Finger spatial route (V-A, V-B);
* rig: none, or a rig route (R-A, R-B) whose link is driven by a finite position servo following the
  virtual q1; the cable torque on the link and the servo work are recorded;
* cable: the Phase 2 lumped plant, or the distributed plant on the rig route;
* history: fixed parameters, or creep / settling / conditioning plus a persistent specimen history;
* sensing: telemetry-only corruption (current gain and bias, current-bias drift, encoder drift) that never
  reaches the motor's inner loop;
* hardware: inner-loop imperfections (true torque constant error, motor Coulomb friction, slower controller
  update, emulator delay), which change the actual mechanics.

Emulator command (sign convention of TiLSimulator, x_out > 0 stretches the cable):
    x_out = [l_v(q) - l_v(q0)] - [l_R(q_r) - l_R(q_r0)]
"""
from dataclasses import dataclass, field, asdict, replace
import copy

import numpy as np

from .coupling import TiLSimulator
from .finger import FingerParams
from .phase4_cable import CableMaterial, DistributedCablePlant, chain_from_route
from .phase4_geometry import P0_TOTAL_LENGTH, Phase4Finger, rig_route, va_route, vb_route
from .two_tendon import HOME, LimitedMotor, TendonPort, TwoTendonFinger, TwoTendonSimulator


@dataclass
class Fidelity:
    level: str = "P2"
    virtual: str = "V-B"  # "P0" (fixed tendons), "V-A", "V-B"
    rig: str = "R-A"  # "none", "R-A", "R-B"
    cable: str = "distributed"  # "lumped" (Phase 2 plant) or "distributed"
    friction: str = "presliding"  # or "frictionless" (distributed only)
    viscoelastic: bool = True
    history: str = "fixed"  # "fixed" or "evolving"
    sensing: str = "clean"  # "clean" (P0 telemetry) or "imperfect" (telemetry-only corruption)
    hardware: str = "nominal"  # "nominal" or "imperfect" (inner-loop)
    material: dict = field(default_factory=dict)  # CableMaterial overrides (both channels)
    contacts: dict = field(default_factory=dict)  # chain_from_route contact overrides
    route: dict = field(default_factory=dict)  # rig_route keyword overrides
    virtual_route: dict = field(default_factory=dict)  # vb_route keyword overrides
    initial_history: list = field(default_factory=lambda: [{}, {}])  # per channel persistent specimen state
    retension_at_reset: bool = True
    rig_servo: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def preset(level):
    """The four reference-plant fidelity levels (PHASE4_DESIGN.md)."""
    if level == "P0":
        return Fidelity("P0", virtual="P0", rig="none", cable="lumped")
    if level == "P1":
        return Fidelity("P1", virtual="V-B", rig="none", cable="lumped")
    if level == "P2":
        return Fidelity("P2")
    if level == "P3":
        return Fidelity("P3", history="evolving", sensing="imperfect", hardware="imperfect")
    raise ValueError(level)


EVOLVING_MATERIAL = dict(creep_strain_rate=1e-4, creep_threshold_N=12., creep_reference_N=18., creep_exponent=2.,
                         settle_rate_m_per_Ns=2e-6, settle_cap_m=4e-4, conditioning_per_m=1.5)
IMPERFECT_SENSING = dict(current_gain=(1.03, .97), current_bias_A=(.01, -.008), bias_drift_A_per_sqrt_s=2e-3,
                         encoder_drift_rad_per_s=(1e-4, -1e-4))
IMPERFECT_HARDWARE = dict(K_t=(.097, .103), tau_c_m=.002, controller_period=5e-4, emulator_delay=1e-3)


@dataclass
class RigServoParams:
    inertia: float = 5e-4  # kg m^2, rig link about its joint
    viscous: float = 1e-3  # N m s/rad
    Kp: float = 200.  # N m/rad (about 100 Hz with the link inertia)
    Kd: float = .5  # N m s/rad (damping ratio about 0.8)
    torque_limit: float = 5.  # N m


class RigLink:
    """Finite position servo driving the rig link toward the virtual q1 (semi-implicit Euler)."""
    def __init__(self, params=None, q0=HOME[0]):
        self.p = params or RigServoParams()
        self.q, self.qd = float(q0), 0.
        self.torque = self.cable_torque = 0.
        self.work = self.cable_work = 0.
        self.saturated = False

    def step(self, dt, q_ref, qd_ref, cable_torque):
        p = self.p
        raw = p.Kp*(q_ref - self.q) + p.Kd*(qd_ref - self.qd)
        self.torque = float(np.clip(raw, -p.torque_limit, p.torque_limit))
        self.saturated = self.torque != raw
        self.cable_torque = float(cable_torque)
        # implicit in the damping terms
        qd = (p.inertia*self.qd + dt*(self.torque + cable_torque))/(p.inertia + dt*p.viscous)
        self.work += self.torque*qd*dt
        self.cable_work += cable_torque*qd*dt
        self.qd = qd
        self.q += dt*qd


class RigPort(TendonPort):
    """Emulator boundary of one channel: virtual displacement minus the rig path change."""
    def __init__(self, finger, index, owner):
        super().__init__(finger, index)
        self.owner = owner

    def tendon_displacement(self):
        return self.finger.tendon_displacement()[self.index] - self.owner.rig_displacement[self.index]


class Phase4Simulator(TwoTendonSimulator):
    def __init__(self, configs, finger_params=None, fidelity=None, home=HOME):
        self.fidelity = fid = copy.deepcopy(fidelity or preset("P2"))
        self.configs = copy.deepcopy(configs)
        p = copy.deepcopy(finger_params or FingerParams(q1_range=(-np.pi/6, np.pi/3), q2_range=(0., 5*np.pi/6), T_max=80.))
        p.timestep = self.configs[0].finger.timestep
        if any(c.finger.timestep != p.timestep for c in self.configs):
            raise ValueError("shared outer timestep required")
        if fid.hardware == "imperfect":
            for i, c in enumerate(self.configs):
                c.motor.K_t = IMPERFECT_HARDWARE["K_t"][i]
                c.motor.tau_c_m = IMPERFECT_HARDWARE["tau_c_m"]
                c.emulator.delay = IMPERFECT_HARDWARE["emulator_delay"]
        if fid.sensing == "imperfect":
            for i, c in enumerate(self.configs):
                c.motor.current_gain = IMPERFECT_SENSING["current_gain"][i]
                c.motor.current_bias = IMPERFECT_SENSING["current_bias_A"][i]
                c.motor_sensor.bias_drift_A_per_sqrt_s = IMPERFECT_SENSING["bias_drift_A_per_sqrt_s"]
        if fid.virtual == "P0":
            self.finger = TwoTendonFinger(p)
        else:
            route = va_route() if fid.virtual == "V-A" else vb_route(p, **fid.virtual_route)
            self.finger = Phase4Finger(p, route)
        self.finger.reset(home, [c.cable.pretension for c in self.configs])
        self.rig = None
        self.rig_route = None
        self.rig_displacement = np.zeros(2)
        if fid.rig != "none":
            self.rig_route = rig_route(fid.rig, **{"total_length": P0_TOTAL_LENGTH, **fid.route})
            self.rig = RigLink(RigServoParams(**fid.rig_servo), home[0])
            self.rig_l0 = self.rig_route.evaluate(self.rig.q)["total_length"]
        self.chains = []
        for i, c in enumerate(self.configs):
            chain = TiLSimulator(c)
            chain.reset()
            if fid.cable == "distributed":
                if self.rig_route is None:
                    raise ValueError("the distributed cable needs a rig route")
                material = dict(asdict(CableMaterial()), **fid.material)
                if not fid.viscoelastic:
                    material["maxwell_ratio"] = 0.
                if fid.history == "evolving":
                    material.update({k: v for k, v in EVOLVING_MATERIAL.items() if k not in fid.material})
                params = chain_from_route(self.rig_route, self.rig.q, material=CableMaterial(**material),
                                          pretension=c.cable.pretension, contact_overrides=fid.contacts,
                                          friction=fid.friction, history=copy.deepcopy(fid.initial_history[i]),
                                          retension_at_reset=fid.retension_at_reset)
                chain.plant = DistributedCablePlant(params)
            motor = LimitedMotor(c.motor, rng=np.random.default_rng(c.seed + 1))
            motor.reset(c.cable.pretension)
            chain.motor = motor
            chain.finger = RigPort(self.finger, i, self)
            self.chains.append(chain)
        self.encoder_drift = np.array(IMPERFECT_SENSING["encoder_drift_rad_per_s"]) if fid.sensing == "imperfect" else np.zeros(2)
        self.t = 0.
        self.k = 0
        self.virtual_work = np.zeros(2)
        self.physical_output_work = np.zeros(2)

    @property
    def distributed(self):
        return self.fidelity.cable == "distributed"

    def _rig_update(self):
        """Advance the rig link one outer step and hand the plants their geometry for the substeps."""
        if self.rig is None:
            return
        route, rig, dt = self.rig_route, self.rig, self.dt
        q_start = rig.q
        geo = route.evaluate(q_start)
        if not geo["valid"]:
            self.flags_invalid_rig = True
        L0 = geo["segment_lengths"]
        torque = 0.
        if self.distributed:
            dL = route.segment_jacobian(q_start)
            torque = -sum(float(ch.plant.T @ dL) for ch in self.chains)
        rig.step(dt, float(self.finger.q[0]), float(self.finger.qd[0]), torque)
        L1 = route.segment_lengths(rig.q)
        # Emulator compensation uses the rig angle at the start of the step (causal, measured).
        self.rig_displacement = np.full(2, geo["total_length"] - self.rig_l0)
        if self.distributed:
            for ch in self.chains:
                ch.plant.set_geometry(L0, (L1 - L0)/dt, geo["wraps"])
        self.rig_geometry = geo

    def step(self, displacement, velocity=(0., 0.)):
        self._rig_update()
        super().step(displacement, velocity)

    def record(self):
        result = super().record()
        for i in (1, 2):
            if self.encoder_drift[i-1]:
                result[f"motor{i}"]["position"] += self.encoder_drift[i-1]*self.t  # telemetry only
        if self.rig is not None:
            geo = getattr(self, "rig_geometry", self.rig_route.evaluate(self.rig.q))
            result["rig"] = dict(angle=self.rig.q, velocity=self.rig.qd, tracking_error=self.rig.q - float(self.finger.q[0]),
                                 servo_torque=self.rig.torque, cable_torque=self.rig.cable_torque, servo_work=self.rig.work,
                                 cable_work_on_link=self.rig.cable_work, servo_saturated=float(self.rig.saturated),
                                 path_length=geo["total_length"], wraps=geo["wraps"], valid=float(geo["valid"]),
                                 emulator_compensation=self.rig_displacement[0])
        return result
