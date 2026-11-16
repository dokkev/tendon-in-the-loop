"""Phase 4 hardware-topology HIL: grounded motors, physical MCP link and PIP pulley, force-rendered virtual load.

Causality (user decision 2026-11-16):
* Mechanism: the rig link (q1, carrying P1 at the MCP) and the PIP pulley P2 (q2, relative to the link) are physical
  degrees of freedom. They move only under the cable forces at P1/P2 and the moving coaxial idler routing, plus their
  own bearing friction and hard stops. Rig inertia is taken equal to the virtual finger's (matched design; inertia
  mismatch is not rendered). The rig plane is horizontal: physical gravity does not act on q.
* Tendons: two distributed cables (phase4_cable) on the hardware routes (phase4_geometry.hardware_routes).
* Actuator motors 1, 2 (grounded, position mode): the unchanged P0 QDD motor, controller and telemetry.
* Contact rendering: transparent penalty contact of the pad (normal only), see VirtualEnvironment.
* Joint damping: the rig's physical bearing damping, matched to the virtual finger's b_joint; it is not rendered
  (rendering a damping term through the delayed force channel destabilises the low-inertia PIP pulley).
* Emulation motors 1, 2 (grounded, force mode): finite-mass carriages with a current-loop lag and load-cell feedback,
  commanded with the tension that renders the virtual environment, T_e = -A^-T tau_env(q, qdot) (A: the rig's
  nominal winding matrix from its routes; A21 = 0 for the default MCP axial passage), where tau_env holds
  the virtual finger's gravity, return springs/damping and object contact, computed by MuJoCo from the measured rig
  state (kinematically driven virtual finger; the object is integrated there).
The joint labels are the physical rig angles. The observer still sees only the eight actuator-motor channels.
"""
from collections import deque
from dataclasses import dataclass, field, asdict
import copy

import mujoco
import numpy as np

from .finger import FingerParams
from .phase4_cable import CableMaterial, ContactParams, CableChainParams, DistributedCablePlant
from .phase4_geometry import hardware_routes
from .sensing import MotorSensor
from .two_tendon import ARM, HOME, LimitedMotor, TwoTendonFinger


@dataclass
class ForceEmulatorParams:
    mass: float = .05  # kg, carriage + rotor reflected
    viscous: float = 2.  # N s/m, passive
    active_damping: float = 40.  # N s/m, controller velocity damping (stabilises the carriage on the cable stiffness)
    force_max: float = 80.  # N
    tau_force: float = 5e-4  # s, current-loop lag
    load_cell_gain: float = 1.  # force feedback gain on the measured cable tension
    stroke: float = .05  # m, hard stop (|x|)


class ForceEmulator:
    def __init__(self, params=None):
        self.p = params or ForceEmulatorParams()
        self.reset(10.)

    def reset(self, T_hold):
        self.x = self.v = 0.
        self.force = T_hold
        self.command = T_hold
        self.saturated = False
        self.stop_hit = False

    def step(self, dt, command, T_load, T_measured):
        p = self.p
        self.command = command
        desired = command + p.load_cell_gain*(command - T_measured) - p.active_damping*self.v
        self.saturated = abs(desired) > p.force_max
        desired = float(np.clip(desired, -p.force_max, p.force_max))
        self.force += dt/p.tau_force*(desired - self.force) if p.tau_force > 0 else desired - self.force
        v = (p.mass*self.v + dt*(self.force - T_load))/(p.mass + dt*p.viscous)
        x = self.x + dt*v
        self.stop_hit = abs(x) > p.stroke
        if self.stop_hit:
            x = float(np.clip(x, -p.stroke, p.stroke)); v = 0.
        self.x, self.v = x, v


@dataclass
class HardwareFidelity:
    level: str = "H2"
    friction: bool = True  # idler bearings (presliding, load dependent) and joint bearings
    viscoelastic: bool = True
    history: str = "fixed"  # or "evolving"
    sensing: str = "clean"  # or "imperfect" (telemetry only)
    hardware: str = "nominal"  # or "imperfect" (inner loop: K_t error, motor friction, slower controller, slower emulator)
    material: dict = field(default_factory=dict)
    contacts: dict = field(default_factory=dict)  # overrides by kind ("rolling") or by contact name
    initial_history: list = field(default_factory=lambda: [{}, {}])
    retension_at_reset: bool = True
    routes: dict = field(default_factory=dict)  # hardware_routes keyword overrides
    joint_bearing: dict = field(default_factory=lambda: dict(coulomb=2e-4, viscous=1e-5, v_eps=.01))
    emulator: dict = field(default_factory=dict)
    render_delay_steps: int = 0  # outer steps between the virtual computation and the emulator command
    render_contact: dict = field(default_factory=dict)  # VirtualEnvironment penalty-contact overrides

    def to_dict(self):
        return asdict(self)


def hardware_preset(level):
    """H1-H3 on the intended hardware topology (P1-P3 of the instruction for this rig)."""
    if level == "H1":  # geometry-resolved hardware routing, physical joints, frictionless contacts and bearings
        return HardwareFidelity("H1", friction=False)
    if level == "H2":  # + stateful contact/bearing friction (distributed transmission)
        return HardwareFidelity("H2")
    if level == "H3":  # + slow history and sensing / inner-loop imperfections
        return HardwareFidelity("H3", history="evolving", sensing="imperfect", hardware="imperfect")
    raise ValueError(level)


EVOLVING = dict(creep_strain_rate=1e-4, creep_threshold_N=12., creep_reference_N=18., creep_exponent=2.,
                settle_rate_m_per_Ns=2e-6, settle_cap_m=4e-4, conditioning_per_m=1.5)
SENSING = dict(current_gain=(1.03, .97), current_bias_A=(.01, -.008), bias_drift_A_per_sqrt_s=2e-3,
               encoder_drift_rad_per_s=(1e-4, -1e-4))
HARDWARE = dict(K_t=(.097, .103), tau_c_m=.002, controller_period=5e-4, emulator_tau_force=1e-3, render_delay_steps=2)
RENDER_SOLREF = .02  # s, contact time constant the force channel can render (stiffer contacts are not passive here)


def nominal_A(routes):
    """Equivalent winding matrix of the rig at home: joint-pulley radii plus the routing arm of the actuator-side strand."""
    A = np.zeros((2, 2))
    for i, route in enumerate(routes):
        G = route.segment_jacobian(HOME[0])
        joints = [(k, n) for k, n in enumerate(route.contacts) if n.kind == "joint"]
        for k, n in joints:
            A[i, n.dof] += n.radius
        A[i, 0] += -G[:joints[0][0] + 1].sum()
    return A


def chain_params(route, fid, material, pretension, history):
    contacts = []
    geo = route.evaluate(HOME[0])
    for node, wrap in zip(route.contacts, geo["wraps"]):
        base = dict(name=node.name, kind=node.kind, radius=node.radius, wrap=float(wrap), dof=node.dof)
        if node.kind == "rolling":
            base.update(inertia=2e-7, bearing_coulomb=1e-4, bearing_mu=.002, bearing_radius=.002, bearing_viscous=1e-7,
                        presliding_stiffness=5000., mu=0.)
        elif node.kind == "sliding":
            base.update(mu=.05, presliding_stiffness=5000.)
        base.update(fid.contacts.get(node.kind, {})); base.update(fid.contacts.get(node.name, {}))
        contacts.append(ContactParams(**base))
    return CableChainParams(material=material, contacts=contacts, segment_lengths=tuple(float(x) for x in geo["segment_lengths"]),
                            pretension=pretension, friction="presliding" if fid.friction else "frictionless",
                            history=copy.deepcopy(history), retension_at_reset=fid.retension_at_reset)


class Mechanism:
    """Physical rig: MuJoCo rigid-body inertia of the finger links (no gravity, no springs, no object)."""
    def __init__(self, p, bearing, friction=True):
        f = TwoTendonFinger(p)
        f.model.opt.gravity[:] = 0
        for j in f.jq:
            f.model.jnt_stiffness[j] = 0.
        f.model.dof_damping[:] = 0.
        self.damping = np.array([p.b_joint, p.b_joint])  # matched physical joint damping (not rendered)
        f.model.geom_contype[f.obj_geom] = 0; f.model.geom_conaffinity[f.obj_geom] = 0
        self.f, self.dof = f, f.dof
        self.range = np.array([p.q1_range, p.q2_range])
        self.bearing = bearing if friction else dict(coulomb=0., viscous=0., v_eps=.01)
        self.q, self.qd = HOME.copy(), np.zeros(2)
        self.KE_work = 0.
        self.E_bearing = 0.
        self.stop_hits = 0
        self.update_inertia()

    def update_inertia(self):
        d = self.f.data
        d.qpos[self.f.qadr] = self.q; d.qvel[:] = 0; d.qvel[self.dof] = self.qd
        mujoco.mj_forward(self.f.model, d)
        full = np.zeros((self.f.model.nv, self.f.model.nv)); mujoco.mj_fullM(self.f.model, d, full)
        self.M = full[np.ix_(self.dof, self.dof)]
        self.bias = d.qfrc_bias[self.dof].copy()  # Coriolis/centrifugal only (gravity off)

    def step(self, dt, torque):
        b = self.bearing
        friction = -b["coulomb"]*np.tanh(self.qd/b["v_eps"]) - b["viscous"]*self.qd
        stop = np.zeros(2)
        for i in range(2):
            lo, hi = self.range[i]
            if self.q[i] < lo:
                stop[i] = 20.*(lo - self.q[i]) - .05*self.qd[i]
            elif self.q[i] > hi:
                stop[i] = 20.*(hi - self.q[i]) - .05*self.qd[i]
        if stop.any():
            self.stop_hits += 1
        total = torque + friction + stop - self.bias - self.damping*self.qd
        self.qd = self.qd + dt*np.linalg.solve(self.M, total)
        self.q = self.q + dt*self.qd
        self.KE_work += float(torque @ self.qd)*dt
        self.E_bearing += float(-friction @ self.qd + self.damping @ self.qd**2)*dt

    @property
    def kinetic(self):
        return .5*float(self.qd @ self.M @ self.qd)


class VirtualEnvironment:
    """Virtual load computed from the measured rig state (rendered by the emulators).

    Gravity and the preloaded return springs come from the MuJoCo virtual finger. Contact is a transparent penalty
    model of the pad: a horizontal top face at rest height obj_z, supported by the pad spring k_obj (quasi-static,
    massless), touched by sample points on the underside of both link capsules; the contact stiffness is in series
    with the pad spring and the normal is +z (pad pushes the finger up). MuJoCo's soft box-capsule contact is not
    used here: at the softness the force channel can render, penetration exceeds the capsule radius and its contact
    normal flips. The pad's Coulomb friction is not rendered (normal only).
    """
    def __init__(self, p, pretension, A, contact_stiffness=5000., contact_damping=2., samples=40, edge_band=.002):
        self.p = p
        self.f = f = TwoTendonFinger(p)
        f.reset(HOME, pretension)
        # Preload the virtual return springs against the pretension through the rig's own winding matrix.
        torque = A.T @ np.broadcast_to(np.asarray(pretension, float), 2)
        for i, j in enumerate(f.jq):
            f.model.qpos_spring[f.qadr[i]] = HOME[i] + (f.data.qfrc_bias[f.dof[i]] - torque[i])/f.model.jnt_stiffness[j]
        f.spring_ref = f.model.qpos_spring[f.qadr].copy()
        self.grav = mujoco.MjData(f.model)
        self.k = 1/(1/contact_stiffness + 1/p.k_obj)  # contact in series with the pad spring
        self.c = contact_damping
        self.s1 = np.linspace(0., p.L1, samples)
        self.s2 = np.linspace(0., p.L2, samples)
        self.edge = edge_band  # weights taper linearly to zero over this band at the pad edges (no force jumps)
        self.tau = np.zeros(2)
        self.contact = dict(contact_state=0., contact_force=0., contact_location=np.zeros(3))
        self.object = (0., 0.)

    def _points(self, q, qd):
        """Lowest capsule-surface points (x, z), their velocities and Jacobians d(x,z)/dq."""
        p = self.p
        out = []
        c1, s1, c12, s12 = np.cos(q[0]), np.sin(q[0]), np.cos(q.sum()), np.sin(q.sum())
        for s in self.s1:
            pos = np.array([s*c1, -s*s1 - p.r_link])
            J = np.array([[-s*s1, 0.], [-s*c1, 0.]])
            out.append((pos, J))
        pip = np.array([p.L1*c1, -p.L1*s1])
        for s in self.s2:
            pos = pip + np.array([s*c12, -s*s12 - p.r_link])
            J = np.array([[-p.L1*s1 - s*s12, -s*s12], [-p.L1*c1 - s*c12, -s*c12]])
            out.append((pos, J))
        return out

    def update(self, q, qd):
        f, p = self.f, self.p
        g = self.grav; g.qpos[:] = 0; g.qpos[f.qadr] = q; g.qvel[:] = 0
        mujoco.mj_forward(f.model, g)
        gravity = -g.qfrc_bias[f.dof]
        spring = -f.model.jnt_stiffness[f.jq]*(q - f.model.qpos_spring[f.qadr])
        lo, hi = p.obj_x - p.obj_half[0], p.obj_x + p.obj_half[0]
        tau_c = np.zeros(2); total = 0.; moment = 0.
        for pos, J in self._points(q, qd):
            depth = p.obj_z - pos[1]
            weight = min(1., max(0., (pos[0] - lo)/self.edge))*min(1., max(0., (hi - pos[0])/self.edge))
            if weight > 0 and depth > 0:
                rate = -(J @ qd)[1]
                F = weight*max(0., self.k*depth + self.c*rate)/len(self.s1)  # per-sample share of a line contact
                tau_c += J.T @ np.array([0., F]); total += F; moment += F*pos[0]
        self.tau = spring + gravity + tau_c
        x = moment/total if total > 0 else 0.
        self.contact = dict(contact_state=float(total > 0), contact_force=float(total), contact_location=np.array([x, 0., p.obj_z]))
        self.object = (-total/p.k_obj, 0.)
        return self.tau


class HardwareHIL:
    def __init__(self, configs, finger_params=None, fidelity=None):
        self.fid = fid = copy.deepcopy(fidelity or hardware_preset("H2"))
        self.configs = copy.deepcopy(configs)
        p = copy.deepcopy(finger_params or FingerParams(q1_range=(-np.pi/6, np.pi/3), q2_range=(0., 5*np.pi/6), T_max=80.))
        p.timestep = self.configs[0].finger.timestep
        self.dt = p.timestep
        self.plant_dt = self.configs[0].timing.plant_dt
        self.n_sub = int(round(self.dt/self.plant_dt))
        self.log_every = int(round(self.configs[0].timing.log_dt/self.dt))
        if fid.hardware == "imperfect":
            for i, c in enumerate(self.configs):
                c.motor.K_t = HARDWARE["K_t"][i]; c.motor.tau_c_m = HARDWARE["tau_c_m"]
            fid.emulator.setdefault("tau_force", HARDWARE["emulator_tau_force"])
            fid.render_delay_steps = max(fid.render_delay_steps, HARDWARE["render_delay_steps"])
        if fid.sensing == "imperfect":
            for i, c in enumerate(self.configs):
                c.motor.current_gain = SENSING["current_gain"][i]; c.motor.current_bias = SENSING["current_bias_A"][i]
                c.motor_sensor.bias_drift_A_per_sqrt_s = SENSING["bias_drift_A_per_sqrt_s"]
        self.encoder_drift = np.array(SENSING["encoder_drift_rad_per_s"]) if fid.sensing == "imperfect" else np.zeros(2)
        pret = [c.cable.pretension for c in self.configs]
        self.routes = hardware_routes(**fid.routes)
        self.A = nominal_A(self.routes)
        self.mech = Mechanism(p, fid.joint_bearing, fid.friction)
        self.env = VirtualEnvironment(p, pret, self.A, **fid.render_contact)
        material = dict(asdict(CableMaterial()), **fid.material)
        if not fid.viscoelastic:
            material["maxwell_ratio"] = 0.
        if fid.history == "evolving":
            material.update({k: v for k, v in EVOLVING.items() if k not in fid.material})
        self.ch = []
        for i, (c, route) in enumerate(zip(self.configs, self.routes)):
            plant = DistributedCablePlant(chain_params(route, fid, CableMaterial(**material), c.cable.pretension, fid.initial_history[i]))
            motor = LimitedMotor(c.motor, rng=np.random.default_rng(c.seed + 1)); motor.reset(c.cable.pretension)
            em = ForceEmulator(ForceEmulatorParams(**fid.emulator)); em.reset(c.cable.pretension)
            joints = [(k, n.dof, n.radius) for k, n in enumerate(route.contacts) if n.kind == "joint"]
            self.ch.append(dict(cfg=c, route=route, plant=plant, motor=motor, emulator=em, joints=joints,
                                sensor=MotorSensor(c.motor_sensor, seed=c.seed + 5), rng=np.random.default_rng(c.seed),
                                record_rng=np.random.default_rng(c.seed + 2)))
        self.T_cmd = np.array(pret, float)
        # a deque of length n+1 returns the command issued n outer steps earlier
        self._render = deque([self.T_cmd.copy()]*(fid.render_delay_steps + 1), maxlen=fid.render_delay_steps + 1) if fid.render_delay_steps else None
        self.t = 0.; self.k = 0
        self.render_clipped = 0
        self.W_render = np.zeros(2)  # emulator work on the cables
        self.cable_torque = np.zeros(2)

    @property
    def log_due(self):
        return self.k % self.log_every == 0

    @property
    def q(self):
        return self.mech.q

    def _render_command(self):
        tau = self.env.update(self.mech.q, self.mech.qd)
        raw = -np.linalg.solve(self.A.T, tau)
        cmd = np.clip(raw, .5, 60.)
        self.render_clipped += int(np.any(cmd != raw))
        if self._render is not None:
            self._render.append(cmd)
            cmd = self._render[0]
        return cmd

    def step(self, displacement, velocity=(0., 0.)):
        self.T_cmd = self._render_command()
        self.mech.update_inertia()
        q1_0 = self.mech.q[0]
        geo = [(ch["route"].segment_lengths(q1_0), ch["route"].segment_jacobian(q1_0), ch["route"].evaluate(q1_0)["wraps"]) for ch in self.ch]
        dt = self.plant_dt
        for _ in range(self.n_sub):
            torque = np.zeros(2)
            for i, ch in enumerate(self.ch):
                motor, em, plant = ch["motor"], ch["emulator"], ch["plant"]
                r = motor.p.r_drum
                u = motor.command_from_reference(float(displacement[i])/r, float(velocity[i])/r)
                motor.step(dt, u, plant.T[0])
                x_old = em.x
                em.step(dt, float(self.T_cmd[i]), plant.T[-1], plant.T[-1])
                L0, G, wraps = geo[i]
                plant.set_geometry(L0 + G*(self.mech.q[0] - q1_0), G*self.mech.qd[0], wraps)
                rates = np.zeros(plant.n)
                for k, dof, radius in ch["joints"]:
                    rates[k] = radius*self.mech.qd[dof]
                plant.set_joint_rates(rates)
                plant.step(dt, motor.x_in, motor.xd_in, em.x, em.v)
                self.W_render[i] += plant.T[-1]*(em.x - x_old)
                forces = plant.joint_forces()
                for k, dof, radius in ch["joints"]:
                    torque[dof] += radius*forces[k]
                torque[0] -= float(plant.T @ G)  # moving coaxial-idler routing (virtual work of the geometry)
            self.cable_torque = torque
            self.mech.step(dt, torque)
        self.t += self.dt; self.k += 1

    def record(self):
        out = {}
        for i, ch in enumerate(self.ch, 1):
            c, motor, plant, em = ch["cfg"], ch["motor"], ch["plant"], ch["emulator"]
            obs = ch["sensor"].sample(self.t, motor.observation(c.timing.log_dt))
            obs["estimated_motor_torque"] = motor.p.K_t_nominal*obs["current"]
            if self.encoder_drift[i-1]:
                obs["position"] += self.encoder_drift[i-1]*self.t
            out[f"motor{i}"] = obs
            noise = c.sensors.tension_noise
            meas = lambda T: float(T + (ch["rng"].normal(0., noise) if noise > 0 else 0.))
            out[f"emulator{i}"] = dict(commanded_force=float(self.T_cmd[i-1]), actuator_force=em.force, position=em.x, velocity=em.v,
                                       force_tracking_error=float(self.T_cmd[i-1] - plant.T[-1]), force_saturated=float(em.saturated),
                                       stroke_limit_hit=float(em.stop_hit), motor_limit_hit=float(motor.limit_hit), rendering_work=self.W_render[i-1])
            out[f"transmission{i}"] = dict(input_tension=meas(plant.T[0]), output_tension=meas(plant.T[-1]),
                                           input_displacement=motor.x_in, output_displacement=em.x)
            gt = plant.ground_truth()
            out[f"ground_truth{i}"] = dict(motor_current_true=motor.s.i, electromagnetic_torque_true=motor.effective_K_t*motor.s.i,
                                           motor_sensor_bias=ch["sensor"].bias, effective_motor_torque_constant=motor.effective_K_t,
                                           input_tension_true=float(plant.T[0]), output_tension_true=float(plant.T[-1]), **gt)
        con, (pos, vel) = self.env.contact, self.env.object
        out["simulation"] = dict(joint_positions=self.mech.q.copy(), joint_velocities=self.mech.qd.copy(), contact_state=con["contact_state"],
                                 contact_force=con["contact_force"], contact_location=con["contact_location"], object_position=pos,
                                 object_velocity=vel, rendered_environment_torque=self.env.tau.copy(), emulator_commands=self.T_cmd.copy(),
                                 cable_joint_torque=self.cable_torque.copy(), nominal_winding_matrix=self.A.copy())
        out["rig"] = dict(kinetic_energy=self.mech.kinetic, cable_work_on_mechanism=self.mech.KE_work, bearing_dissipation=self.mech.E_bearing,
                          stop_hits=float(self.mech.stop_hits), render_clipped=float(self.render_clipped))
        return out
