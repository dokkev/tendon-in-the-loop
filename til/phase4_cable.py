"""Phase 4 distributed cable: N+1 unilateral viscoelastic segments separated by N stateful contacts.

Coordinates (one channel, actuator at segment 0, emulator at segment N):

* ``x_in``  actuator reel-in (positive stretches segment 0); ``x_out`` emulator pull (positive stretches segment N).
* ``u_k``   material transported over contact k toward the actuator (k = 1..N). For a rolling idler,
  ``u_k = r_k theta_k`` (no slip); for a sliding guide it is the slip of the cable over the guide.
* segment material ``m_i`` (unstretched length): m_0 = M_0 - x_in + u_1, m_i = M_i - u_i + u_{i+1},
  m_N = M_N - u_N - x_out. Extension ``e_i = L_i(q_r) - m_i - c_i`` with geometric length ``L_i`` from the rig
  route and slow material growth ``c_i`` (creep, and settling on the emulator termination).

Contact k obeys  c_k dw_k/dt = (T_{k-1} - T_k) - f_k - b_k w_k  with ``w_k = du_k/dt``, ``c_k = J_k/r_k^2``
(rolling, zero for sliding) plus the cable mass lumped at the contact (half of each adjacent segment), and an elastoplastic (presliding) friction force ``f_k`` bounded by
a contact-specific sliding level:

* sliding guide: (T_a + T_b) tanh(mu beta/2), the symmetric form of T_high/T_low <= exp(mu beta). Radius is not
  in the exponent.
* rolling idler: mu_b r_b N_k / r_k with the bearing load N_k = |T_a t_a + T_b t_b|.
* both: plus a load-independent bearing torque tau_c / r_k.
* optional bending term ``bending_coefficient * beta / r_bend`` (radius-dependent, labelled separately).

The N contact rates are solved implicitly per step (Newton on the tridiagonal system, Gauss-Seidel bisection
fallback). The segment law is the Phase 2 law per segment (equilibrium spring with cubic stiffening, Maxwell
branch, viscous term scaled by 1/L, unilateral).
"""
from dataclasses import dataclass, field, asdict
from math import pi, tanh

import numpy as np


@dataclass
class ContactParams:
    name: str = "contact"
    kind: str = "sliding"  # "sliding" (fixed guide), "rolling" (free idler) or "joint" (pulley fixed to a mechanism joint)
    dof: int = -1  # joint contacts: mechanism degree of freedom whose rotation (relative to the spans) moves the cable
    radius: float = .01  # m
    wrap: float = pi/2  # rad, nominal; replaced by the rig geometry at every step when a route is attached
    inertia: float = 0.  # kg m^2 (rolling only)
    mu: float = .05  # cable-guide friction coefficient (sliding level)
    level_model: str = ""  # "" -> "capstan" for sliding, "bearing" for rolling
    bearing_coulomb: float = 0.  # N m
    bearing_mu: float = 0.  # bearing friction coefficient
    bearing_radius: float = .002  # m
    bearing_viscous: float = 0.  # N m s/rad
    presliding_stiffness: float = 5000.  # N/m on the transport coordinate
    bending_coefficient: float = 0.  # N m per rad (force = coefficient * beta / r_bend)
    bending_radius: float = 0.  # m, 0 -> radius
    static_ratio: float = 1.
    stribeck_velocity: float = 4e-4  # m/s

    @property
    def model(self):
        return self.level_model or ("capstan" if self.kind == "sliding" else "bearing")

    @property
    def mass(self):
        return self.inertia/self.radius**2 if self.kind == "rolling" else 0.  # joint pulley inertia belongs to the mechanism


@dataclass
class CableMaterial:
    """Effective cable law (category C, hypothetical). Same symbols as the Phase 2 Specimen."""
    id: str = "hypothetical_distributed_A"
    diameter_m: float = .0005
    modulus_pa: float = 1e10
    nonlinear_strain_coefficient: float = 400.
    maxwell_ratio: float = .35
    relaxation_time_s: float = .5
    viscous_Ns: float = .22360679774997896  # N s; segment dashpot = viscous_Ns / L_i (1 N s/m at the P0 length)
    taut_regularization_m: float = 1e-6  # viscous force fades in over this extension (continuity at slack)
    linear_density_kg_per_m: float = 2.7e-4  # cable mass, lumped at the contacts (half of each adjacent segment)
    # Slow history (all zero = off)
    creep_strain_rate: float = 0.  # 1/s at the reference overstress
    creep_threshold_N: float = 12.
    creep_reference_N: float = 18.
    creep_exponent: float = 2.
    settle_rate_m_per_Ns: float = 0.
    settle_cap_m: float = .0004
    conditioning_per_m: float = 0.  # per metre of sliding transport over guides
    conditioning_rolling_weight: float = .25  # rolling transport counts at this weight
    stiffness_loss_fraction: float = .15
    friction_growth_fraction: float = .5

    @property
    def EA(self):
        return self.modulus_pa*pi*self.diameter_m**2/4


@dataclass
class CableChainParams:
    material: CableMaterial = field(default_factory=CableMaterial)
    contacts: list = field(default_factory=list)
    segment_lengths: tuple = (.2236, .2236)  # m, geometric lengths at the reference rig angle
    pretension: float = 10.
    friction: str = "presliding"  # or "frictionless"
    history: dict = field(default_factory=dict)  # persistent specimen history: creep, settle, d, maxwell offsets
    retension_at_reset: bool = True  # session start re-establishes pretension (history stays recorded)

    def to_dict(self):
        return asdict(self)


class DistributedCablePlant:
    """Drop-in plant for TiLSimulator (attributes s.T1, s.T2, s.W_out, step(), reset(), ground_truth())."""

    def __init__(self, params=None):
        self.p = params or CableChainParams(contacts=[ContactParams()])
        n = len(self.p.contacts)
        if len(self.p.segment_lengths) != n + 1:
            raise ValueError("N contacts need N+1 segments")
        if min(self.p.segment_lengths) <= 0:
            raise ValueError("segment lengths must be positive")
        self.n = n
        self.L_ref = np.array(self.p.segment_lengths, float)
        m = self.p.material
        self.k = m.EA/self.L_ref
        self.k3 = m.nonlinear_strain_coefficient*self.k/self.L_ref**2
        self.eta = m.viscous_Ns/self.L_ref
        cable_mass = m.linear_density_kg_per_m*(self.L_ref[:-1] + self.L_ref[1:])/2
        self.joint = np.array([c.kind == "joint" for c in self.p.contacts], bool)
        self.free = ~self.joint
        self.c = np.where(self.joint, 0., np.array([c.mass for c in self.p.contacts]) + cable_mass)
        self.w_prescribed = np.zeros(n)
        self.ks = np.array([c.presliding_stiffness for c in self.p.contacts])
        self.radius = np.array([c.radius for c in self.p.contacts])
        self.b = np.array([c.bearing_viscous/c.radius**2 if c.kind == "rolling" else 0. for c in self.p.contacts])
        self.rolling = np.array([c.kind == "rolling" for c in self.p.contacts])
        cs = self.p.contacts
        self._capstan = np.array([c.model == "capstan" for c in cs])
        self._mu = np.array([c.mu for c in cs], float)
        self._bearing = np.array([c.bearing_mu*c.bearing_radius/c.radius for c in cs], float)
        self._constant = np.array([c.bearing_coulomb/c.radius for c in cs], float)
        self._bending = np.array([c.bending_coefficient/(c.bending_radius or c.radius) for c in cs], float)
        self._static = np.array([c.static_ratio for c in cs], float)
        self._stribeck = np.array([c.stribeck_velocity for c in cs], float)
        self.L = self.L_ref.copy()
        self.Ld = np.zeros(n + 1)
        self.wraps = np.array([c.wrap for c in self.p.contacts], float)
        self.solver_stats = dict(newton=0, fallback=0, max_iterations=0)
        self.reset()

    # ---------------------------------------------------------------------------------------------
    def set_joint_rates(self, rates):
        """Transport rates of the joint-pulley contacts, prescribed by the mechanism (R * relative joint rate)."""
        self.w_prescribed = np.asarray(rates, float).copy()

    def joint_forces(self):
        """Net cable force T_{k-1} - T_k across every contact (actuator side minus emulator side)."""
        return self.T[:-1] - self.T[1:]

    def set_geometry(self, lengths, rates=None, wraps=None):
        """Rig geometry update (called once per outer step by the HIL before the substeps)."""
        self.L = np.asarray(lengths, float).copy()
        self.Ld = np.zeros(self.n + 1) if rates is None else np.asarray(rates, float).copy()
        if wraps is not None:
            self.wraps = np.asarray(wraps, float).copy()

    @property
    def factor(self):
        return 1 - self.p.material.stiffness_loss_fraction*self.d

    def _material_offsets(self):
        offsets = self.creep.copy()
        offsets[-1] += self.settle
        return offsets

    def reset(self):
        h = self.p.history
        n = self.n
        self.creep = np.array(h.get("creep", np.zeros(n + 1)), float)
        self.settle = float(h.get("settle", 0.))
        self.d = float(h.get("d", 0.))
        self.t = 0.
        self.u = np.zeros(n)
        self.w = np.zeros(n)
        self.z = np.zeros(n)  # presliding spring deflection, m (force = ks z)
        self.f = np.zeros(n)
        self.sliding_distance = np.zeros(n)
        T0 = self.p.pretension
        e0 = np.array([self._invert(T0, i) for i in range(n + 1)])
        if self.p.retension_at_reset:
            # Session start: tensioned to the known pretension against the current material history.
            self.M = self.L - e0 - self._material_offsets()
        else:
            # No re-tensioning: the specimen keeps the material it ended the previous session with.
            self.M = np.array(h.get("material", self.L_ref - e0), float)
        # Maxwell branch: relaxed unless a history is supplied (an absolute state, or a fraction of the relaxed
        # extension; a fraction < 1 means the branch still carries load from recent stretching).
        self.v = np.array(h["maxwell_state"], float) if "maxwell_state" in h else e0*float(h.get("maxwell_fraction", 1.))
        self.x_in = self.x_out = 0.
        e, ed = self._extensions(self.u, np.zeros(n))
        self.T, self.elastic, _, _ = self._segments(e, ed, self.v, 1e-9, update=False)
        self.e = e
        self.W_in = self.W_out = self.W_support = self.W_parameter = self.W_joint = 0.
        self.E_diss = self.E_aging = 0.
        self.E_friction = self.E_viscous = self.E_maxwell = self.E_numerical = 0.
        self.stored = self._energy(e, self.v)
        self.initial_energy = self.stored
        self.s = _StateView(self)

    def _invert(self, T, i):
        lo, hi = 0., T/(self.k[i]*self.factor) + 1e-12
        for _ in range(80):
            e = (lo + hi)/2
            if self.factor*(self.k[i]*e + self.k3[i]*e**3) < T:
                lo = e
            else:
                hi = e
        return (lo + hi)/2

    # ---------------------------------------------------------------------------------------------
    def _extensions(self, u, w, x_in=None, x_out=None, xd_in=0., xd_out=0.):
        x_in = self.x_in if x_in is None else x_in
        x_out = self.x_out if x_out is None else x_out
        m = self.M.copy()
        m[0] += -x_in
        m[-1] += -x_out
        m[:-1] += u
        m[1:] -= u
        md = np.zeros(self.n + 1)
        md[0] = -xd_in
        md[-1] = -xd_out
        md[:-1] += w
        md[1:] -= w
        return self.L - m - self._material_offsets(), self.Ld - md

    def _segments(self, e, ed, v_old, dt, update=True):
        mat = self.p.material
        ep = np.maximum(e, 0.)
        a = dt/mat.relaxation_time_s
        v = (v_old + a*ep)/(1 + a) if update else v_old
        km = mat.maxwell_ratio*self.k
        delayed = np.maximum(0., ep - v)
        elastic = self.factor*(self.k*ep + self.k3*ep**3) + km*delayed
        taut = e > 0
        activation = np.minimum(1., ep/mat.taut_regularization_m) if mat.taut_regularization_m > 0 else 1.
        T = np.where(taut, np.maximum(0., elastic + self.eta*activation*ed), 0.)
        tangent = np.where(taut, self.factor*(self.k + 3*self.k3*ep*ep) + np.where(delayed > 0, km/(1 + a), 0.), 0.)
        return T, np.where(taut, elastic, 0.), tangent, v

    def _levels(self, T):
        """Sliding level of every contact from the current segment tensions (force units, >= 0)."""
        growth = 1 + self.p.material.friction_growth_fraction*self.d
        Ta, Tb, beta = T[:-1], T[1:], self.wraps
        capstan = (Ta + Tb)*np.tanh(self._mu*growth*beta/2)
        load = np.sqrt(np.maximum(0., Ta*Ta + Tb*Tb - 2*Ta*Tb*np.cos(beta)))
        level = np.where(self._capstan, capstan, self._bearing*growth*load)
        return level + self._constant + self._bending*beta

    def _friction(self, w, levels, dt):
        if self.p.friction == "frictionless":
            zero = np.zeros(self.n)
            return zero, self.z*0, zero, zero
        level = levels*(1 + (self._static - 1)*np.exp(-(w/self._stribeck)**2))
        trial = self.ks*(self.z + dt*w)
        f = np.where(self.joint, 0., np.clip(trial, -level, level))  # no slip on a joint pulley
        z = np.where(self.joint, self.z, f/self.ks)
        tangent = np.where(self.joint, 0., np.where(np.abs(trial) < level, self.ks*dt, 0.))
        plastic = np.where(self.joint, 0., dt*w - (z - self.z))
        return f, z, tangent, np.maximum(0., f*plastic)

    def _level_gradient(self, T):
        """d(level_k)/dT_{k-1} and d(level_k)/dT_k (used in the Newton matrix when a contact slides)."""
        growth = 1 + self.p.material.friction_growth_fraction*self.d
        Ta, Tb, beta = T[:-1], T[1:], self.wraps
        cap = np.tanh(self._mu*growth*beta/2)
        load = np.sqrt(np.maximum(0., Ta*Ta + Tb*Tb - 2*Ta*Tb*np.cos(beta)))
        safe = np.where(load > 1e-12, load, 1.)
        ga = np.where(self._capstan, cap, np.where(load > 1e-12, self._bearing*growth*(Ta - Tb*np.cos(beta))/safe, 0.))
        gb = np.where(self._capstan, cap, np.where(load > 1e-12, self._bearing*growth*(Tb - Ta*np.cos(beta))/safe, 0.))
        return ga, gb

    def _residual(self, w, dt, x_in, xd_in, x_out, xd_out, jacobian=True):
        u = self.u + dt*w
        e, ed = self._extensions(u, w, x_in, x_out, xd_in, xd_out)
        T, elastic, tangent, v = self._segments(e, ed, self.v, dt)
        levels = self._levels(T)
        f, z, ftan, plastic = self._friction(w, levels, dt)
        R = self.c*(w - self.w)/dt - (T[:-1] - T[1:]) + f + self.b*w
        if not jacobian:
            return R, None, (u, e, ed, T, elastic, v, f, z, plastic)
        n = self.n
        g = dt*tangent + np.where(e > 0, self.eta, 0.)
        J = np.zeros((n, n))
        idx = np.arange(n)
        J[idx, idx] = self.c/dt + g[:-1] + g[1:] + ftan + self.b
        if n > 1:
            J[idx[:-1], idx[:-1] + 1] = -g[1:-1]
            J[idx[1:], idx[1:] - 1] = -g[1:-1]
        if self.p.friction != "frictionless":
            sliding = ftan == 0.
            if sliding.any():
                ga, gb = self._level_gradient(T)
                sign = np.sign(f)
                for k in np.flatnonzero(sliding):
                    # f_k = sign*level_k(T_{k-1}, T_k); T_{k-1} depends on w_{k-1} (+g) and w_k (-g), T_k on w_k (+g), w_{k+1} (-g)
                    if k > 0:
                        J[k, k-1] += sign[k]*ga[k]*g[k]
                    J[k, k] += sign[k]*(-ga[k]*g[k] + gb[k]*g[k+1])
                    if k < n - 1:
                        J[k, k+1] += -sign[k]*gb[k]*g[k+1]
        return R, J, (u, e, ed, T, elastic, v, f, z, plastic)

    def _solve(self, dt, args):
        w = self.w.copy()
        w[self.joint] = self.w_prescribed[self.joint]
        n = self.n
        F = self.free
        if n == 0 or not F.any():
            R, J, aux = self._residual(w, dt, *args)
            return w, R*0, aux
        R, J, aux = self._residual(w, dt, *args)
        R, J = R[F], J[np.ix_(F, F)]
        norm = np.max(np.abs(R))
        for it in range(40):
            if norm < 1e-9:
                self.solver_stats["newton"] += 1
                self.solver_stats["max_iterations"] = max(self.solver_stats["max_iterations"], it)
                return w, R, aux
            step = np.zeros(n)
            step[F] = np.linalg.solve(J, R)
            alpha = 1.
            for _ in range(12):  # backtracking on the max-norm of the residual
                trial = w - alpha*step
                Rt, Jt, auxt = self._residual(trial, dt, *args)
                Rt, Jt = Rt[F], Jt[np.ix_(F, F)]
                nt = np.max(np.abs(Rt))
                if nt < norm or alpha < 1e-3:
                    break
                alpha /= 2
            w, R, J, aux, norm = trial, Rt, Jt, auxt, nt
        # Fallback: Gauss-Seidel sweeps; each contact residual is monotone in its own rate.
        self.solver_stats["fallback"] += 1
        for sweep in range(400):
            for k in np.flatnonzero(F):
                def rk(x):
                    trial = w.copy(); trial[k] = x
                    return self._residual(trial, dt, *args, jacobian=False)[0][k]
                lo, hi = w[k] - 1., w[k] + 1.
                for _ in range(60):
                    if rk(lo) <= 0 <= rk(hi):
                        break
                    lo, hi = lo - 2*(hi - lo), hi + 2*(hi - lo)
                for _ in range(60):
                    mid = (lo + hi)/2
                    if rk(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                w[k] = (lo + hi)/2
            R, _, aux = self._residual(w, dt, *args, jacobian=False)
            R = R[F]
            if np.max(np.abs(R)) < 1e-7:
                return w, R, aux
        raise FloatingPointError(f"distributed cable solve failed: residual={np.max(np.abs(R))}")

    # ---------------------------------------------------------------------------------------------
    def step(self, dt, x_in, xd_in, x_out, xd_out):
        mat = self.p.material
        old_T = self.T.copy()
        old_L = self.L.copy()
        self.L = self.L + self.Ld*dt  # rig geometry advances at the rate set for this outer step
        w, R, (u, e, ed, T, elastic, v, f, z, plastic) = self._solve(dt, (x_in, xd_in, x_out, xd_out))
        # Dissipation over the step (same discrete forms as the Phase 2 plant).
        viscous = float(np.sum(np.maximum(0., (T - elastic)*ed)))*dt
        maxwell = float(np.sum(mat.maxwell_ratio*self.k*np.maximum(0., np.maximum(e, 0) - v)**2))/mat.relaxation_time_s*dt
        bearing = float(np.sum(self.b*w*w))*dt
        self.E_numerical += float(np.sum(.5*self.c*(w - self.w)**2))  # implicit-Euler kinetic-energy loss
        friction = float(np.sum(plastic))
        self.E_viscous += viscous; self.E_maxwell += maxwell; self.E_friction += friction + bearing
        self.E_diss += viscous + maxwell + friction + bearing
        self.u, self.w, self.v, self.z, self.f = u, w, v, z, f
        self.sliding_distance += np.abs(w)*dt
        self.x_prev = (self.x_in, self.x_out)
        old_x = (self.x_in, self.x_out)
        self.x_in, self.x_out = x_in, x_out
        # Slow history: creep (Norton with threshold), termination settling, conditioning.
        aging = mat.creep_strain_rate > 0 or mat.settle_rate_m_per_Ns > 0 or mat.conditioning_per_m > 0
        if aging:
            before = self._energy(self._extensions(u, w)[0], v)
            if mat.creep_strain_rate > 0:
                over = np.maximum(0., T - mat.creep_threshold_N)/mat.creep_reference_N
                self.creep += dt*self.L_ref*mat.creep_strain_rate*over**mat.creep_exponent
            if mat.settle_rate_m_per_Ns > 0 and T[-1] > mat.creep_threshold_N and mat.settle_cap_m > 0:
                self.settle += dt*mat.settle_rate_m_per_Ns*(T[-1] - mat.creep_threshold_N)*max(0., 1 - self.settle/mat.settle_cap_m)
            after_rest = self._energy(self._extensions(u, w)[0], v)
            self.E_aging += max(0., before - after_rest)
            if mat.conditioning_per_m > 0:
                weights = np.where(self.rolling, mat.conditioning_rolling_weight, 1.)
                self.d = min(1., self.d + mat.conditioning_per_m*float(np.sum(weights*np.abs(w)))*dt)
            after_age = self._energy(self._extensions(u, w)[0], v)
            self.W_parameter += after_age - after_rest
            # Re-evaluate tensions with the slow states applied (internal states held).
            e, ed = self._extensions(u, w, x_in, x_out, xd_in, xd_out)
            T, elastic, _, _ = self._segments(e, ed, v, dt, update=False)
        self.T, self.elastic, self.e = T, elastic, e
        self.W_in += .5*(old_T[0] + self.T[0])*(x_in - old_x[0])
        self.W_out += .5*(old_T[-1] + self.T[-1])*(x_out - old_x[1])
        self.W_support += float(np.sum(.5*(old_T + self.T)*(self.L - old_L)))
        if self.joint.any():  # power the mechanism delivers to the cable at the joint pulleys
            Tm = .5*(old_T + self.T)
            self.W_joint += float(np.sum(((Tm[1:] - Tm[:-1])*w*dt)[self.joint]))
        self.stored = self._energy(e, v)
        self.t += dt
        self.residual_force = float(np.max(np.abs(R))) if self.n else 0.
        return self.T[0], self.T[-1]

    def _energy(self, e, v):
        mat = self.p.material
        ep = np.maximum(e, 0.)
        spring = self.factor*(.5*self.k*ep**2 + .25*self.k3*ep**4)
        maxwell = .5*mat.maxwell_ratio*self.k*np.maximum(0., ep - v)**2
        contacts = .5*self.c*self.w**2 + .5*self.ks*self.z**2
        return float(np.sum(spring) + np.sum(maxwell) + np.sum(contacts))

    def energy_residual(self):
        """r_E = W_in + W_out + W_support + W_param - dE_stored - E_diss - E_aging - E_numerical.

        E_numerical is the kinetic energy removed by the implicit-Euler contact inertia update,
        computed from the integrator states (reported separately, not fitted)."""
        return (self.W_in + self.W_out + self.W_support + self.W_joint + self.W_parameter
                - (self.stored - self.initial_energy) - self.E_diss - self.E_aging - self.E_numerical)

    def export_history(self):
        """Persistent specimen state to carry into the next session of the same specimen."""
        material = self.M.copy()
        material[0] -= self.x_in; material[-1] -= self.x_out
        material[:-1] += self.u; material[1:] -= self.u
        return dict(creep=self.creep.tolist(), settle=self.settle, d=self.d, material=material.tolist())

    def ground_truth(self):
        T = self.T
        return dict(cable_segment_extensions=self.e.copy(), cable_segment_strains=self.e/self.L_ref,
                    segment_tensions=T.copy(), viscoelastic_state=self.v.copy(), friction_state=self.z.copy(),
                    contact_friction_force=self.f.copy(), contact_transport=self.u.copy(), contact_rates=self.w.copy(),
                    contact_wraps=self.wraps.copy(), segment_geometric_lengths=self.L.copy(),
                    friction_torque=float(np.sum(self.f*self.radius)),
                    effective_stiffness=float(self.factor*self.p.material.EA/np.sum(self.L_ref)),
                    effective_friction=float(np.mean([c.mu for c in self.p.contacts]) if self.n else 0.)*(1 + self.p.material.friction_growth_fraction*self.d),
                    slack_state=(T <= 0).astype(float), settle_state=self.settle, creep_state=self.creep.copy(),
                    degradation_state=self.d, dissipated_energy=self.E_diss, frictional_dissipation=self.E_friction,
                    viscous_dissipation=self.E_viscous, maxwell_dissipation=self.E_maxwell, stored_energy=self.stored,
                    parameter_work=self.W_parameter, aging_dissipation=self.E_aging, support_work=self.W_support, joint_work=self.W_joint,
                    numerical_dissipation=self.E_numerical,
                    energy_residual=self.energy_residual(), sliding_distance=self.sliding_distance.copy(),
                    effective_reference_length=float(np.sum(self.M + self._material_offsets())),
                    intervention_state=0.)


class _StateView:
    """Phase-2-compatible attribute view (TiLSimulator reads s.T1, s.T2, s.phi, s.phi_d, s.tau_f, s.W_out)."""
    def __init__(self, plant):
        object.__setattr__(self, "_p", plant)

    def __getattr__(self, name):
        p = self._p
        first = int(np.argmax(p.rolling)) if p.n and p.rolling.any() else None
        return dict(T1=lambda: float(p.T[0]), T2=lambda: float(p.T[-1]), W_out=lambda: p.W_out, W_in=lambda: p.W_in,
                    phi=lambda: float(p.u[first]/p.radius[first]) if first is not None else 0.,
                    phi_d=lambda: float(p.w[first]/p.radius[first]) if first is not None else 0.,
                    tau_f=lambda: float(np.sum(p.f*p.radius)), t=lambda: p.t)[name]()


def chain_from_route(route, q_ref, material=None, pretension=10., contact_overrides=None, friction="presliding", **kw):
    """Contacts and segment lengths from a phase4_geometry.RigRoute at the reference rig angle."""
    geo = route.evaluate(q_ref)
    contacts = []
    for node, wrap in zip(route.contacts, geo["wraps"]):
        base = dict(name=node.name, kind=node.kind, radius=node.radius, wrap=float(wrap))
        if node.kind == "rolling":
            base.update(inertia=4e-7, bearing_coulomb=2e-4, bearing_mu=.002, bearing_radius=.003, bearing_viscous=2e-7,
                        presliding_stiffness=5000., mu=0.)
        else:
            base.update(mu=.05, presliding_stiffness=5000.)
        base.update((contact_overrides or {}).get(node.kind, {}))
        base.update((contact_overrides or {}).get(node.name, {}))
        contacts.append(ContactParams(**base))
    return CableChainParams(material=material or CableMaterial(), contacts=contacts,
                            segment_lengths=tuple(float(x) for x in geo["segment_lengths"]), pretension=pretension,
                            friction=friction, **kw)
