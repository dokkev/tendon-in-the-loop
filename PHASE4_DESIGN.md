# Phase 4 design — high-fidelity two-tendon reference plant (Milestone A)

Status: design only. Nothing in this document is a result. Source baseline: commit `74d79fde`
(`Verify separate two-tendon TiL architecture and motor-only baselines`); the existing suite passes
**97 tests** at that commit. Phase 1–3 and the two-tendon verification study (P0) are frozen: no source,
dataset, checkpoint, calibration, report or figure of theirs is modified. Phase 4 code lives in new modules
(`til/phase4_*.py`), behind its own configuration objects, with its own tests, scripts, reports and data paths.

## 1. Audit of the existing reference (P0)

| Component | Current implementation | Limitation relevant to Phase 4 |
|---|---|---|
| Virtual tendon geometry | `TwoTendonFinger`: two MuJoCo **fixed** tendons, `e = A(q − q0)`, `J_l = −A` constant | No configuration dependence; not a spatial route |
| Rig (physical) routing | `Routing` dataclass: free-span lengths, guide radii and wrap angles condensed into one equivalent pulley; wrap is a constant | Wrap does not change with the moving rig link; guides are not individual contacts |
| Cable | `TransmissionPlant`: two axial segments, cubic stiffening, one Maxwell branch per segment, unilateral, viscous | Two segments only; no tension distribution between guides |
| Friction | One presliding (elastoplastic) state on the equivalent pulley; level `R_p (T1+T2) tanh(μβ/2)` + bearing + bending term | Single lumped state; the capstan is applied as a symmetric static bound, not per contact |
| Slow history | creep, settling (output side), conditioning `d` by sliding distance, parameter-work ledger | Reasonable; reuse the separation |
| Motor / emulator / sensors | `QDDMotor` + `LimitedMotor`, finite-mass force-limited emulator, `MotorSensor` telemetry (quantisation, noise, filter, delay) | Inner position loop reads the **true** rotor state; telemetry corruption only reaches the observer |
| Coupling | `TwoTendonSimulator`: two inherited chains, shared finger integrated once per step | Keep unchanged as P0 |
| Website demo | `TendonInTheLoop2DoFDemo.tsx` | Teaching drawing. Its excursions `e1 = R1 q1`, `e2 = r(q1+q2)` are imposed, not derived; the crossed idler arcs vary by `+q1` and `−q1` and cancel. Not ground truth. |

## 2. Two geometries that must not be conflated

The HIL concept has **two** routing geometries. Phase 4 resolves both, separately.

**V — virtual finger geometry (MuJoCo).** Determines `ℓ_v(q)`, `J_v(q) = ∂ℓ_v/∂q`, the generalized force
`τ = −J_v(q)ᵀ T_out` and the emulator command. The virtual tendon carries one uniform tension (MuJoCo
spatial tendon, `stiffness = damping = frictionloss = 0`). Any friction "inside the finger" is represented only
to the extent the physical rig reproduces it.

**R — physical rig geometry.** The routing the real cable experiences between actuator and emulator: fixed
guides, idlers on the rig's joint axis, an idler on the moving rig link. Determines wrap angles `β_k(q1)`,
contact normal loads and free-span lengths, i.e. the **distributed friction and compliance**. The rig link is
driven to follow the virtual `q1` by a separate link actuator.

Emulator command with both geometries (avoids double counting the rig's own path change):

```
x_out,j = −[ℓ_v,j(q) − ℓ_v,j(q0)] − [ℓ_R,j(q1) − ℓ_R,j(q1,0)]
```

### Three length quantities, kept separate

| Quantity | Definition | Where it enters |
|---|---|---|
| Geometric path length | current centerline: free spans + tangent-to-tangent arcs | `ℓ_v`, `ℓ_R`, kinematics |
| Material transport | cable passing over a contact (idler rotation `r_k θ_k`, or slip `s_k` on a sliding guide) | friction power, sliding distance, conditioning |
| Elastic extension | segment stretch beyond its reference length | tension |

A free (rolling) idler's rotation never changes the geometric path; it only carries material. A pulley
**fixed to a link with the cable terminated on it** changes the path by `r Δq` through its arc — already part of
the geometric path, so it is not added again as transport. Tests check both cases.

## 3. Proposed geometry (documented example, not the intended apparatus)

The intended rig is not specified unambiguously (missing: pulley radii and positions, which idlers rotate
freely, cable attachments, guide order, rig-link length, exit guides). Phase 4 therefore uses **documented,
physically realizable example routings** and keeps P0 available.

**V-A (consistency route).** Spatial tendons wrapping MuJoCo cylinders concentric with the joints, terminated on
the links. Constant moment arms equal to `A`. Must reproduce P0's `ℓ_v`, `J_v` within 1e−8 — a regression check
of the spatial machinery, not a new physics level.

**V-B (variable-moment-arm route, the P1 virtual geometry).** Tendons pass through guide holes (sites) offset from
the joint axes, as in the legacy single-tendon finger, terminating on link 1 (tendon 1) and link 2 (tendon 2).
Site offsets are chosen so that `J_v(q0) = −A` exactly at home and the moment arms change with configuration
(target: ≥ 20 % variation over the joint range). Topology changes (a tendon losing contact with a guide) are
detected and flagged; the joint range is restricted to the valid region.

**R-A (rig route, demo topology).** Per channel: fixed base guide → idler coaxial with the rig joint (wrap
`β0 + q1`, crossed entry) → free span along the rig link → link-mounted idler (half wrap) → free span → second
coaxial idler (wrap `β0 − q1`) → fixed exit guide → emulator. Geometric length is invariant in `q1`; the
**wrap angles, contact loads `N_k = 2T̄ sin(β_k/2)` and capstan ratios change with `q1`**. This is the
configuration-dependent friction the four-motor rig is meant to expose.

**R-B (rig route, length-varying).** As R-A, but the return strand bypasses the coaxial idler through a fixed
offset guide, so `ℓ_R(q1)` varies nonlinearly. Used as a geometry sweep, not as the primary route.

## 4. Distributed cable (P2)

A channel is a chain of `N+1` segments separated by `N` contacts (actuator drum and emulator clamp are the two
boundary ports). Supported contact types:

* **Rolling idler:** cable does not slip; idler angle `θ_k`, inertia `J_k`, bearing friction with an
  elastoplastic presliding state `z_k` and load-dependent level `τ_c + μ_b r_b N_k`. Radius enters bearing and
  bending terms only.
* **Sliding guide:** cable slip `s_k`; presliding state on the slip coordinate; sliding limit from the capstan
  relation `T_high = T_low e^{μβ}` with `β` the current wrap. Radius does not enter the exponent; bending loss is
  a separately labelled term.

Segment constitutive law: standard linear solid (equilibrium spring `EA/L_i` in parallel with a Maxwell branch
`k_m, τ_m`) plus optional cubic stiffening; unilateral (slack segments carry no force and store no energy);
small viscous term. Unknowns per substep: contact rates; solved implicitly (Newton with a bracketing fallback,
as the existing plant does for one unknown). Limits checked by tests: one rolling contact ⇒ the existing Phase 2
plant; frictionless contacts ⇒ springs in series; one sliding contact at steady slip ⇒ the capstan ratio.

## 5. History and imperfections (P3)

Separate state mechanisms, each switchable:

| Time scale | State | Law (hypothetical, category C) |
|---|---|---|
| fast | presliding `z_k` per contact | elastoplastic, recoverable |
| seconds | Maxwell `v_i` per segment | linear, recoverable |
| minutes–hours | creep `c_i` per segment | rate ∝ (T/T_ref)ⁿ above threshold, irreversible |
| once | termination settling `s` | saturating, irreversible |
| use | conditioning `d` | driven by material transport over contacts |
| sensing | encoder bias drift, current gain/bias, quantisation, sample timing | telemetry-only **or** inner-loop configurations, never silently both |

Specimens carry a persistent ID and history across sessions. Recalibration may re-zero the estimator; it never
resets the cable's history unless the experiment declares a physical intervention.

## 6. Energy ledger

Derived per channel from the implemented states:

```
W_in + W_out + W_support + W_param = ΔE_stored + E_diss + r_E
E_stored = Σ segments (equilibrium + Maxwell spring energy) + Σ contacts (½ J ω² + presliding spring)
E_diss   = Maxwell dashpots + viscous + bearing/sliding friction (plastic part) + aging
W_support = work done by the rig link on the cable through moving contacts (R-A/R-B)
```

Transmission energy, virtual-port work, link-actuator work and motor electrical power are reported separately;
no whole-loop passivity claim follows from the transmission ledger.

## 7. Verification gates (before any learning)

Tolerances are inherited from the two-tendon study where they exist and are not lowered later.

| Gate | Pass criterion |
|---|---|
| V-A reproduces P0 | `ℓ_v`, `J_v`, torques within 1e−8 SI on the 143-point grid |
| V-B Jacobian | central differences vs analytic/MuJoCo within 1e−7 m/rad over the valid range; topology flags correct |
| Virtual work | `τᵀ δq = −Tᵀ δℓ_v` within 1e−8 |
| Rig geometry | R-A length invariance within 1e−9 m; tangency and continuity at every grid point |
| Cable limits | one-contact ⇒ Phase 2 plant within 1e−6 N; series springs; capstan ratio within 1 % at steady slip |
| Constitutive | relaxation and creep match the analytic SLS / Norton solutions within 1 % |
| Energy | isolated ledger residual decreasing with step refinement and below 1e−6 J per cycle |
| Coupled | independent channels, single finger integration, contact raises tension, pretension equilibrium |
| Numerics | outer/plant step refinement within the existing 0.25° / 0.5° / 1 N tolerances |

Failures are reported, never hidden by changing tolerances.

## 8. Evaluation plan (after the gates)

1. **Frozen transfer:** P0 estimators (rigid, compliance, physics EKF, three GRU seeds) unchanged, on matched
   tasks for P0/P1/P2/P3: free, independent q1, independent q2, coordinated, reversal, contact, release,
   sustained load, repeated cycles.
2. **One-mechanism ablations:** constant vs variable Jacobian; lumped vs distributed; frictionless vs stick-slip;
   elastic vs viscoelastic; fixed vs evolving; clean vs imperfect sensing; frozen vs recalibrated.
3. **Controlled recalibration** (methods A–E) with explicit information regimes and equal reported budgets:

| Regime | Allowed |
|---|---|
| privileged offline identification | joint + load-cell labels on calibration runs only |
| one-time reference calibration | jig homing, known pretension |
| runtime inference | the eight `motor{1,2}/…` channels only |
| online adaptation | motor channels only; only if 1–3 leave a reproducible failure |

Splits are by specimen and complete operating history; the final test set is sealed until all methods and
hyperparameters are frozen.

## 9. Decisions needed before Milestone B

1. Rig route: R-A as primary (demo topology) with R-B as a sweep — or real dimensions if available.
2. Virtual route: V-B with moment arms matched to `A` at home.
3. Rig-link actuator: finite position servo (its tracking error and work recorded) rather than ideal.
4. Material parameters: all category C (hypothetical) with sweeps, unless measured cable data are supplied.

## 10. Revision 2026-11-16 — intended hardware topology (supersedes sections 2–3 for the study)

User decision, from the rig sketch: the four motors are grounded; tendon 1 drives the MCP directly through P1; tendon 2
reaches the PIP pulley P2 through grounded idlers and an idler coaxial with the MCP. The joints are **physical** and
move under the tendon forces; the emulation motors run in **force mode** and render the virtual finger's load.

| Element | Model (`til/phase4_hardware.py`, `hardware_routes` in `til/phase4_geometry.py`) |
|---|---|
| Tendon 1 | actuator port → P1 (R1 = 15 mm, fixed to the link, no slip, half wrap; transport R1·q1) → emulator port |
| Tendon 2 | actuator → grounded idler → MCP coaxial idler (free, wrap π/2 − q1) → link → P2 (R2 = 10 mm, no slip; transport R2·q2) → link → stacked coaxial idler (wrap π/2 + q1) → grounded idler → emulator |
| Equivalent winding | the actuator-side strand of tendon 2 shortens by r·q1, so A = [[R1, 0], [r, R2]] = P0's A follows from the geometry |
| Mechanism | link (q1) and P2 (q2) with the virtual finger's link inertia (matched design), matched joint damping, bearing friction (H2+), hard stops; horizontal rig plane |
| Generalized force | τ = Σ_joint pulleys R (T_before − T_after) e_dof − Σ_i T_i ∂L_i/∂q (moving coaxial-idler routing) |
| Emulators | force mode: finite carriage mass, current-loop lag, load-cell feedback, active damping; command T_e = −A⁻ᵀ τ_env(q, q̇) |
| Virtual environment | MuJoCo finger driven kinematically by the measured rig state: gravity, return springs, object contact (rendered contact time constant 20 ms, see gates) |
| Joint labels | physical rig angles; the observer still reads only the eight actuator-motor channels |

Levels on this topology: **H1** resolved routing, physical joints, frictionless contacts/bearings; **H2** + stateful
bearing/guide friction; **H3** + slow history, telemetry and inner-loop imperfections. **P0** stays the frozen
historical reference. The first Phase 4 architecture (virtual joints, V-B spatial route, rig-link servo, R-A/R-B) is
kept in `til/phase4_hil.py` with its gates as an alternative, not used in the study.

Sign convention correction to section 2: in this code base x_out > 0 stretches the cable, so the virtual-joint emulator
command is x_out = [ℓ_v(q) − ℓ_v(q0)] − [ℓ_R(q_r) − ℓ_R(q_r0)].
