"""Phase 4 estimators: the frozen P0 set and the recalibration methods A-E.

Information regimes (recorded with every result):
  runtime inference      the eight motor{1,2}/{position,velocity,current,applied_command} channels only
  one-time reference     known-home jig (encoder re-zero from the first frame), known 10 N pretension
  privileged offline     joint labels on calibration runs only (never test runs)

Methods
  A rigid geometry          x = A (q - q0); no fitted parameter          (A-rehome: plus known-home re-zero)
  B low-order compliance    x = A (q - q0) + c (Tbar - 10) + o, per channel (the P0 'compliance' method, refit)
  C reduced physics         x = A_hat dq + quadratic(dq) + c Tbar + h s(x) + o, per channel: configuration-
                            dependent winding (quadratic length map), elastic compliance, and a direction-
                            dependent offset from a causal play operator s (friction memory); solved for q by Newton
  D physics EKF             C's measurement model inside the P0 reduced free-space EKF; covariances from the
                            same calibration runs
  E motor-only GRU          the P0 architecture (8-32-2), retrained on the same calibration runs, 3 seeds
"""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .observer import fit_scaler, windows
from .two_tendon import ARM, HOME
from .two_tendon_experiment import gravity
from .two_tendon_observer import PhysicsObserver, TwoMotorObserver, corrected, fit_compliance, predict, rigid, tension

ROOT = Path(__file__).resolve().parents[1]
FROZEN_CALIBRATION = ROOT/"figures/two-tendon/calibrated-physics.json"
FROZEN_MODELS = ROOT/"figures/two-tendon/models"


def load_frozen():
    cal = json.loads(FROZEN_CALIBRATION.read_text())
    models = []
    for seed in (0, 1, 2):
        cp = torch.load(FROZEN_MODELS/f"seed{seed}/checkpoint.pt", weights_only=True)
        m = TwoMotorObserver(); m.load_state_dict(cp["state_dict"])
        models.append((m, {k: np.array(v, np.float32) for k, v in cp["scaler"].items()}))
    return cal, models


def frozen_estimates(run, cal, models):
    out = dict(rigid=rigid(run), compliance=corrected(run, cal), physics=PhysicsObserver(cal).predict(run))
    for i, (m, scaler) in enumerate(models):
        out[f"gru-seed{i}"] = predict(m, run, scaler)
    return out


def rehomed(run):
    """Known-home jig reference: the first acquired motor frame is defined as home (one-time setup privilege)."""
    r = copy.copy(run); r["x"] = run["x"].copy()
    r["x"][:, [0, 4]] -= run["x"][0, [0, 4]]
    return r


# ------------------------------------------------------------------------------------------------- method C
def play_operator(x, width):
    """Causal normalized play (backlash) operator: a direction-dependent memory state in [-1, 1]."""
    s = np.zeros(len(x)); state = 0.; prev = x[0]
    for k, value in enumerate(x):
        state = min(1., max(-1., state + (value - prev)/width)); prev = value; s[k] = state
    return s


def _basis(dq):
    dq = np.atleast_2d(dq)
    return np.column_stack([dq[:, 0], dq[:, 1], dq[:, 0]**2, dq[:, 0]*dq[:, 1], dq[:, 1]**2])


def _basis_jacobian(dq):
    a, b = dq
    return np.array([[1., 0.], [0., 1.], [2*a, 0.], [b, a], [0., 2*b]])


def reduced_features(run, width):
    ti, to = tension(run)
    force = (ti + to)/2 - 10.
    x = .02*run["x"][:, [0, 4]]
    s = np.column_stack([play_operator(x[:, j], width) for j in range(2)])
    return x, force, s


def fit_reduced(runs, width=3e-4):
    train = [r for r in runs if r["split"] == "train"]
    if not train:
        raise ValueError("calibration runs required")
    coef = []
    for j in range(2):
        rows, ys = [], []
        for r in train:
            x, force, s = reduced_features(r, width)
            rows.append(np.column_stack([_basis(r["y"] - HOME), force[:, j], s[:, j], np.ones(len(x))]))
            ys.append(x[:, j])
        X, y = np.concatenate(rows), np.concatenate(ys)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        coef.append(c.tolist())
    return dict(method="C", coefficients=coef, play_width_m=width, training_ids=[r["id"] for r in train],
                terms=["dq1", "dq2", "dq1^2", "dq1*dq2", "dq2^2", "compliance (Tbar-10)", "play offset", "offset"],
                scope="Privileged offline identification: joint labels on calibration runs only.")


def reduced_measurement(cal, run):
    x, force, s = reduced_features(run, cal["play_width_m"])
    c = np.array(cal["coefficients"])
    return x - force*c[:, 5] - s*c[:, 6] - c[:, 7]  # = winding map g(dq)


def reduced_g(cal, dq):
    c = np.array(cal["coefficients"])
    return _basis(dq)[0] @ c[:, :5].T, (c[:, :5] @ _basis_jacobian(dq))


def reduced_predict(run, cal):
    z = reduced_measurement(cal, run)
    dq = np.zeros(2); out = []
    for zk in z:
        for _ in range(4):
            g, G = reduced_g(cal, dq)
            dq = dq - np.linalg.lstsq(G, g - zk, rcond=None)[0]
        out.append(HOME + dq)
    return np.array(out)


class ReducedEKF(PhysicsObserver):
    """P0's reduced free-space dynamics with method C's identified winding map as the measurement and torque map."""
    def __init__(self, cal):
        super().__init__(cal)

    def transition(self, x, T, dt):
        q, v = x[:2], x[2:]
        G = reduced_g(self.c, q - HOME)[1]  # d(winding)/dq, the identified moment-arm matrix
        force = G.T @ (T - 10.) - .08*(q - HOME) + gravity(q) - gravity(HOME)
        vnew = np.linalg.solve(self.M + dt*self.B + dt*dt*self.K, self.M @ v + dt*force)
        return np.r_[q + dt*vnew, vnew]

    def predict(self, run):
        _, to = tension(run); z = reduced_measurement(self.c, run)
        dt = float(np.median(np.diff(run["time"]))); x = np.r_[HOME, [0., 0.]]
        P = np.diag([np.deg2rad(.2)**2]*2 + [.05**2]*2)
        Q = np.array(self.c["state_process_covariance"]); R = np.array(self.c["length_measurement_covariance_m2"])
        out = []
        for i, T in enumerate(to):
            a = self.transition(x, T, dt)
            F = np.column_stack([(self.transition(x + np.eye(4)[j]*1e-5, T, dt) - a)/1e-5 for j in range(4)])
            P = F @ P @ F.T + Q
            g, G = reduced_g(self.c, a[:2] - HOME)
            H = np.column_stack((G, np.zeros((2, 2))))
            K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
            x = a + K @ (z[i] - g)
            I = np.eye(4) - K @ H; P = I @ P @ I.T + K @ R @ K.T
            out.append(x[:2].copy())
        return np.array(out)


def fit_reduced_ekf(runs, width=3e-4):
    cal = fit_reduced(runs, width)
    train = [r for r in runs if r["split"] == "train"]
    obs = ReducedEKF(dict(cal, state_process_covariance=np.eye(4).tolist(), length_measurement_covariance_m2=np.eye(2).tolist()))
    meas, proc = [], []
    for r in train:
        dt = float(np.median(np.diff(r["time"]))); _, to = tension(r)
        z = reduced_measurement(cal, r)
        meas.append(z - np.array([reduced_g(cal, q - HOME)[0] for q in r["y"]]))
        qd = np.gradient(r["y"].astype(float), dt, axis=0); state = np.column_stack((r["y"], qd))
        forecast = np.array([obs.transition(s, T, dt) for s, T in zip(state[:-1], to[:-1])])
        proc.append(state[1:] - forecast)
    cal["length_measurement_covariance_m2"] = (np.cov(np.concatenate(meas).T) + np.eye(2)*1e-12).tolist()
    cal["state_process_covariance"] = (np.cov(np.concatenate(proc).T) + np.diag([1e-12, 1e-12, 1e-6, 1e-6])).tolist()
    cal["method"] = "D"
    return cal


# ------------------------------------------------------------------------------------------------- method E
def train_gru(runs, seed, epochs=60):
    """Same architecture, optimizer, windows and validation selection as the P0 GRU."""
    torch.set_num_threads(2); torch.manual_seed(seed); np.random.seed(seed); torch.use_deterministic_algorithms(True)
    scaler = fit_scaler(runs); x, y = windows(runs, scaler)
    model = TwoMotorObserver(); opt = torch.optim.Adam(model.parameters(), lr=.003)
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    best, state, selected, history = float("inf"), None, 0, []
    val_runs = [r for r in runs if r["split"] == "validation"]
    for epoch in range(epochs):
        model.train(); losses = []
        for xb, yb in loader:
            est, _ = model(xb); loss = ((est[:, 32:] - yb[:, 32:])**2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); losses.append(loss.item())
        val = float(np.mean([np.mean(((predict(model, r, scaler) - r["y"])/scaler["y_std"])**2) for r in val_runs]))
        history.append(dict(epoch=epoch + 1, training_loss=float(np.mean(losses)), validation_loss=val))
        if val < best:
            best, state, selected = val, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(state)
    return model, scaler, dict(seed=seed, selected_epoch=selected, history=history, windows=int(len(x)))


# ------------------------------------------------------------------------------------------------- metrics
def aggregate(rows):
    """Mean of per-run scores (each run weighs equally), per mask and metric."""
    out = {}
    for mask in rows[0]:
        vals = [r[mask] for r in rows if r.get(mask)]
        if not vals:
            continue
        out[mask] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        out[mask]["runs"] = len(vals)
    return out


# ------------------------------------------------------------------------------------------------- design-A variants
# The hardware rig's winding matrix is a design value recorded with every run ("rig_A"). For P0 runs it equals ARM.
def run_A(run):
    return np.array(run["config"].get("rig_A", ARM.tolist()), float)


def rigid_A(run, A=None):
    A = run_A(run) if A is None else A
    return HOME + np.linalg.solve(A, .02*run["x"][:, [0, 4]].T).T


def corrected_A(run, calibration, A=None):
    A = run_A(run) if A is None else A
    ti, to = tension(run)
    delta = ((ti + to)/2 - 10.)*calibration["compliance_m_per_N"] + calibration["offset_m"]
    return HOME + np.linalg.solve(A, (.02*run["x"][:, [0, 4]] - delta).T).T


class PhysicsObserverA(PhysicsObserver):
    """P0 reduced EKF with the run's design winding matrix in the torque map and the measurement."""
    def __init__(self, calibration, A):
        super().__init__(calibration); self.A = np.asarray(A, float)

    def transition(self, x, T, dt):
        q, v = x[:2], x[2:]
        force = self.A.T @ (T - 10.) - .08*(q - HOME) + gravity(q) - gravity(HOME)
        vnew = np.linalg.solve(self.M + dt*self.B + dt*dt*self.K, self.M @ v + dt*force)
        return np.r_[q + dt*vnew, vnew]

    def predict(self, run):
        ti, to = tension(run); dt = float(np.median(np.diff(run["time"]))); x = np.r_[HOME, [0., 0.]]
        P = np.diag([np.deg2rad(.2)**2]*2 + [.05**2]*2)
        Q = np.array(self.c["state_process_covariance"]); R = np.array(self.c["length_measurement_covariance_m2"])
        H = np.column_stack((self.A, np.zeros((2, 2)))); out = []
        for i, T in enumerate(to):
            a = self.transition(x, T, dt)
            F = np.column_stack([(self.transition(x + np.eye(4)[j]*1e-5, T, dt) - a)/1e-5 for j in range(4)])
            P = F @ P @ F.T + Q; K = np.linalg.solve(H @ P @ H.T + R, H @ P).T
            z = .02*run["x"][i, [0, 4]] - ((ti[i] + to[i])/2 - 10.)*np.array(self.c["compliance_m_per_N"]) - np.array(self.c["offset_m"])
            x = a + K @ (z - self.A @ (a[:2] - HOME)); I = np.eye(4) - K @ H; P = I @ P @ I.T + K @ R @ K.T
            out.append(x[:2].copy())
        return np.array(out)


def fit_compliance_A(runs, A):
    """P0 train-only compliance + EKF covariance calibration, with the rig's design winding matrix."""
    A = np.asarray(A, float)
    training = [r for r in runs if r["split"] == "train"]
    if not training:
        raise ValueError("calibration runs required")
    slopes, offsets = [], []
    for i in range(2):
        zs, ys = [], []
        for r in training:
            ti, to = tension(r)
            zs.append((ti[:, i] + to[:, i])/2 - 10.)
            ys.append(.02*r["x"][:, 4*i] - (r["y"] - HOME) @ A[i])
        z, y = np.concatenate(zs), np.concatenate(ys)
        slope, offset = np.linalg.lstsq(np.column_stack((z, np.ones(len(z)))), y, rcond=None)[0]
        slopes.append(float(slope)); offsets.append(float(offset))
    cal = dict(compliance_m_per_N=slopes, offset_m=offsets, training_ids=[r["id"] for r in training], A=A.tolist(),
               scope="Privileged offline: joint labels on calibration runs only; design winding matrix.")
    obs = PhysicsObserverA(dict(cal, state_process_covariance=np.eye(4).tolist(), length_measurement_covariance_m2=np.eye(2).tolist()), A)
    meas, proc = [], []
    for r in training:
        dt = float(np.median(np.diff(r["time"]))); ti, to = tension(r)
        measured = .02*r["x"][:, [0, 4]] - ((ti + to)/2 - 10.)*slopes - offsets
        meas.append(measured - (r["y"] - HOME) @ A.T)
        qd = np.gradient(r["y"].astype(float), dt, axis=0); state = np.column_stack((r["y"], qd))
        proc.append(state[1:] - np.array([obs.transition(s, T, dt) for s, T in zip(state[:-1], to[:-1])]))
    cal["length_measurement_covariance_m2"] = (np.cov(np.concatenate(meas).T) + np.eye(2)*1e-12).tolist()
    cal["state_process_covariance"] = (np.cov(np.concatenate(proc).T) + np.diag([1e-12, 1e-12, 1e-6, 1e-6])).tolist()
    return cal


def design_estimates(run, cal):
    """Frozen P0 calibration numbers with the run's design winding matrix (known geometry, no refit)."""
    A = run_A(run)
    return {"rigid-designA": rigid_A(run, A), "compliance-designA": corrected_A(run, cal, A),
            "physics-designA": PhysicsObserverA(cal, A).predict(run)}
