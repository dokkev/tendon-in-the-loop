"""Render a recorded Phase 4 hardware-topology episode: the MuJoCo virtual finger next to the simulated rig.

Left: MuJoCo render of the virtual finger and pad at the recorded rig angles and pad deflection.
Right: schematic of the simulated hardware drawn from recorded states only: grounded actuator and emulation
motors (drum angle, carriage displacement), tendon 1 on P1 (MCP pulley), tendon 2 through the MCP idler pulleys
(axial passage, drawn in projection) to P2 (PIP pulley), cable tensions at both ends, the rendered emulator command.
Bottom: joint angles and pad contact force with the current time marked.
Every drawn quantity comes from the HDF5 record; the drawing dimensions are schematic.
"""
import argparse, json, sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from til.finger import FingerParams
from til.two_tendon import TwoTendonFinger

C = dict(t1="#2a78d6", t2="#eb6834", ink="#0b0b0b", muted="#52514e", grid="#e4e3df", surface="#fcfcfb", pad="#1baf7a")


def load(path):
    with h5py.File(path) as f:
        g = next(iter(f["episodes"].values()))
        keys = ["time", "simulation/joint_positions", "simulation/object_position", "simulation/contact_force", "simulation/emulator_commands",
                "protocol/desired_joints"] + [f"{p}{i}/{k}" for i in (1, 2) for p, k in (("motor", "position"), ("emulator", "position"),
                ("transmission", "input_tension"), ("transmission", "output_tension"))]
        d = {k: g[k][()] for k in keys}
        d["config"] = json.loads(g.attrs["config"])
        d["fidelity"] = json.loads(g.attrs["resolved_configuration"]).get("fidelity", {})
    return d


def mujoco_frames(d, idx, size=(360, 420)):
    spec = d["config"]
    p = FingerParams(obj_x=spec["object_x"] if spec["contact"] else .2, k_obj=spec["object_stiffness"],
                     q1_range=(-np.pi/6, np.pi/3), q2_range=(0., 5*np.pi/6), T_max=80.)
    f = TwoTendonFinger(p)
    f.model.site_rgba[:, 3] = 0.
    f.model.geom_rgba[f.obj_geom] = [.106, .686, .478, 1.]
    r = mujoco.Renderer(f.model, *size)
    cam = mujoco.MjvCamera(); cam.lookat[:] = [.045, 0, -.025]; cam.distance = .17; cam.azimuth = 90; cam.elevation = 0
    bg = np.array([252, 252, 251], np.uint8)
    out = []
    for i in idx:
        f.data.qpos[f.qadr] = d["simulation/joint_positions"][i]
        f.data.qpos[f.obj_qadr] = d["simulation/object_position"][i]
        mujoco.mj_forward(f.model, f.data)
        r.update_scene(f.data, cam)
        img = r.render().copy()
        img[(img < 4).all(-1)] = bg
        out.append(img)
    return out


def rot(v, q):
    c, s = np.cos(q), np.sin(q)
    return np.array([v[0]*c + v[1]*s, -v[0]*s + v[1]*c])


def draw_rig(ax, d, i):
    q = d["simulation/joint_positions"][i]
    ax.clear(); ax.set_xlim(-.32, .11); ax.set_ylim(-.13, .13); ax.set_aspect("equal"); ax.axis("off")
    R1, R2, L, ra = .015, .010, .05, .004
    pip = rot((L, 0.), q[0])
    # link and pulleys
    ax.plot([0, pip[0]], [0, pip[1]], color=C["muted"], lw=7, alpha=.35, solid_capstyle="round")
    for center, r, ang, label in (((0., 0.), R1, q[0], "P1"), (pip, R2, q.sum(), "P2")):
        ax.add_patch(plt.Circle(center, r, color=C["muted"], alpha=.18))
        ax.add_patch(plt.Circle(center, r, fill=False, color=C["ink"], lw=1))
        tip = np.asarray(center) + rot((r, 0.), ang)
        ax.plot([center[0], tip[0]], [center[1], tip[1]], color=C["ink"], lw=1)
        ax.text(center[0], center[1] + r + .008, label, ha="center", fontsize=7, color=C["ink"])
    ax.plot([-.006, .006, 0, -.006], [-.03, -.03, 0, -.03], color=C["muted"], lw=1)  # ground support
    # MCP idler pulleys (axial passage, projected): two pairs drawn beside P1
    for zc in (.024, -.024):
        ax.add_patch(plt.Circle((-.012, zc), ra, fill=False, color=C["t2"], lw=1))
    # tendon 1: actuator -> P1 -> emulator
    xa, xe = -.25, -.25
    T = {k: d[k][i] for k in ("transmission1/input_tension", "transmission1/output_tension", "transmission2/input_tension", "transmission2/output_tension")}
    lw = lambda t: .6 + t/6
    ax.plot([xa + .03, 0], [-R1, -R1], color=C["t1"], lw=lw(T["transmission1/input_tension"]))
    ax.plot([xe + .03, 0], [R1, R1], color=C["t1"], lw=lw(T["transmission1/output_tension"]))
    th = np.linspace(np.pi/2, 3*np.pi/2, 30)
    ax.plot(R1*np.cos(-th + np.pi), R1*np.sin(-th + np.pi), color=C["t1"], lw=1.5)
    # tendon 2: grounded idlers -> MCP idler pulleys -> along the link -> P2
    side = rot((0., 1.), q[0])
    for zc, key, sgn in ((-.06, "transmission2/input_tension", -1), (.06, "transmission2/output_tension", 1)):
        ax.plot([xa + .03, -.035], [zc, zc], color=C["t2"], lw=lw(T[key]))
        ax.add_patch(plt.Circle((-.035, zc - np.sign(zc)*.006), .006, fill=False, color=C["muted"], lw=1))
        ax.plot([-.041, -.016], [zc - np.sign(zc)*.006, sgn*.024], color=C["t2"], lw=lw(T[key]), alpha=.8)
        a = sgn*R2*side
        ax.plot([a[0], pip[0] + a[0]], [a[1], pip[1] + a[1]], color=C["t2"], lw=lw(T[key]))
    th2 = np.linspace(-np.pi/2, np.pi/2, 30) - q[0]
    ax.plot(pip[0] + R2*np.cos(th2), pip[1] + R2*np.sin(th2), color=C["t2"], lw=1.5)
    # grounded motors with drum angle and carriage displacement (displacements magnified x5)
    for zc, label, kind, key, color in ((-.015, "actuator 1", "a", "motor1/position", C["t1"]), (.015, "emulator 1", "e", "emulator1/position", C["t1"]),
                                        (-.06, "actuator 2", "a", "motor2/position", C["t2"]), (.06, "emulator 2", "e", "emulator2/position", C["t2"])):
        x0 = -.29
        ax.add_patch(plt.Rectangle((x0, zc - .009), .03, .018, color=C["muted"], alpha=.25))
        ax.plot([x0 - .004]*2, [zc - .012, zc + .012], color=C["ink"], lw=2)
        if kind == "a":
            ang = d[key][i]
            ax.plot([xa + .03, xa + .03 + .006*np.cos(ang)], [zc, zc + .006*np.sin(ang)], color=C["ink"], lw=1)
            ax.add_patch(plt.Circle((xa + .03, zc), .006, fill=False, color=C["ink"], lw=1))
        else:
            dx = 5*d[key][i]
            ax.add_patch(plt.Rectangle((xa + .022 + dx, zc - .005), .012, .01, color=color, alpha=.6))
        ax.text(x0, zc + (.013 if zc > 0 else -.02), label, fontsize=6.5, color=C["ink"])
    t_in1, t_out1 = T["transmission1/input_tension"], T["transmission1/output_tension"]
    t_in2, t_out2 = T["transmission2/input_tension"], T["transmission2/output_tension"]
    ax.text(-.2, -.115, f"T_a1 {t_in1:5.1f} N   T_e1 {t_out1:5.1f} N\nT_a2 {t_in2:5.1f} N   T_e2 {t_out2:5.1f} N", fontsize=7, color=C["ink"], family="monospace")
    ax.text(-.32, .12, "simulated rig (schematic; MCP idler pulleys shown in projection)", fontsize=7, color=C["muted"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("episode"); ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20); ap.add_argument("--start", type=float, default=0.); ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--snapshots", type=float, nargs="*", default=[]); a = ap.parse_args()
    d = load(a.episode); t = d["time"]
    end = a.end or t[-1]
    idx = [int(np.argmin(abs(t - s))) for s in np.arange(a.start, end, 1/a.fps)]
    snaps = [int(np.argmin(abs(t - s))) for s in a.snapshots]
    imgs = mujoco_frames(d, idx + snaps)
    q = np.degrees(d["simulation/joint_positions"]); qr = np.degrees(d["protocol/desired_joints"])
    fig = plt.figure(figsize=(9.6, 5.4), facecolor=C["surface"])
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.25], width_ratios=[1, 1.15], hspace=.25, wspace=.05)
    ax_mj, ax_rig, ax_ts = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    ax_ts.plot(t, q[:, 0], color=C["t1"], lw=1.5, label="q1 MCP (rig)"); ax_ts.plot(t, q[:, 1], color=C["t2"], lw=1.5, label="q2 PIP (rig)")
    ax_ts.plot(t, qr[:, 0], color=C["t1"], lw=1, ls=":"); ax_ts.plot(t, qr[:, 1], color=C["t2"], lw=1, ls=":", label="desired (dotted)")
    ax_ts.set_ylabel("deg", fontsize=8); ax_ts.set_xlabel("time (s)", fontsize=8); ax_ts.tick_params(labelsize=7)
    for s in ("top", "right"):
        ax_ts.spines[s].set_visible(False)
    ax_ts.grid(color=C["grid"], lw=.6); ax_ts.set_facecolor(C["surface"])
    axF = ax_ts.twinx(); axF.fill_between(t, 0, d["simulation/contact_force"], color=C["pad"], alpha=.15, lw=0)
    axF.set_yticks([]); axF.set_ylim(0, max(1., d["simulation/contact_force"].max())*3)
    ax_ts.legend(fontsize=7, frameon=False, loc="upper right", ncol=3)
    ax_ts.text(.005, .9, "shaded: pad contact force", transform=ax_ts.transAxes, fontsize=7, color=C["muted"])
    marker = ax_ts.axvline(0, color=C["ink"], lw=1)
    level = d["fidelity"].get("level", d["config"].get("fidelity_level"))
    def frame(k, i, img):
        ax_mj.clear(); ax_mj.imshow(img); ax_mj.axis("off")
        ax_mj.set_title(f"MuJoCo virtual finger  t = {t[i]:.2f} s", fontsize=8, color=C["ink"])
        ax_mj.text(5, img.shape[0] - 10, f"pad force {d['simulation/contact_force'][i]:.2f} N   emulator commands "
                   f"{d['simulation/emulator_commands'][i][0]:.1f} / {d['simulation/emulator_commands'][i][1]:.1f} N", fontsize=6.5, color=C["muted"])
        draw_rig(ax_rig, d, i); marker.set_xdata([t[i], t[i]])
        fig.suptitle(f"Phase 4 {level}: {d['config']['motion']}{' + contact' if d['config']['contact'] else ''} (synthetic)", fontsize=9, color=C["ink"])
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    frames = [frame(k, i, img) for k, (i, img) in enumerate(zip(idx, imgs[:len(idx)]))]
    small = [f_.resize((f_.width*2//3, f_.height*2//3), Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=128) for f_ in frames]
    small[0].save(out.with_suffix(".gif"), save_all=True, append_images=small[1:], duration=int(1000/a.fps), loop=0, optimize=True)
    for s, i, img in zip(a.snapshots, snaps, imgs[len(idx):]):
        frame(0, i, img).save(out.parent/f"{out.stem}_t{s:.2f}.png")
    print(out.with_suffix(".gif"), len(frames), "frames")


if __name__ == "__main__":
    main()
