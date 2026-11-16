"""Phase 4 geometry: virtual finger routes (V-A, V-B) and physical rig routes (R-A, R-B).

Two geometries are kept separate (PHASE4_DESIGN.md section 2):

* V, the virtual finger (MuJoCo spatial tendons). It gives l_v(q), J_v(q) and the generalized force
  tau = -J_v(q)^T T_out that MuJoCo applies through a unit-gain tendon actuator.
* R, the physical rig between the actuator drum and the emulator clamp. It gives the wrap angles,
  free spans and segment lengths that the distributed cable model (phase4_cable) uses, as functions of
  the rig-link angle q_r, which a separate link servo drives to follow the virtual q1.

Coordinates are planar (x, z) with the finger's convention: a positive joint or rig-link angle rotates
+x toward -z (clockwise in the x-z plane). All routes here are documented example routes, not
the dimensions of a built apparatus.
"""
from dataclasses import dataclass, field, asdict
from math import pi
import copy
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from .finger import FingerParams, build_mjcf
from .two_tendon import ARM, HOME, TwoTendonFinger

TWO_PI = 2*pi


def rotate(point, q):
    """Rotate a planar (x, z) point by joint angle q (positive = clockwise, +x toward -z)."""
    x, z = point
    c, s = np.cos(q), np.sin(q)
    return np.array([x*c + z*s, -x*s + z*c])


# --------------------------------------------------------------------------------------------------
# Planar belt solver (rig routes)
# --------------------------------------------------------------------------------------------------
@dataclass
class Node:
    """A route node. radius 0 marks a cable port (actuator drum exit or emulator clamp)."""
    name: str
    center: tuple
    radius: float = 0.
    direction: int = 1  # +1 counter-clockwise travel around the node (x-z plane), -1 clockwise
    frame: str = "base"  # "base" (fixed) or "link" (rotates with the rig link about the origin)
    kind: str = "port"  # "port", "rolling" (free idler, no slip) or "sliding" (fixed guide)
    layer: int = 0  # out-of-plane (y) layer; -1 = spans both layers (e.g. the link idler). Ports inherit.
    dof: int = -1  # joint pulleys: mechanism joint index (0 = MCP/q1, 1 = PIP/q2)


def _common_tangent(ci, rho_i, cj, rho_j):
    """Directed tangent from circle i to circle j. rho = direction*radius (signed).

    For travel direction t and left normal l = rot90(t), a point of tangency is p = c - rho*l.
    Requiring (p_j - p_i) parallel to t gives l.(c_j - c_i) = rho_j - rho_i.
    """
    c = cj - ci
    D = float(np.hypot(*c))
    s = (rho_j - rho_i)/D
    if not abs(s) < 1:
        raise ValueError("no common tangent: circles overlap for the requested wrap directions")
    alpha = np.arctan2(c[1], c[0]) - np.arcsin(s)
    l = np.array([-np.sin(alpha), np.cos(alpha)])
    return alpha, ci - rho_i*l, cj - rho_j*l, float(np.sqrt(D*D - (rho_j - rho_i)**2))


@dataclass
class RigRoute:
    """A planar rig route from the actuator port to the emulator port."""
    name: str
    nodes: list
    q_range: tuple = (-pi/6, pi/3)
    min_wrap_rad: float = np.deg2rad(2.)

    def __post_init__(self):
        if self.nodes[0].kind != "port" or self.nodes[-1].kind != "port":
            raise ValueError("a route starts and ends at a port")
        if any(n.kind == "port" for n in self.nodes[1:-1]) or any(n.radius <= 0 for n in self.nodes[1:-1]):
            raise ValueError("interior nodes must be contacts with positive radius")
        # Wrap branch: wraps are unwrapped continuously from the reference configuration, so a contact that
        # loses its wrap shows a negative wrap (invalid) instead of a spurious near-2*pi wrap.
        self._ref_wraps = None
        try:
            self._ref_wraps = self.evaluate(float(np.clip(HOME[0], *self.q_range)))["wraps"]
        except ValueError:
            self._ref_wraps = None

    @property
    def contacts(self):
        return self.nodes[1:-1]

    def centers(self, q):
        return [rotate(n.center, q) if n.frame == "link" else np.asarray(n.center, float) for n in self.nodes]

    def evaluate(self, q):
        """Spans, wraps, arcs and lumped segment lengths at rig angle q.

        Segment i (0..N) runs between node i and node i+1 and is assigned its free span plus half of
        each adjacent contact arc. This lumping is a model choice (documented in PHASE4_REFERENCE_PLANT.md).
        """
        centers = self.centers(q)
        spans, headings, points = [], [], []
        for a, b, ca, cb in zip(self.nodes[:-1], self.nodes[1:], centers[:-1], centers[1:]):
            alpha, pa, pb, length = _common_tangent(ca, a.direction*a.radius, cb, b.direction*b.radius)
            spans.append(length); headings.append(alpha); points.append((pa, pb))
        wraps = []
        for k, n in enumerate(self.contacts, start=1):
            wraps.append(float(np.mod(n.direction*(headings[k] - headings[k-1]), TWO_PI)))
        wraps = np.array(wraps)
        if self._ref_wraps is not None:
            wraps = wraps - TWO_PI*np.round((wraps - self._ref_wraps)/TWO_PI)
        radii = np.array([n.radius for n in self.contacts])
        arcs = radii*wraps
        spans = np.array(spans)
        half = np.r_[0., arcs/2] + np.r_[arcs/2, 0.]
        clearance = self._clearance(centers, points)
        valid = bool(np.all(wraps > self.min_wrap_rad) and np.all(wraps < TWO_PI - self.min_wrap_rad)
                     and np.all(spans > 0) and clearance > 0 and self.q_range[0] - 1e-9 <= q <= self.q_range[1] + 1e-9)
        return dict(spans=spans, wraps=wraps, arcs=arcs, segment_lengths=spans + half,
                    total_length=float(spans.sum() + arcs.sum()), headings=np.array(headings),
                    tangent_points=points, centers=centers, clearance=clearance, valid=valid)

    def _clearance(self, centers, points):
        """Smallest gap between a free span and any contact it does not touch (coaxial stacks excluded)."""
        gap = np.inf
        layers = [n.layer for n in self.nodes]
        layers[0], layers[-1] = layers[1], layers[-2]
        for i, (pa, pb) in enumerate(points):
            ends = {tuple(np.round(centers[i], 12)), tuple(np.round(centers[i+1], 12))}
            span_layers = {layers[i], layers[i+1]} - {-1} or {-1}
            d = pb - pa
            for k, n in enumerate(self.nodes):
                if n.kind == "port" or k in (i, i+1) or tuple(np.round(centers[k], 12)) in ends:
                    continue
                if n.layer != -1 and -1 not in span_layers and n.layer not in span_layers:
                    continue
                u = np.clip((centers[k] - pa) @ d/(d @ d), 0., 1.)
                gap = min(gap, float(np.hypot(*(pa + u*d - centers[k]))) - n.radius)
        return gap

    def segment_lengths(self, q):
        return self.evaluate(q)["segment_lengths"]

    def segment_jacobian(self, q, h=1e-6):
        """dL_i/dq by central differences of the exact tangent construction."""
        return (self.segment_lengths(q + h) - self.segment_lengths(q - h))/(2*h)

    def to_dict(self):
        return dict(name=self.name, q_range=list(self.q_range), min_wrap_rad=self.min_wrap_rad,
                    nodes=[asdict(n) for n in self.nodes])


def _ccw_tangent_point(center, radius, heading, direction):
    l = np.array([-np.sin(heading), np.cos(heading)])
    return np.asarray(center, float) - direction*radius*l


def rig_route(kind="R-A", idler_radius=.012, guide_radius=.008, link_length=.07, guide_wrap=pi/4,
              port_span=None, guide_drop=.03, offset_guide=(0., .006), offset_radius=.012,
              total_length=None, check=True, guide_kind="sliding", link_kind="rolling", link_dof=-1, link_radius=None):
    """Example rig routes (PHASE4_DESIGN.md section 3).

    R-A: actuator port -> fixed entry guide (sliding, CW) -> idler 1 coaxial with the rig joint (rolling,
    CCW, wrap pi/2 - q) -> span along the link -> link-mounted idler (rolling, CCW, half wrap) -> span back
    -> idler 2 coaxial with the joint (rolling, CCW, wrap pi/2 + q) -> fixed exit guide (sliding, CW) ->
    emulator port. The two coaxial wraps vary by -q and +q, so the geometric length is invariant in q
    while the wraps, contact loads and capstan ratios change.

    R-B: as R-A, but the return strand bypasses idler 2 through a fixed offset guide (sliding, CCW),
    so the geometric length varies nonlinearly with q.

    ``port_span`` (free length from each port to its guide) defaults to the value that makes the total
    home-length equal to the P0 lumped fixture (0.2 + 0.2 + 0.03*pi/2 m) when ``total_length`` is given.
    """
    r, rg = idler_radius, guide_radius
    rl = link_radius or r
    if abs(rl - r) > 1e-15:
        raise ValueError("this builder keeps the link pulley radius equal to the coaxial idler radius")
    # Entry guide: CW, the cable leaves it heading straight down along x = -r into idler 1.
    gin = np.array([-r - rg, guide_drop])
    gout = np.array([-r - rg, -guide_drop])
    def ports(span):
        # Arrival at the entry guide with heading -(pi/2 - guide_wrap); exit guide symmetric.
        h_in = -(pi/2 - guide_wrap)
        p_in = _ccw_tangent_point(gin, rg, h_in, -1)
        start = p_in - span*np.array([np.cos(h_in), np.sin(h_in)])
        h_out = -pi/2 - guide_wrap
        p_out = _ccw_tangent_point(gout, rg, h_out, -1)
        end = p_out + span*np.array([np.cos(h_out), np.sin(h_out)])
        return start, end
    def build(span):
        start, end = ports(span)
        nodes = [Node("actuator_port", tuple(start)),
                 Node("entry_guide", tuple(gin), rg, -1, "base", guide_kind, 0),
                 Node("joint_idler_1", (0., 0.), r, 1, "base", "rolling", 0),
                 Node("link_idler" if link_kind == "rolling" else "P2", (link_length, 0.), r, 1, "link", link_kind, -1, link_dof)]
        if kind == "R-A":
            nodes.append(Node("joint_idler_2", (0., 0.), r, 1, "base", "rolling", 1))
            nodes.append(Node("exit_guide", tuple(gout), rg, -1, "base", guide_kind, 1))
        elif kind == "R-B":
            # Return layer: a fixed sliding guide offset from the rig axis replaces the coaxial idler 2.
            ob = np.asarray(offset_guide, float)
            nodes.append(Node("offset_guide", tuple(ob), offset_radius, 1, "base", "sliding", 1))
            # Exit guide placed so that the offset guide's departing strand (heading down) is tangent.
            gout_b = np.array([ob[0] - offset_radius - rg, -guide_drop])
            nodes.append(Node("exit_guide", tuple(gout_b), rg, -1, "base", guide_kind, 1))
        else:
            raise ValueError("kind must be R-A or R-B")
        if kind == "R-B":
            p_out = _ccw_tangent_point(nodes[-1].center, rg, -pi/2 - guide_wrap, -1)
            end = p_out + span*np.array([np.cos(-pi/2 - guide_wrap), np.sin(-pi/2 - guide_wrap)])
        nodes.append(Node("emulator_port", tuple(end)))
        return RigRoute(kind, nodes)
    if port_span is None:
        if total_length is None:
            port_span = .08
        else:
            # The home length is affine in the port span (both port spans change by the same amount).
            a = build(.05).evaluate(HOME[0])["total_length"]
            port_span = .05 + (total_length - a)/2
    if port_span < .02:
        raise ValueError(f"port span {port_span:.4f} m too short: the route is longer than the requested total length")
    route = build(port_span)
    if check and not route.evaluate(HOME[0])["valid"]:
        raise ValueError("rig route invalid at home")
    return route


P0_TOTAL_LENGTH = .2 + .2 + .03*pi/2  # FIXTURES["large_guide"] free spans plus the guide arc


class FixedRoute:
    """A route whose geometry does not depend on the rig angle (segments and wraps given explicitly).

    Used for the MCP axial passage: tendon 2 is turned onto the MCP axis by an idler on the base, runs along the
    axis, and is turned back into the link plane by an idler on the link. A segment on the joint axis keeps its length
    when the link rotates (its slight twist is not modelled), so neither strand changes length with q1. This is a
    three-dimensional arrangement, represented here by its invariant lengths and wraps, not by the planar solver.
    """
    def __init__(self, name, contacts, segment_lengths, wraps):
        self.name, self.nodes = name, [Node("actuator_port", (0., 0.))] + list(contacts) + [Node("emulator_port", (0., 0.))]
        self._L = np.asarray(segment_lengths, float); self._w = np.asarray(wraps, float)
        if len(self._L) != len(contacts) + 1 or len(self._w) != len(contacts):
            raise ValueError("N contacts need N+1 segments and N wraps")
        self.q_range = (-pi/6, pi/3)

    @property
    def contacts(self):
        return self.nodes[1:-1]

    def evaluate(self, q):
        arcs = np.array([n.radius for n in self.contacts])*self._w
        return dict(spans=self._L - (np.r_[0., arcs/2] + np.r_[arcs/2, 0.]), wraps=self._w.copy(), arcs=arcs,
                    segment_lengths=self._L.copy(), total_length=float(self._L.sum()), valid=bool(self.q_range[0] - 1e-9 <= q <= self.q_range[1] + 1e-9),
                    tangent_points=[], centers=[], clearance=np.inf)

    def segment_lengths(self, q):
        return self._L.copy()

    def segment_jacobian(self, q, h=1e-6):
        return np.zeros(len(self._L))

    def to_dict(self):
        return dict(name=self.name, kind="fixed (MCP axial passage)", nodes=[asdict(n) for n in self.nodes],
                    segment_lengths=self._L.tolist(), wraps=self._w.tolist())


def hardware_routes(R1=.015, R2=.010, link_length=.05, idler_radius=.006, axial_idler_radius=.004, axial_length=.02,
                    total_length=P0_TOTAL_LENGTH, mcp_passage="axial", r_coax=None, coax_offset=None, guide_kind="rolling"):
    """The intended four-motor rig (user sketch and decisions, 2026-11-16): grounded motors, physical MCP link and PIP pulley.

    Tendon 1: actuator port -> P1 (MCP pulley: fixed to the link, no slip, half wrap) -> emulator port.
              Transport over P1 is R1*q1; the path does not change with q.
    Tendon 2, mcp_passage="axial" (default, decoupled): actuator port -> grounded idler -> MCP base idler (turns the
              cable onto the MCP axis) -> axial segment -> MCP link idler (turns it along the link) -> P2 (PIP pulley,
              no slip, half wrap; transport R2*q2 relative to the link) -> second link idler -> axial segment -> second
              base idler -> grounded idler -> emulator port. The MCP idler pulleys are free and distinct from P1.
              No strand changes length with q1: A = [[R1, 0], [0, R2]], tendon 2 puts no torque on the MCP.
    Tendon 2, mcp_passage="coaxial" (coupled variant): the cable wraps a free idler coaxial with the MCP (wrap
              pi/2 - q1 on the actuator side, pi/2 + q1 on the return), so A = [[R1, 0], [r_coax, R2]].
              ``coax_offset`` moves the actuator-side coaxial idler off the axis (configuration-dependent coupling).
    Documented example dimensions, not measured ones.
    """
    lp = (total_length - pi*R1)/2
    t1 = RigRoute("HW-1", [Node("actuator_port", (-lp, -R1)), Node("P1", (0., 0.), R1, 1, "base", "joint", 0, 0),
                           Node("emulator_port", (-lp, R1))])
    if mcp_passage == "coaxial":
        r = R2 if r_coax is None else r_coax
        if abs(r - R2) > 1e-15:
            raise ValueError("the coaxial builder keeps the coaxial idler radius equal to R2")
        t2 = rig_route("R-A", idler_radius=r, guide_radius=idler_radius, link_length=link_length, total_length=total_length,
                       guide_kind=guide_kind, link_kind="joint", link_dof=1)
        t2.name = "HW-2-coaxial"
        if coax_offset is not None:
            node = t2.nodes[2]
            t2.nodes[2] = Node(node.name, tuple(np.asarray(coax_offset, float)), node.radius, node.direction, node.frame, node.kind,
                               node.layer, node.dof)
            t2._ref_wraps = None
            t2._ref_wraps = t2.evaluate(HOME[0])["wraps"]
        return t1, t2
    if mcp_passage != "axial":
        raise ValueError("mcp_passage must be 'axial' or 'coaxial'")
    ra, rg = axial_idler_radius, idler_radius
    contacts = [Node("entry_guide", (0., 0.), rg, -1, "base", guide_kind),
                Node("mcp_base_idler_1", (0., 0.), ra, 1, "base", "rolling"),
                Node("mcp_link_idler_1", (0., 0.), ra, 1, "link", "rolling"),
                Node("P2", (link_length, 0.), R2, 1, "link", "joint", -1, 1),
                Node("mcp_link_idler_2", (0., 0.), ra, 1, "link", "rolling"),
                Node("mcp_base_idler_2", (0., 0.), ra, 1, "base", "rolling"),
                Node("exit_guide", (0., 0.), rg, -1, "base", guide_kind)]
    wraps = np.array([pi/4, pi/2, pi/2, pi, pi/2, pi/2, pi/4])
    radii = np.array([n.radius for n in contacts])
    arcs = radii*wraps
    spans = np.array([0., .03, axial_length, link_length - ra, link_length - ra, axial_length, .03, 0.])
    spans[0] = spans[-1] = (total_length - arcs.sum() - spans.sum())/2
    if spans[0] < .02:
        raise ValueError("port span too short for the requested total length")
    L = spans + np.r_[0., arcs/2] + np.r_[arcs/2, 0.]
    return t1, FixedRoute("HW-2-axial", contacts, L, wraps)


# --------------------------------------------------------------------------------------------------
# Virtual finger routes (MuJoCo spatial tendons)
# --------------------------------------------------------------------------------------------------
@dataclass
class VirtualRoute:
    """Virtual finger tendon routing, in the link frames of the existing finger model.

    V-A: spatial tendons wrapping MuJoCo cylinders concentric with the joints (a no-slip pulley winding
    constraint with termination on the pulley side), giving constant moment arms equal to A.
    V-B: tendons through guide sites (eyelets) offset from the joint axes; moment arms change with q.
    """
    kind: str = "V-B"
    radii: tuple = (.015, .010, .010)  # V-A: tendon-1 pulley at q1, tendon-2 pulleys at q1 and q2
    termination_angle_rad: tuple = (1.05, 1.05, 2.35)  # V-A: polar angle of the site leaving each pulley
    sites: dict = field(default_factory=dict)  # V-B: resolved site coordinates (x, z) by name

    def to_dict(self):
        return asdict(self)


def _arm(base, site, q):
    """Planar moment arm (signed: positive shortens with positive q) of a straight segment base -> R(q) site."""
    p = rotate(site, q)
    d = p - base
    L = np.hypot(*d)
    # dL/dq = d.(dp/dq)/L with dp/dq = (-x sin q + z cos q, -x cos q - z sin q) = rotate(site,q) derivative
    x, z = site
    dp = np.array([-x*np.sin(q) + z*np.cos(q), -x*np.cos(q) - z*np.sin(q)])
    return -float(d @ dp)/L, L


def _solve(fun, lo, hi, iterations=200):
    flo = fun(lo)
    for _ in range(iterations):
        mid = (lo + hi)/2
        fm = fun(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return (lo + hi)/2


def vb_route(p=None, tendon1_span=(.02, .02), tendon2_base=.02, guide_back=.032, distal_site=.026,
             guide_height=.0025):
    """Solve the V-B eyelet offsets so that J_v(HOME) = -A exactly.

    Tendon 1: base eyelet (-a, -h) -> link-1 eyelet (b, -h); h solved for arm 0.015 at q1 = HOME[0].
    Tendon 2: base eyelet (-c, -h2) -> link-1 guide (L1 - g, -k) -> link-2 eyelet (e, -k2); h2 and k2
    solved for arms 0.010 about joint 1 at HOME[0] and about joint 2 at HOME[1].
    """
    p = p or FingerParams()
    a, b = tendon1_span
    h = _solve(lambda h: _arm(np.array([-a, -h]), (b, -h), HOME[0])[0] - ARM[0, 0], 1e-4, .04)
    g = np.array([p.L1 - guide_back, -guide_height])
    h2 = _solve(lambda h2: _arm(np.array([-tendon2_base, -h2]), tuple(g), HOME[0])[0] - ARM[1, 0], 1e-5, .03)
    # About joint 2 the guide is fixed (link-1 frame); express it relative to the joint-2 origin.
    gj = g - np.array([p.L1, 0.])
    k2 = _solve(lambda k2: _arm(gj, (distal_site, -k2), HOME[1])[0] - ARM[1, 1], 1e-5, .03)
    sites = dict(t1_base=(-a, -h), t1_link1=(b, -h), t2_base=(-tendon2_base, -h2), t2_guide=tuple(g),
                 t2_link2=(distal_site, -k2))
    return VirtualRoute("V-B", sites={k: tuple(float(v) for v in s) for k, s in sites.items()})


def vb_analytic(route, q, p=None):
    """Closed-form V-B lengths and Jacobian (independent of MuJoCo), for verification."""
    p = p or FingerParams()
    s = route.sites
    a1, L1 = _arm(np.array(s["t1_base"]), s["t1_link1"], q[0])
    a21, L21 = _arm(np.array(s["t2_base"]), s["t2_guide"], q[0])
    gj = np.array(s["t2_guide"]) - np.array([p.L1, 0.])
    a22, L22 = _arm(gj, s["t2_link2"], q[1])
    return np.array([L1, L21 + L22]), -np.array([[a1, 0.], [a21, a22]])


def vb_topology(route, q, p=None, min_arm=1e-3, min_length=2e-3):
    """Topology/validity flags for V-B: a segment must not collapse or pass through its joint axis."""
    p = p or FingerParams()
    lengths, J = vb_analytic(route, q, p)
    s = route.sites
    gj = np.array(s["t2_guide"]) - np.array([p.L1, 0.])
    seg = [_arm(np.array(s["t1_base"]), s["t1_link1"], q[0])[1], _arm(np.array(s["t2_base"]), s["t2_guide"], q[0])[1],
           _arm(gj, s["t2_link2"], q[1])[1]]
    arms = -np.array([J[0, 0], J[1, 0], J[1, 1]])
    return dict(valid=bool(np.all(arms > min_arm) and min(seg) > min_length), arms=arms, segment_lengths=np.array(seg))


def va_route():
    return VirtualRoute("V-A")


def virtual_mjcf(route, p):
    """MJCF for the finger with two spatial tendons following ``route`` and two unit-gain actuators."""
    root = ET.fromstring(build_mjcf(p))
    for tag in ("tendon", "actuator", "sensor"):
        root.remove(root.find(tag))
    world = root.find("worldbody")
    link1 = root.find(".//body[@name='link1']")
    link2 = root.find(".//body[@name='link2']")
    def site(parent, name, xz):
        ET.SubElement(parent, "site", name=name, pos=f"{float(xz[0])!r} 0 {float(xz[1])!r}", size=".0008")
    tendons = ET.SubElement(root, "tendon")
    if route.kind == "V-A":
        r1, r21, r2 = route.radii
        a1, a21, a2 = route.termination_angle_rad
        def cylinder(parent, name, r):
            ET.SubElement(parent, "geom", name=name, type="cylinder", size=f"{r!r} .002", euler="1.5707963267948966 0 0",
                          contype="0", conaffinity="0", density="0", rgba=".7 .7 .7 .3")
        base = world.find("body[@name='base']")
        site(base, "t1_base", (-.04, -r1)); site(base, "t2_base", (-.04, -r21))
        cylinder(link1, "pulley1", r1); cylinder(link1, "pulley21", r21); cylinder(link2, "pulley2", r2)
        side = -.6
        site(link1, "side1", (1.5*r1*np.cos(side), 1.5*r1*np.sin(side)))
        site(link1, "side21", (1.5*r21*np.cos(side), 1.5*r21*np.sin(side)))
        site(link2, "side2", (1.5*r2*np.cos(side), 1.5*r2*np.sin(side)))
        site(link1, "t1_term", (2*r1*np.cos(a1), 2*r1*np.sin(a1)))
        site(link1, "t2_guide_a", (2*r21*np.cos(a21), 2*r21*np.sin(a21)))
        site(link1, "t2_guide_b", (p.L1 - .03, -r2))
        site(link2, "t2_term", (2*r2*np.cos(a2), 2*r2*np.sin(a2)))
        t1 = ET.SubElement(tendons, "spatial", name="flexor1", width=".0006", stiffness="0", damping="0", frictionloss="0")
        ET.SubElement(t1, "site", site="t1_base"); ET.SubElement(t1, "geom", geom="pulley1", sidesite="side1")
        ET.SubElement(t1, "site", site="t1_term")
        t2 = ET.SubElement(tendons, "spatial", name="flexor2", width=".0006", stiffness="0", damping="0", frictionloss="0")
        for item in (("site", "t2_base"), ("geom", "pulley21", "side21"), ("site", "t2_guide_a"), ("site", "t2_guide_b"),
                     ("geom", "pulley2", "side2"), ("site", "t2_term")):
            if item[0] == "site":
                ET.SubElement(t2, "site", site=item[1])
            else:
                ET.SubElement(t2, "geom", geom=item[1], sidesite=item[2])
    elif route.kind == "V-B":
        s = route.sites
        base = world.find("body[@name='base']")
        site(base, "t1_base", s["t1_base"]); site(base, "t2_base", s["t2_base"])
        site(link1, "t1_link1", s["t1_link1"]); site(link1, "t2_guide", s["t2_guide"]); site(link2, "t2_link2", s["t2_link2"])
        t1 = ET.SubElement(tendons, "spatial", name="flexor1", width=".0006", stiffness="0", damping="0", frictionloss="0")
        for n in ("t1_base", "t1_link1"):
            ET.SubElement(t1, "site", site=n)
        t2 = ET.SubElement(tendons, "spatial", name="flexor2", width=".0006", stiffness="0", damping="0", frictionloss="0")
        for n in ("t2_base", "t2_guide", "t2_link2"):
            ET.SubElement(t2, "site", site=n)
    else:
        raise ValueError("virtual route must be V-A or V-B")
    actuators = ET.SubElement(root, "actuator")
    for i in (1, 2):
        ET.SubElement(actuators, "general", name=f"pull{i}", tendon=f"flexor{i}", gainprm="1 0 0", biasprm="0 0 0",
                      ctrlrange=f"-{p.T_max} 0")
    return ET.tostring(root, encoding="unicode")


class Phase4Finger(TwoTendonFinger):
    """The two-tendon finger with a spatial (V-A or V-B) tendon route.

    Bodies, joints, springs, contact and the object are those of the P0 finger. Only the tendon geometry
    differs. reset(), tendon_jacobian() and apply_tension() are inherited unchanged; the Jacobian is read
    from MuJoCo's actuator moments, which for a spatial tendon equal dl/dq at the current configuration.
    """
    def __init__(self, params=None, route=None):
        self.p = params or FingerParams(q1_range=(-pi/6, pi/3), q2_range=(0., 5*pi/6), T_max=80.)
        self.route = route or vb_route(self.p)
        self.model = mujoco.MjModel.from_xml_string(virtual_mjcf(self.route, self.p))
        self.data = mujoco.MjData(self.model)
        self.tids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, f"flexor{i}") for i in (1, 2)]
        self.aids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pull{i}") for i in (1, 2)]
        self.jq = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("q1", "q2")]
        self.dof = [self.model.jnt_dofadr[j] for j in self.jq]
        self.qadr = [self.model.jnt_qposadr[j] for j in self.jq]
        self.obj_j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_z")
        self.obj_qadr = self.model.jnt_qposadr[self.obj_j]
        self.obj_dof = self.model.jnt_dofadr[self.obj_j]
        self.obj_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.finger_geoms = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("link1_geom", "link2_geom")}
        self.tip_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self._force6 = np.zeros(6)
        self.length_offsets = np.zeros(2)
        self.reset()

    def lengths_and_jacobian(self, q):
        d = mujoco.MjData(self.model)
        d.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, d)
        J = np.zeros((2, self.model.nv))
        for i, aid in enumerate(self.aids):
            start = d.moment_rowadr[aid]
            for k in range(d.moment_rownnz[aid]):
                J[i, d.moment_colind[start + k]] = d.actuator_moment[start + k]
        return d.ten_length[self.tids].copy(), J[:, self.dof]

    def required_tension(self, q):
        d = mujoco.MjData(self.model)
        d.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, d)
        required = (d.qfrc_bias - d.qfrc_passive)[self.dof]
        return np.linalg.solve(-self.lengths_and_jacobian(q)[1].T, required)


def joint_grid(n1=11, n2=13, p=None):
    p = p or FingerParams(q1_range=(-pi/6, pi/3), q2_range=(0., 5*pi/6))
    return [np.array([a, b]) for a in np.linspace(*p.q1_range, n1) for b in np.linspace(*p.q2_range, n2)]
