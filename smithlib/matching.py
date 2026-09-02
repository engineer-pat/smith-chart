"""Matching-network synthesis: L-networks, stubs, quarter-wave transformers.

Each solver returns a dataclass carrying both the *electrical* answer (the
reactances or stub lengths) and the *path* -- the ordered list of impedance
points the design walks through on the Smith chart.  The plotting layer draws
that path directly, so the picture and the numbers can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import gamma_from_z, z_from_gamma
from .tline import line_input_z, stub_input_z

__all__ = [
    "MatchStep",
    "LMatch",
    "StubMatch",
    "QuarterWave",
    "l_match",
    "single_stub_match",
    "quarter_wave_match",
    "reactance_to_component",
    "susceptance_to_component",
    "format_component",
]


# --------------------------------------------------------------------------
# component value helpers
# --------------------------------------------------------------------------

def reactance_to_component(X, freq_hz, z0=50.0, normalized=True):
    """Turn a *series* reactance into ('L', henries) or ('C', farads).

    X > 0 is inductive, X < 0 is capacitive.  Set ``normalized=False`` if X is
    already in ohms.
    """
    Xo = float(X) * (z0 if normalized else 1.0)
    w = 2.0 * np.pi * float(freq_hz)
    if abs(Xo) < 1e-12:
        return ("short", 0.0)
    if Xo > 0:
        return ("L", Xo / w)
    return ("C", -1.0 / (w * Xo))


def susceptance_to_component(B, freq_hz, z0=50.0, normalized=True):
    """Turn a *shunt* susceptance into ('C', farads) or ('L', henries).

    B > 0 is capacitive, B < 0 is inductive.  Set ``normalized=False`` if B is
    already in siemens.
    """
    Bs = float(B) / (z0 if normalized else 1.0)
    w = 2.0 * np.pi * float(freq_hz)
    if abs(Bs) < 1e-15:
        return ("open", 0.0)
    if Bs > 0:
        return ("C", Bs / w)
    return ("L", -1.0 / (w * Bs))


def format_component(kind, value):
    """Human-readable component value, e.g. ``('L', 3.9e-9) -> '3.90 nH'``."""
    if kind in ("short", "open"):
        return kind
    unit = "H" if kind == "L" else "F"
    for scale, prefix in ((1e-12, "p"), (1e-9, "n"), (1e-6, "u"), (1e-3, "m")):
        if abs(value) < scale * 1000:
            return f"{value / scale:.3g} {prefix}{unit}"
    return f"{value:.3g} {unit}"


# --------------------------------------------------------------------------
# result containers
# --------------------------------------------------------------------------

@dataclass
class MatchStep:
    """One element or line section in a matching network.

    ``z_after`` is the normalized impedance looking toward the load *after*
    this step, so ``[start] + [s.z_after for s in steps]`` is the chart path.
    """

    label: str                 # e.g. "shunt C" or "series line"
    detail: str = ""           # e.g. "2.34 pF" or "0.184 wl"
    z_after: complex = 0j
    kind: str = "element"      # "element" | "line" | "stub"
    arc: str = "resistance"    # "resistance" | "conductance" | "gamma"


@dataclass
class LMatch:
    """A two-element L-network solution."""

    topology: str              # "shunt-series" or "series-shunt"
    shunt_B: float             # normalized susceptance (0 if unused)
    series_X: float            # normalized reactance
    z_load: complex = 1 + 0j
    steps: list = field(default_factory=list)
    freq_hz: float | None = None
    z0: float = 50.0

    @property
    def path(self):
        """Normalized impedance points from load to source."""
        return [self.z_load] + [s.z_after for s in self.steps]

    @property
    def network_q(self):
        """Loaded Q of the *network*, which is what sets the bandwidth.

        This is the Q at the intermediate node -- the corner the path turns at
        between the two circles.  The load's own Q is deliberately excluded:
        it is the same for every candidate solution and would mask the
        difference between them.  Larger Q means a sharper corner and a
        narrower match.
        """
        mids = self.path[1:-1]
        if not mids:
            return 0.0
        return max(abs(z.imag) / z.real for z in mids if z.real > 0)

    def components(self):
        """[(role, kind, value, text)] for each element, load-side first."""
        out = []
        for s in self.steps:
            role = s.label
            if role.startswith("shunt"):
                k, v = susceptance_to_component(self.shunt_B, self.freq_hz, self.z0)
            else:
                k, v = reactance_to_component(self.series_X, self.freq_hz, self.z0)
            out.append((role, k, v, format_component(k, v)))
        return out


@dataclass
class StubMatch:
    """A single-stub tuner solution."""

    d_wl: float                # line length from load to stub, in wavelengths
    l_wl: float                # stub length, in wavelengths
    stub_kind: str             # "open" or "short"
    orientation: str           # "shunt" or "series"
    z_at_stub: complex = 1 + 0j
    z_load: complex = 1 + 0j
    steps: list = field(default_factory=list)

    @property
    def path(self):
        return [self.z_load] + [s.z_after for s in self.steps]


@dataclass
class QuarterWave:
    """A quarter-wave transformer solution."""

    z0_transformer: float      # in ohms
    d_wl: float                # distance from load to the transformer input
    R_real: float              # the real impedance the transformer sees, ohms
    z_load: complex = 1 + 0j
    steps: list = field(default_factory=list)

    @property
    def path(self):
        return [self.z_load] + [s.z_after for s in self.steps]


# --------------------------------------------------------------------------
# L-network
# --------------------------------------------------------------------------

def l_match(ZL, z0=50.0, freq_hz=None):
    """All valid L-network solutions matching ``ZL`` (ohms) to ``z0``.

    Returns a list of :class:`LMatch`, usually two (the "high-pass" and
    "low-pass" arrangements of the same topology) and occasionally four when
    the load sits where both topologies are reachable.

    The two topologies, written load-side first:

    * ``shunt-series`` -- shunt element across the load, then a series element
      toward the source.  Available when the load's conductance is small
      enough, which for a real load means R_L > z0.
    * ``series-shunt`` -- series element first, then shunt.  Available when
      R_L < z0.
    """
    zl = complex(ZL) / z0
    r, x = zl.real, zl.imag
    if r <= 0:
        raise ValueError("load must have positive resistance")

    sols: list[LMatch] = []

    # --- topology A: shunt across the load, then series ---------------------
    # 1/(y_L + jB) must have unit real part.
    yl = 1.0 / zl
    g, b = yl.real, yl.imag
    disc = g - g * g            # = g/1 - g^2 in normalized units
    if disc >= -1e-15:
        root = np.sqrt(max(disc, 0.0))
        for s in (+1.0, -1.0):
            B = -b + s * root
            z_mid = 1.0 / (yl + 1j * B)
            X = -z_mid.imag                       # series element cancels it
            z_end = z_mid + 1j * X
            if abs(z_end - 1.0) > 1e-6:
                continue
            sols.append(LMatch(
                topology="shunt-series", shunt_B=B, series_X=X,
                z_load=zl, freq_hz=freq_hz, z0=z0,
                steps=[
                    MatchStep(f"shunt {'C' if B > 0 else 'L'}",
                              z_after=z_mid, arc="conductance"),
                    MatchStep(f"series {'L' if X > 0 else 'C'}",
                              z_after=z_end, arc="resistance"),
                ],
            ))
            if root == 0.0:
                break

    # --- topology B: series with the load, then shunt ------------------------
    # 1/(z_L + jX) must have unit real part.
    disc = r - r * r
    if disc >= -1e-15:
        root = np.sqrt(max(disc, 0.0))
        for s in (+1.0, -1.0):
            X = -x + s * root
            z_mid = zl + 1j * X
            y_mid = 1.0 / z_mid
            B = -y_mid.imag                       # shunt element cancels it
            z_end = 1.0 / (y_mid + 1j * B)
            if abs(z_end - 1.0) > 1e-6:
                continue
            sols.append(LMatch(
                topology="series-shunt", shunt_B=B, series_X=X,
                z_load=zl, freq_hz=freq_hz, z0=z0,
                steps=[
                    MatchStep(f"series {'L' if X > 0 else 'C'}",
                              z_after=z_mid, arc="resistance"),
                    MatchStep(f"shunt {'C' if B > 0 else 'L'}",
                              z_after=z_end, arc="conductance"),
                ],
            ))
            if root == 0.0:
                break

    if not sols:
        raise ValueError(f"no L-network matches ZL={ZL} to {z0} ohm")
    return sols


# --------------------------------------------------------------------------
# single-stub tuner
# --------------------------------------------------------------------------

def single_stub_match(ZL, z0=50.0, stub_kind="open", orientation="shunt"):
    """Both single-stub solutions for matching ``ZL`` (ohms) to ``z0``.

    Walk down the line from the load until the *normalized admittance* is
    1 + jb (for a shunt stub) or the impedance is 1 + jx (for a series stub),
    then add a stub whose susceptance/reactance is the negative of that.

    Returns a list of two :class:`StubMatch`, sorted by ``d_wl``.
    """
    zl = complex(ZL) / z0
    sols: list[StubMatch] = []

    # An already-matched load needs no stub.  Without this the root finder
    # returns two indistinguishable near-zero "solutions".
    if abs(gamma_from_z(zl)) < 1e-12:
        return [StubMatch(d_wl=0.0, l_wl=0.0, stub_kind=stub_kind,
                          orientation=orientation, z_at_stub=zl, z_load=zl,
                          steps=[])]

    for d in _stub_distances(zl, orientation):
        z_d = line_input_z(zl, d)
        if orientation == "shunt":
            b = (1.0 / z_d).imag                  # need stub to supply -jb
            l = _stub_length_for_susceptance(-b, stub_kind)
            y_end = 1.0 / z_d + 1j * (-b)
            z_end = 1.0 / y_end
            arc = "conductance"
        else:
            xr = z_d.imag                          # need stub to supply -jx
            l = _stub_length_for_reactance(-xr, stub_kind)
            z_end = z_d + 1j * (-xr)
            arc = "resistance"
        sols.append(StubMatch(
            d_wl=d, l_wl=l, stub_kind=stub_kind, orientation=orientation,
            z_at_stub=z_d, z_load=zl,
            steps=[
                MatchStep("series line", f"{d:.4f} wl",
                          z_after=z_d, kind="line", arc="gamma"),
                MatchStep(f"{orientation} {stub_kind} stub", f"{l:.4f} wl",
                          z_after=z_end, kind="stub", arc=arc),
            ],
        ))
    return sorted(sols, key=lambda s: s.d_wl)


def _stub_distances(zl, orientation):
    """Distances (in wavelengths) where the line reaches the unit g or r circle."""
    # Solve numerically -- robust for every load including r == 1 and pure
    # reactances, where the closed-form expression degenerates.
    n = 20001
    d = np.linspace(0.0, 0.5, n)
    step = 0.5 / (n - 1)
    z = line_input_z(zl, d)
    target = (1.0 / z).real if orientation == "shunt" else z.real
    f = target - 1.0
    roots = []
    for i in range(len(d) - 1):
        if f[i] == 0.0:
            roots.append(d[i])
        elif np.isfinite(f[i]) and np.isfinite(f[i + 1]) and f[i] * f[i + 1] < 0:
            roots.append(_bisect(zl, orientation, d[i], d[i + 1]))
    # De-duplicate roots the sampling grid bracketed twice.  The tolerance must
    # exceed the grid step, or two brackets straddling one root survive as two.
    out = []
    for rt in roots:
        if all(abs(rt - o) > 4 * step for o in out):
            out.append(rt)
    return out[:2]


def _bisect(zl, orientation, lo, hi, tol=1e-12, iters=200):
    def f(d):
        z = line_input_z(zl, d)
        v = (1.0 / z).real if orientation == "shunt" else z.real
        return v - 1.0

    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if hi - lo < tol:
            break
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _stub_length_for_susceptance(b, kind):
    """Shortest stub length (wavelengths) presenting normalized susceptance b."""
    if kind == "open":
        # y_stub = j tan(beta l)
        l = np.arctan(b) / (2.0 * np.pi)
    else:
        # y_stub = -j cot(beta l)
        l = np.arctan2(-1.0, b) / (2.0 * np.pi)
    return l % 0.5


def _stub_length_for_reactance(x, kind):
    """Shortest stub length (wavelengths) presenting normalized reactance x."""
    if kind == "short":
        # z_stub = j tan(beta l)
        l = np.arctan(x) / (2.0 * np.pi)
    else:
        # z_stub = -j cot(beta l)
        l = np.arctan2(-1.0, x) / (2.0 * np.pi)
    return l % 0.5


# --------------------------------------------------------------------------
# quarter-wave transformer
# --------------------------------------------------------------------------

def quarter_wave_match(ZL, z0=50.0, branch=0):
    """Quarter-wave transformer for a possibly complex ``ZL``.

    A quarter-wave section only transforms *real* impedances, so a complex load
    is first walked down the line to the nearest point where the impedance is
    purely real -- that is, to the real axis of the chart.  There are two such
    points per half wavelength (``branch=0`` is the nearer one).
    """
    zl = complex(ZL) / z0
    d = _real_axis_distances(zl)[branch % max(len(_real_axis_distances(zl)), 1)]
    z_real = line_input_z(zl, d)
    R = z_real.real * z0
    z0_qw = float(np.sqrt(R * z0))
    z_end = 1.0 + 0j
    return QuarterWave(
        z0_transformer=z0_qw, d_wl=d, R_real=R, z_load=zl,
        steps=[
            MatchStep("series line", f"{d:.4f} wl",
                      z_after=z_real, kind="line", arc="gamma"),
            MatchStep("lambda/4 transformer", f"{z0_qw:.2f} ohm",
                      z_after=z_end, kind="line", arc="gamma"),
        ],
    )


def _real_axis_distances(zl):
    """Distances (wavelengths) at which the line impedance becomes real."""
    g = gamma_from_z(zl)
    if abs(g) < 1e-12:
        return [0.0]
    # gamma is real when its angle reaches 0 (Vmax) or pi (Vmin).
    ang = np.angle(g)
    d_max = (ang % (2 * np.pi)) / (4 * np.pi)            # rotate to angle 0
    d_min = ((ang - np.pi) % (2 * np.pi)) / (4 * np.pi)  # rotate to angle pi
    return sorted({round(d_max, 12), round(d_min, 12)})
