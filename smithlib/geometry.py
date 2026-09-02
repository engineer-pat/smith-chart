"""Pure geometry of the Smith chart, independent of any plotting library.

Two facts generate the whole chart:

* A constant-resistance line (z = r + jx, x free) maps to a circle centred at
  ``r/(1+r)`` on the real gamma axis with radius ``1/(1+r)``.  Every one of
  them passes through gamma = +1.
* A constant-reactance line (z = r + jx, r free) maps to a circle centred at
  ``1 + j/x`` with radius ``1/|x|``, of which only the arc inside the unit
  disc is drawn.

The admittance (Y) chart is the same picture rotated 180 degrees, which is why
a shunt element moves you along a mirrored circle.
"""

from __future__ import annotations

import numpy as np

from .core import gamma_from_z

__all__ = [
    "R_TICKS",
    "X_TICKS",
    "resistance_circle",
    "reactance_arc",
    "conductance_circle",
    "susceptance_arc",
    "vswr_circle",
    "q_contour",
    "arc_between",
    "unit_circle",
]

R_TICKS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
X_TICKS = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def unit_circle(n=721):
    t = np.linspace(0, 2 * np.pi, n)
    return np.exp(1j * t)


def _clip_unit(g, tol=1e-9):
    """Keep only the samples that lie inside the unit disc."""
    return g[np.abs(g) <= 1.0 + tol]


def resistance_circle(r, n=721):
    """Full constant-r circle in the gamma plane."""
    c = r / (1.0 + r)
    rad = 1.0 / (1.0 + r)
    t = np.linspace(0, 2 * np.pi, n)
    return c + rad * np.exp(1j * t)


def reactance_arc(x, n=2001, rmax=1e4):
    """The visible arc of a constant-x circle (the part inside |gamma| <= 1)."""
    r = np.concatenate([[0.0], np.geomspace(1e-4, rmax, n - 1)])
    return _clip_unit(gamma_from_z(r + 1j * x))


def conductance_circle(g_norm, n=721):
    """Constant-conductance circle: the resistance circle mirrored through 0."""
    return -resistance_circle(g_norm, n)


def susceptance_arc(b, n=2001, gmax=1e4):
    """Constant-susceptance arc: the reactance arc mirrored through 0."""
    return -reactance_arc(b, n, gmax)


def vswr_circle(s, n=721):
    """Constant-VSWR (constant |gamma|) circle for VSWR ``s``."""
    mag = (s - 1.0) / (s + 1.0)
    return mag * unit_circle(n)


def q_contour(q, n=2001, half="upper"):
    """Constant-Q contour, the locus of |x|/r = q.

    Drawn as an arc through gamma = -1 and +1; matching networks are kept
    inside a chosen Q contour to guarantee a minimum bandwidth.
    """
    r = np.concatenate([[0.0], np.geomspace(1e-4, 1e4, n - 1)])
    sign = 1.0 if half == "upper" else -1.0
    return _clip_unit(gamma_from_z(r + 1j * sign * q * r))


def arc_between(z1, z2, kind="resistance", n=301, spiral=True):
    """Sample the physical path between two normalized impedances.

    ``kind`` selects which quantity is held constant, i.e. which component was
    added:

    * ``"resistance"`` -- a *series* reactance: r fixed, x sweeps z1 -> z2.
    * ``"conductance"`` -- a *shunt* susceptance: g fixed, b sweeps.
    * ``"gamma"`` -- a length of *transmission line*: rotate clockwise about
      the origin (spiralling in if the magnitudes differ, i.e. a lossy line).

    Returns an array of gamma values from z1 to z2, suitable for plotting and
    for placing directional arrowheads.
    """
    z1, z2 = complex(z1), complex(z2)
    t = np.linspace(0.0, 1.0, n)

    # The held-constant quantity is taken from z1, not averaged with z2, so the
    # arc is guaranteed to start exactly where the caller said it does.  For a
    # genuine series/shunt move the two agree anyway.
    if kind == "resistance":
        x = z1.imag + t * (z2.imag - z1.imag)
        return gamma_from_z(z1.real + 1j * x)

    if kind == "conductance":
        y1, y2 = 1.0 / z1, 1.0 / z2
        b = y1.imag + t * (y2.imag - y1.imag)
        return gamma_from_z(1.0 / (y1.real + 1j * b))

    if kind == "gamma":
        g1, g2 = gamma_from_z(z1), gamma_from_z(z2)
        a1, a2 = np.angle(g1), np.angle(g2)
        # Always take the clockwise route: motion toward the generator.
        sweep = (a1 - a2) % (2 * np.pi)
        ang = a1 - t * sweep
        m1, m2 = abs(g1), abs(g2)
        mag = m1 + t * (m2 - m1) if spiral else np.full_like(t, m1)
        return mag * np.exp(1j * ang)

    raise ValueError(f"unknown arc kind {kind!r}")
