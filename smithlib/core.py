"""Fundamental reflection-coefficient and impedance relationships.

Everything here is plain numpy and has no plotting or UI dependency, so it can
be imported from scripts, notebooks, the Quarto docs, or the Streamlit app.

Conventions used throughout the package
---------------------------------------
* ``z``  -- *normalized* impedance, z = Z / Z0 (dimensionless, complex)
* ``Z``  -- actual impedance in ohms
* ``g``/``gamma`` -- voltage reflection coefficient, complex, |gamma| <= 1 for
  passive loads
* Angles are in degrees at the public API boundary, radians internally.
* "Toward the generator" is the clockwise direction on the Smith chart.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gamma_from_z",
    "z_from_gamma",
    "gamma_from_Z",
    "Z_from_gamma",
    "vswr",
    "return_loss_db",
    "mismatch_loss_db",
    "reflected_power_fraction",
    "q_of_z",
    "y_from_z",
    "z_from_y",
]


def gamma_from_z(z):
    """Reflection coefficient from a *normalized* impedance.

    gamma = (z - 1) / (z + 1)

    An open circuit (z -> inf) maps to gamma = +1, a short (z = 0) to
    gamma = -1, and a matched load (z = 1) to the origin.
    """
    z = np.asarray(z, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (z - 1.0) / (z + 1.0)
    # z = inf is a legitimate input (open circuit); it should give gamma = 1.
    g = np.where(np.isinf(z), 1.0 + 0.0j, g)
    return g if g.ndim else complex(g)


def z_from_gamma(g):
    """Normalized impedance from a reflection coefficient.

    z = (1 + gamma) / (1 - gamma)

    gamma = 1 returns ``inf`` rather than raising.
    """
    g = np.asarray(g, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (1.0 + g) / (1.0 - g)
    return z if z.ndim else complex(z)


def gamma_from_Z(Z, z0=50.0):
    """Reflection coefficient from an impedance in ohms, referenced to ``z0``."""
    Z = np.asarray(Z, dtype=complex)
    return gamma_from_z(Z / z0)


def Z_from_gamma(g, z0=50.0):
    """Impedance in ohms from a reflection coefficient referenced to ``z0``."""
    return z_from_gamma(g) * z0


def y_from_z(z):
    """Normalized admittance from normalized impedance (y = 1/z).

    An open circuit (z = inf) gives y = 0 rather than NaN, which plain complex
    division would produce.
    """
    z = np.asarray(z, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 1.0 / z
    y = np.where(np.isinf(z), 0.0 + 0.0j, y)
    y = np.where(z == 0, np.inf + 0.0j, y)
    return y if y.ndim else complex(y)


# The transform is its own inverse; alias for readability at call sites.
z_from_y = y_from_z


def vswr(g):
    """Voltage standing wave ratio from a reflection coefficient.

    VSWR = (1 + |gamma|) / (1 - |gamma|), returning ``inf`` at |gamma| = 1.
    """
    m = np.abs(np.asarray(g, dtype=complex))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (1.0 + m) / (1.0 - m)
    s = np.where(m >= 1.0, np.inf, s)
    return s if s.ndim else float(s)


def return_loss_db(g):
    """Return loss in dB (a positive number for a passive load).

    RL = -20 log10 |gamma|.  A perfect match gives ``inf``.
    """
    m = np.abs(np.asarray(g, dtype=complex))
    with np.errstate(divide="ignore"):
        rl = -20.0 * np.log10(m)
    return rl if rl.ndim else float(rl)


def mismatch_loss_db(g):
    """Power lost to reflection, in dB: -10 log10(1 - |gamma|^2)."""
    m2 = np.abs(np.asarray(g, dtype=complex)) ** 2
    with np.errstate(divide="ignore"):
        ml = -10.0 * np.log10(1.0 - m2)
    return ml if ml.ndim else float(ml)


def reflected_power_fraction(g):
    """Fraction of incident power reflected, |gamma|^2."""
    m = np.abs(np.asarray(g, dtype=complex))
    out = m**2
    return out if out.ndim else float(out)


def q_of_z(z):
    """Loaded Q of an impedance, |X| / R.

    Constant-Q contours are the classic overlay for judging matching-network
    bandwidth: a network that swings far from the origin in Q is narrowband.
    """
    z = np.asarray(z, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.abs(z.imag) / z.real
    return q if q.ndim else float(q)
