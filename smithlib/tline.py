"""Transmission-line transformations -- the motion that makes the chart useful.

The single idea behind this module: moving along a uniform lossless line does
not change |gamma|, it only rotates gamma.  Moving *toward the generator* by a
physical distance d rotates gamma clockwise by 2*beta*d radians.  One full
revolution is half a wavelength, which is why chart rims are labelled
0 -> 0.5 wavelengths rather than 0 -> 1.
"""

from __future__ import annotations

import numpy as np

from .core import gamma_from_z, z_from_gamma

__all__ = [
    "rotate_gamma",
    "line_input_z",
    "line_input_Z",
    "electrical_length_deg",
    "stub_input_z",
    "quarter_wave_z0",
    "wavelength",
]

C0 = 299_792_458.0  # speed of light, m/s


def wavelength(freq_hz, eps_eff=1.0, vf=None):
    """Guided wavelength in metres.

    Give either a relative effective permittivity ``eps_eff`` or a velocity
    factor ``vf`` (= 1/sqrt(eps_eff)); ``vf`` wins if both are supplied.
    """
    if vf is None:
        vf = 1.0 / np.sqrt(eps_eff)
    return C0 * vf / np.asarray(freq_hz, dtype=float)


def electrical_length_deg(length_m, freq_hz, eps_eff=1.0, vf=None):
    """Electrical length beta*l in degrees for a physical length in metres."""
    lam = wavelength(freq_hz, eps_eff=eps_eff, vf=vf)
    return 360.0 * np.asarray(length_m, dtype=float) / lam


def rotate_gamma(g, length_wl, toward="generator", alpha_np_per_wl=0.0):
    """Move ``length_wl`` wavelengths along a line and return the new gamma.

    Parameters
    ----------
    g : complex
        Reflection coefficient at the starting reference plane.
    length_wl : float or array
        Distance travelled, in wavelengths.
    toward : {"generator", "load"}
        "generator" rotates clockwise (the usual direction when you stand at
        the source looking into a line); "load" rotates counter-clockwise.
    alpha_np_per_wl : float
        Line attenuation in nepers per wavelength.  Non-zero values spiral the
        locus inward toward the matched centre, which is the real behaviour of
        a lossy cable.  (1 dB/wavelength = 0.1151 Np/wavelength.)

    Notes
    -----
    gamma(d) = gamma_L * exp(-2 * alpha * d) * exp(-j * 2 * beta * d)
    """
    g = np.asarray(g, dtype=complex)
    L = np.asarray(length_wl, dtype=float)
    sign = -1.0 if toward == "generator" else +1.0
    phase = sign * 2.0 * (2.0 * np.pi) * L  # 2*beta*d with beta*d = 2*pi*L
    decay = np.exp(-2.0 * alpha_np_per_wl * np.abs(L))
    out = g * decay * np.exp(1j * phase)
    return out if out.ndim else complex(out)


def line_input_z(zl, length_wl, alpha_np_per_wl=0.0):
    """Normalized input impedance of a line of ``length_wl`` loaded by ``zl``.

    Computed via the reflection coefficient so that ``zl = inf`` (open) and
    ``zl = 0`` (short) both work without special-casing the tan() formula.
    """
    g = gamma_from_z(zl)
    return z_from_gamma(rotate_gamma(g, length_wl, "generator", alpha_np_per_wl))


def line_input_Z(ZL, length_wl, z0=50.0, alpha_np_per_wl=0.0):
    """Same as :func:`line_input_z` but in ohms, for a line of impedance ``z0``."""
    return z0 * line_input_z(np.asarray(ZL, dtype=complex) / z0,
                             length_wl, alpha_np_per_wl)


def stub_input_z(length_wl, kind="open", z0_ratio=1.0):
    """Normalized input impedance of an ideal open- or short-circuit stub.

    ``z0_ratio`` is the stub's characteristic impedance divided by the system
    Z0, for the common case where the stub is a different line width than the
    main line.

    An open stub starts at the right-hand edge of the chart (z = inf) and a
    shorted stub at the left (z = 0); both walk clockwise around the rim.
    """
    zl = np.inf if kind == "open" else 0.0
    return z0_ratio * line_input_z(zl, length_wl)


def quarter_wave_z0(ZL, Zin=50.0):
    """Characteristic impedance of a quarter-wave transformer.

    Z0_qw = sqrt(Zin * ZL).  Only strictly valid for a real ``ZL``; a complex
    load must first be rotated to a real point on the chart (see the docs).
    """
    return np.sqrt(np.asarray(Zin, dtype=complex) * np.asarray(ZL, dtype=complex))
