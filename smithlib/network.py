"""A cascadable one-port network: a load plus a chain of elements.

Elements are stored **load-side first**, the same order you build a matching
network on the bench, and each one knows how to transform an impedance at a
given frequency.  That gives two things the single-frequency solvers cannot:

* a *trajectory* -- the impedance after every element, which is exactly the
  Smith chart path;
* a *sweep* -- the input impedance across a band, which is what decides whether
  a match is actually usable.

Nothing here depends on matplotlib or Streamlit, so it is equally at home in a
measurement script.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tline import C0

__all__ = [
    "Element", "SeriesL", "SeriesC", "SeriesR", "ShuntL", "ShuntC", "ShuntR",
    "Line", "Stub", "Network",
    "Load", "FixedLoad", "SeriesRLC", "ParallelRLC",
]


# --------------------------------------------------------------------------
# loads
# --------------------------------------------------------------------------

class Load:
    """Base class: anything that can report an impedance at a frequency."""

    def Z(self, f):                                    # pragma: no cover
        raise NotImplementedError

    def describe(self):                                # pragma: no cover
        return type(self).__name__


@dataclass
class FixedLoad(Load):
    """A frequency-independent impedance. Fine for a single-point design."""

    Z_ohm: complex = 50 + 0j

    def Z(self, f):
        return np.full_like(np.asarray(f, dtype=float), self.Z_ohm,
                            dtype=complex) if np.ndim(f) else complex(self.Z_ohm)

    def describe(self):
        z = complex(self.Z_ohm)
        return f"fixed {z.real:g} {z.imag:+g}j ohm"


@dataclass
class SeriesRLC(Load):
    """R, L and C in series -- the usual model for a resonant antenna."""

    R: float = 35.0
    L: float = 8e-9
    C: float = 3.17e-12

    def Z(self, f):
        w = 2 * np.pi * np.asarray(f, dtype=float)
        return self.R + 1j * (w * self.L - 1.0 / (w * self.C))

    def describe(self):
        return f"series RLC ({self.R:g} ohm, {self.L*1e9:g} nH, {self.C*1e12:g} pF)"


@dataclass
class ParallelRLC(Load):
    """R, L and C in parallel -- a tank, or a patch near resonance."""

    R: float = 200.0
    L: float = 8e-9
    C: float = 3.17e-12

    def Z(self, f):
        w = 2 * np.pi * np.asarray(f, dtype=float)
        Y = 1.0 / self.R + 1j * (w * self.C - 1.0 / (w * self.L))
        return 1.0 / Y

    def describe(self):
        return f"parallel RLC ({self.R:g} ohm, {self.L*1e9:g} nH, {self.C*1e12:g} pF)"


# --------------------------------------------------------------------------
# elements
# --------------------------------------------------------------------------

class Element:
    """One thing added between the load and the source."""

    #: which Smith chart arc this element travels along
    arc = "resistance"

    def apply(self, Z, f):                             # pragma: no cover
        raise NotImplementedError

    def label(self):                                   # pragma: no cover
        return type(self).__name__


def _series(Z, Zel):
    return Z + Zel


def _shunt(Z, Zel):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / (1.0 / Z + 1.0 / Zel)


@dataclass
class SeriesL(Element):
    L: float = 1e-9
    arc: str = field(default="resistance", init=False, repr=False)

    def apply(self, Z, f):
        return _series(Z, 1j * 2 * np.pi * np.asarray(f) * self.L)

    def label(self):
        return f"series L = {self.L * 1e9:.3g} nH"


@dataclass
class SeriesC(Element):
    C: float = 1e-12
    arc: str = field(default="resistance", init=False, repr=False)

    def apply(self, Z, f):
        return _series(Z, 1.0 / (1j * 2 * np.pi * np.asarray(f) * self.C))

    def label(self):
        return f"series C = {self.C * 1e12:.3g} pF"


@dataclass
class SeriesR(Element):
    R: float = 10.0
    arc: str = field(default="resistance", init=False, repr=False)

    def apply(self, Z, f):
        return _series(Z, complex(self.R))

    def label(self):
        return f"series R = {self.R:.4g} ohm"


@dataclass
class ShuntL(Element):
    L: float = 1e-9
    arc: str = field(default="conductance", init=False, repr=False)

    def apply(self, Z, f):
        return _shunt(Z, 1j * 2 * np.pi * np.asarray(f) * self.L)

    def label(self):
        return f"shunt L = {self.L * 1e9:.3g} nH"


@dataclass
class ShuntC(Element):
    C: float = 1e-12
    arc: str = field(default="conductance", init=False, repr=False)

    def apply(self, Z, f):
        return _shunt(Z, 1.0 / (1j * 2 * np.pi * np.asarray(f) * self.C))

    def label(self):
        return f"shunt C = {self.C * 1e12:.3g} pF"


@dataclass
class ShuntR(Element):
    R: float = 100.0
    arc: str = field(default="conductance", init=False, repr=False)

    def apply(self, Z, f):
        return _shunt(Z, complex(self.R))

    def label(self):
        return f"shunt R = {self.R:.4g} ohm"


@dataclass
class Line(Element):
    """A length of transmission line, specified physically so it sweeps right.

    ``alpha_db_per_m`` is optional loss; ``eps_eff`` sets the propagation
    velocity and therefore how fast the line rotates with frequency.
    """

    length_m: float = 0.01
    z0: float = 50.0
    eps_eff: float = 1.0
    alpha_db_per_m: float = 0.0
    arc: str = field(default="gamma", init=False, repr=False)

    def _gamma_prop(self, f):
        beta = 2 * np.pi * np.asarray(f, dtype=float) * np.sqrt(self.eps_eff) / C0
        alpha = self.alpha_db_per_m / 8.685889638  # dB/m -> Np/m
        return alpha + 1j * beta

    def apply(self, Z, f):
        gl = self._gamma_prop(f) * self.length_m
        t = np.tanh(gl)
        return self.z0 * (Z + self.z0 * t) / (self.z0 + Z * t)

    def label(self):
        return f"line {self.length_m * 1000:.4g} mm, {self.z0:.0f} ohm"


@dataclass
class Stub(Element):
    """An open- or short-circuit stub, in shunt (usual) or in series."""

    length_m: float = 0.01
    z0: float = 50.0
    eps_eff: float = 1.0
    kind: str = "open"           # "open" | "short"
    orientation: str = "shunt"   # "shunt" | "series"

    @property
    def arc(self):
        return "conductance" if self.orientation == "shunt" else "resistance"

    def Z_stub(self, f):
        beta = 2 * np.pi * np.asarray(f, dtype=float) * np.sqrt(self.eps_eff) / C0
        bl = beta * self.length_m
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.kind == "open":
                return -1j * self.z0 / np.tan(bl)
            return 1j * self.z0 * np.tan(bl)

    def apply(self, Z, f):
        Zs = self.Z_stub(f)
        return _shunt(Z, Zs) if self.orientation == "shunt" else _series(Z, Zs)

    def label(self):
        return (f"{self.orientation} {self.kind} stub "
                f"{self.length_m * 1000:.4g} mm, {self.z0:.0f} ohm")


# --------------------------------------------------------------------------
# the network
# --------------------------------------------------------------------------

@dataclass
class Network:
    """A load plus an ordered chain of elements, load-side first."""

    load: Load = field(default_factory=FixedLoad)
    elements: list = field(default_factory=list)
    z0: float = 50.0

    def add(self, element):
        self.elements.append(element)
        return self

    def Zin(self, f):
        """Input impedance looking in from the source end."""
        Z = self.load.Z(f)
        for el in self.elements:
            Z = el.apply(Z, f)
        return Z

    def trajectory(self, f):
        """[(label, Z)] after the load and after each element, at one frequency.

        This is precisely the sequence of points to draw on the Smith chart.
        """
        Z = complex(self.load.Z(f))
        out = [("load", Z)]
        for el in self.elements:
            Z = complex(el.apply(Z, f))
            out.append((el.label(), Z))
        return out

    def sweep(self, freqs):
        """Vectorised input impedance over an array of frequencies."""
        return np.asarray(self.Zin(np.asarray(freqs, dtype=float)),
                          dtype=complex)
