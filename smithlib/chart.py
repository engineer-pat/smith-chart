"""Matplotlib Smith chart with annotation helpers built for teaching.

The point of this class is that you never draw in gamma coordinates by hand.
You hand it impedances and say what happened between them -- a series element,
a shunt element, a length of line -- and it draws the correct arc with an
arrowhead showing which way the design walked.

    sc = SmithChart(title="L-network")
    sc.grid()
    sc.point(0.4 - 0.5j, "load", color=style.C_LOAD)
    sc.move(0.4 - 0.5j, z_mid, kind="conductance", label="shunt C")
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import FancyArrowPatch

from . import geometry as geo
from . import style as S
from .core import gamma_from_z, vswr, z_from_gamma

__all__ = ["SmithChart"]


def _xy(g):
    g = np.asarray(g, dtype=complex)
    return g.real, g.imag


def _radial_offset(g, dist):
    """A displacement pointing away from the chart centre, for label placement."""
    g = complex(g)
    # At the exact centre there is no outward direction; go up, which is
    # where matching paths least often are.
    u = g / abs(g) if abs(g) > 1e-6 else 0.0 + 1.0j
    return (dist * u.real, dist * u.imag)


def _fmt_z(z, z0=None):
    """Format a normalized impedance, optionally also in ohms."""
    z = complex(z)
    s = f"{z.real:.2f} {'+' if z.imag >= 0 else '-'} j{abs(z.imag):.2f}"
    if z0:
        Z = z * z0
        s += f"\n({Z.real:.1f} {'+' if Z.imag >= 0 else '-'} j{abs(Z.imag):.1f} $\\Omega$)"
    return s


class SmithChart:
    """A matplotlib axes set up as a Smith chart, plus drawing helpers."""

    def __init__(self, ax=None, figsize=(7.2, 7.2), title=None, z0=50.0,
                 admittance=False):
        import matplotlib.pyplot as plt

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax, self.fig = ax, ax.figure
        self.z0 = z0
        self.admittance = admittance
        self._legend_handles = []

        ax = self.ax
        ax.set_aspect("equal")
        ax.set_xlim(-1.12, 1.12)
        ax.set_ylim(-1.12, 1.12)
        ax.axis("off")
        self.fig.patch.set_facecolor(S.SURFACE)
        ax.set_facecolor(S.SURFACE)
        if title:
            ax.set_title(title, color=S.INK, fontsize=13, pad=12)

    # ------------------------------------------------------------------
    # the grid
    # ------------------------------------------------------------------

    def grid(self, r_ticks=geo.R_TICKS, x_ticks=geo.X_TICKS, labels=True,
             minor=True):
        """Draw the impedance grid: constant-r circles and constant-x arcs."""
        ax = self.ax
        if minor:
            for r in (0.1, 0.3, 0.4, 0.6, 0.8, 1.5, 3.0, 4.0, 20.0):
                ax.plot(*_xy(geo.resistance_circle(r)), color=S.GRID_MINOR,
                        lw=S.LW_GRID, zorder=0)
            for x in (0.1, 0.3, 0.4, 0.6, 0.8, 1.5, 3.0, 4.0, 20.0):
                for s in (+1, -1):
                    ax.plot(*_xy(geo.reactance_arc(s * x)), color=S.GRID_MINOR,
                            lw=S.LW_GRID, zorder=0)

        for r in r_ticks:
            ax.plot(*_xy(geo.resistance_circle(r)), color=S.GRID_MAJOR,
                    lw=S.LW_GRID_MAJOR, zorder=1)
        for x in x_ticks:
            for s in (+1, -1):
                ax.plot(*_xy(geo.reactance_arc(s * x)), color=S.GRID_MAJOR,
                        lw=S.LW_GRID_MAJOR, zorder=1)

        # real axis (x = 0) and the rim (r = 0)
        ax.plot([-1, 1], [0, 0], color=S.GRID_AXIS, lw=S.LW_GRID_MAJOR, zorder=1)
        ax.plot(*_xy(geo.unit_circle()), color=S.RIM, lw=S.LW_RIM, zorder=2)

        if labels:
            self._label_grid(r_ticks, x_ticks)
        return self

    def _label_grid(self, r_ticks, x_ticks):
        ax = self.ax
        for r in r_ticks:
            if r == 0:
                continue
            g = gamma_from_z(r + 0j)
            ax.text(g.real, -0.028, f"{r:g}", color=S.INK_MUTED, fontsize=7.5,
                    ha="center", va="top", zorder=3,
                    bbox=dict(boxstyle="square,pad=0.12", fc=S.SURFACE,
                              ec="none", alpha=0.85))
        for x in x_ticks:
            for s in (+1, -1):
                # Put the label on the rim, where the constant-x arc lands,
                # nudged outward along its own radius so it never sits on a line.
                g = gamma_from_z(0 + 1j * s * x)
                u = g / abs(g)
                ax.text((g + 0.075 * u).real, (g + 0.075 * u).imag,
                        f"{s * x:g}j", color=S.INK_MUTED, fontsize=7.5,
                        ha="center", va="center", zorder=3)
        ax.text(-1.045, 0, "0", color=S.INK_MUTED, fontsize=8,
                ha="right", va="center")
        ax.text(1.045, 0, r"$\infty$", color=S.INK_MUTED, fontsize=9,
                ha="left", va="center")

    def admittance_grid(self, alpha=0.5):
        """Overlay the mirrored (Y) grid -- handy for shunt-element work."""
        ax = self.ax
        for g in (0.2, 0.5, 1.0, 2.0, 5.0):
            ax.plot(*_xy(geo.conductance_circle(g)), color=S.C_SHUNT_EL,
                    lw=S.LW_GRID, alpha=alpha, zorder=0)
        for b in (0.2, 0.5, 1.0, 2.0, 5.0):
            for s in (+1, -1):
                ax.plot(*_xy(geo.susceptance_arc(s * b)), color=S.C_SHUNT_EL,
                        lw=S.LW_GRID, alpha=alpha, zorder=0)
        return self

    # ------------------------------------------------------------------
    # overlays
    # ------------------------------------------------------------------

    def vswr_circle(self, s, color=None, label=None, ls="--", lw=1.3):
        """Draw a constant-VSWR circle; everything on it has the same |gamma|."""
        color = color or S.INK_MUTED
        self.ax.plot(*_xy(geo.vswr_circle(s)), color=color, ls=ls, lw=lw,
                     zorder=3, label=label)
        return self

    def q_contour(self, q, color=None, label=None, ls=":", lw=1.3):
        """Draw both halves of a constant-Q contour (bandwidth budget line)."""
        color = color or S.ACCENT
        for half in ("upper", "lower"):
            self.ax.plot(*_xy(geo.q_contour(q, half=half)), color=color, ls=ls,
                         lw=lw, zorder=3,
                         label=label if half == "upper" else None)
        return self

    def shade_gamma(self, mag, color=None, alpha=0.10, label=None):
        """Shade the disc |gamma| <= mag, e.g. a return-loss spec region."""
        from matplotlib.patches import Circle
        color = color or S.C_TARGET
        self.ax.add_patch(Circle((0, 0), mag, facecolor=color, alpha=alpha,
                                 edgecolor="none", zorder=0.5, label=label))
        return self

    # ------------------------------------------------------------------
    # points and paths
    # ------------------------------------------------------------------

    def point(self, z, label=None, color=None, marker="o", show_value=False,
              offset=None, fontsize=9):
        """Mark a normalized impedance, optionally annotating its value.

        ``offset`` is a (dx, dy) displacement for the label in chart units.
        Left as ``None`` it is pushed radially outward from the chart centre,
        which is where there is almost always free space.
        """
        color = color or S.C_LOAD
        g = gamma_from_z(z)
        self.ax.plot([g.real], [g.imag], marker=marker, ms=S.MARKER_SIZE,
                     color=color, mec=S.SURFACE, mew=1.6, zorder=6, ls="none")
        if label:
            txt = label
            if show_value:
                txt += "\n" + _fmt_z(z, self.z0)
            if offset is None:
                offset = _radial_offset(g, 0.22)
            tx, ty = g.real + offset[0], g.imag + offset[1]
            self.ax.annotate(
                txt, xy=(g.real, g.imag), xytext=(tx, ty),
                color=S.INK, fontsize=fontsize, ha="center", va="center",
                zorder=7,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                shrinkA=2, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.28", fc=S.SURFACE,
                          ec=color, lw=0.9, alpha=0.94),
            )
        return self

    def move(self, z_from, z_to, kind="resistance", color=None, label=None,
             lw=None, arrows=1, ls="-", zorder=5, annotate_at=0.5,
             label_offset=None):
        """Draw the arc from ``z_from`` to ``z_to`` with direction arrowheads.

        ``kind`` says what caused the motion, and therefore which arc to use:
        ``"resistance"`` (a series element), ``"conductance"`` (a shunt
        element) or ``"gamma"`` (a length of line, clockwise toward the
        generator).  See :func:`smithlib.geometry.arc_between`.
        """
        color = color or {"resistance": S.C_SERIES_EL,
                          "conductance": S.C_SHUNT_EL,
                          "gamma": S.C_LINE}.get(kind, S.SERIES[0])
        pts = geo.arc_between(z_from, z_to, kind=kind)
        self.ax.plot(*_xy(pts), color=color, lw=lw or S.LW_PATH, ls=ls,
                     zorder=zorder, solid_capstyle="round")
        self._arrowheads(pts, color, n=arrows, zorder=zorder + 0.1)
        if label:
            self._label_arc(pts, label, color, at=annotate_at,
                            offset=label_offset)
        return self

    def path(self, steps, z_start, labels=True, colors=None, offsets=None):
        """Draw a whole matching path from :mod:`smithlib.matching`.

        ``steps`` is the ``.steps`` list of an ``LMatch``/``StubMatch``/
        ``QuarterWave`` result; ``z_start`` is the normalized load.
        ``offsets`` optionally gives a per-step (dx, dy) label displacement
        when the automatic radial placement collides with something.
        """
        z = complex(z_start)
        for i, st in enumerate(steps):
            c = None if colors is None else colors[i % len(colors)]
            lab = None
            if labels:
                lab = st.label + (f"\n{st.detail}" if st.detail else "")
            off = None if offsets is None else offsets[i]
            self.move(z, st.z_after, kind=st.arc, color=c, label=lab,
                      label_offset=off)
            z = complex(st.z_after)
        return self

    def line_sweep(self, zl, d_wl, alpha_np_per_wl=0.0, color=None,
                   label=None, arrows=2):
        """Walk ``d_wl`` wavelengths toward the generator from load ``zl``."""
        from .tline import line_input_z
        color = color or S.C_LINE
        d = np.linspace(0.0, d_wl, 400)
        g = gamma_from_z(zl) * np.exp(-2 * alpha_np_per_wl * d) \
            * np.exp(-1j * 4 * np.pi * d)
        self.ax.plot(*_xy(g), color=color, lw=S.LW_PATH, zorder=5,
                     solid_capstyle="round")
        self._arrowheads(g, color, n=arrows, zorder=5.1)
        if label:
            self._label_arc(g, label, color, at=0.5)
        return z_from_gamma(g[-1])

    def curve(self, z, color=None, label=None, lw=None, ls="-", arrows=1,
              zorder=5, label_offset=None):
        """Plot an arbitrary sequence of normalized impedances as a path.

        Use this when the locus is not one of the three standard moves -- for
        instance the path *inside* a line section whose characteristic
        impedance differs from the chart's reference.
        """
        color = color or S.SERIES[0]
        g = gamma_from_z(np.asarray(z, dtype=complex))
        self.ax.plot(*_xy(g), color=color, lw=lw or S.LW_PATH, ls=ls,
                     zorder=zorder, solid_capstyle="round")
        self._arrowheads(g, color, n=arrows, zorder=zorder + 0.1)
        if label:
            self._label_arc(g, label, color, at=0.5, offset=label_offset)
        return self

    def frequency_sweep(self, z_of_f, freqs, color=None, label=None,
                        mark_every=None):
        """Plot an impedance-vs-frequency locus, the everyday use of the chart.

        ``z_of_f`` is a callable returning *normalized* impedance.
        """
        color = color or S.SERIES[0]
        z = np.array([complex(z_of_f(f)) for f in freqs])
        g = gamma_from_z(z)
        self.ax.plot(*_xy(g), color=color, lw=S.LW_PATH, zorder=5, label=label)
        self._arrowheads(g, color, n=2, zorder=5.1)
        if mark_every:
            for f, gi in zip(freqs[::mark_every], g[::mark_every]):
                self.ax.plot([gi.real], [gi.imag], "o", ms=4, color=color,
                             mec=S.SURFACE, mew=1.0, zorder=6)
                self.ax.annotate(f"{f / 1e9:.2f} GHz", xy=(gi.real, gi.imag),
                                 xytext=(6, 6), textcoords="offset points",
                                 fontsize=7, color=S.INK_SOFT, zorder=7)
        return self

    # ------------------------------------------------------------------
    # free-form annotation
    # ------------------------------------------------------------------

    def callout(self, text, z=None, xy=None, xytext=(0.0, 0.0), color=None,
                fontsize=9, ha="left", va="center", arrow=True):
        """Text with an optional leader arrow pointing at a chart location."""
        color = color or S.INK_SOFT
        if xy is None:
            g = gamma_from_z(z)
            xy = (g.real, g.imag)
        kw = dict(color=S.INK, fontsize=fontsize, ha=ha, va=va, zorder=8,
                  bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE,
                            ec=color, lw=0.9, alpha=0.94))
        if arrow:
            kw["arrowprops"] = dict(arrowstyle="->", color=color, lw=1.2,
                                    shrinkA=2, shrinkB=4,
                                    connectionstyle="arc3,rad=0.15")
        self.ax.annotate(text, xy=xy, xytext=xytext, **kw)
        return self

    def rotation_arrow(self, radius=1.075, start_deg=100, sweep_deg=-70,
                       text="toward generator", color=None):
        """A curved arrow outside the rim showing the direction of travel."""
        color = color or S.INK_SOFT
        t = np.radians(np.linspace(start_deg, start_deg + sweep_deg, 120))
        x, y = radius * np.cos(t), radius * np.sin(t)
        self.ax.plot(x, y, color=color, lw=1.2, zorder=4)
        self.ax.add_patch(FancyArrowPatch(
            (x[-2], y[-2]), (x[-1], y[-1]), arrowstyle="-|>",
            mutation_scale=14, color=color, lw=1.2, zorder=4))
        mid = len(t) // 2
        self.ax.text(x[mid] * 1.06, y[mid] * 1.06, text, color=color,
                     fontsize=8, ha="center", va="center",
                     rotation=np.degrees(t[mid]) - 90, rotation_mode="anchor")
        return self

    def wavelength_scale(self, radius=1.02, step=0.05):
        """Tick the rim in wavelengths-toward-generator, like a paper chart."""
        for w in np.arange(0.0, 0.5, step):
            ang = np.pi - 4 * np.pi * w      # 0 wl at the short-circuit point
            self.ax.plot([radius * np.cos(ang), 1.045 * np.cos(ang)],
                         [radius * np.sin(ang), 1.045 * np.sin(ang)],
                         color=S.INK_MUTED, lw=0.8, zorder=3)
        return self

    def legend(self, loc="upper left", **kw):
        self.ax.legend(loc=loc, frameon=True, fontsize=8.5,
                       facecolor=S.SURFACE, edgecolor=S.GRID_MAJOR, **kw)
        return self

    def readout(self, z, loc="lower left"):
        """A small text block with the derived numbers for an impedance."""
        g = gamma_from_z(z)
        Z = complex(z) * self.z0
        lines = [
            f"Z  = {Z.real:.2f} {'+' if Z.imag >= 0 else '-'} j{abs(Z.imag):.2f} $\\Omega$",
            f"z  = {_fmt_z(z)}",
            f"$\\Gamma$  = {abs(g):.3f} $\\angle$ {np.degrees(np.angle(g)):.1f}$\\degree$",
            f"VSWR = {vswr(g):.2f}",
            f"RL = {-20 * np.log10(max(abs(g), 1e-12)):.1f} dB",
        ]
        xy = (0.02, 0.02) if "lower" in loc else (0.02, 0.98)
        self.ax.text(xy[0], xy[1], "\n".join(lines), transform=self.ax.transAxes,
                     fontsize=8.5, color=S.INK, family="monospace",
                     va="bottom" if "lower" in loc else "top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.4", fc=S.SURFACE,
                               ec=S.GRID_MAJOR, lw=0.9, alpha=0.95), zorder=8)
        return self

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _arrowheads(self, pts, color, n=1, zorder=6):
        """Place ``n`` arrowheads along a sampled path to show direction."""
        pts = np.asarray(pts)
        if len(pts) < 3 or n < 1:
            return
        for frac in np.linspace(1, n, n) / (n + 1):
            i = max(1, min(len(pts) - 1, int(frac * (len(pts) - 1))))
            p0, p1 = pts[i - 1], pts[i]
            if abs(p1 - p0) < 1e-9:
                continue
            self.ax.add_patch(FancyArrowPatch(
                (p0.real, p0.imag), (p1.real, p1.imag), arrowstyle="-|>",
                mutation_scale=15, color=color, lw=S.LW_PATH, zorder=zorder))

    def _label_arc(self, pts, text, color, at=0.5, offset=None):
        pts = np.asarray(pts)
        i = int(at * (len(pts) - 1))
        p = pts[i]
        if offset is not None:
            off = complex(p.real + offset[0], p.imag + offset[1])
        else:
            # Push perpendicular to the arc, on whichever side faces outward,
            # so the label clears the curve it names instead of lying on it.
            j = min(i + 1, len(pts) - 1)
            k = max(i - 1, 0)
            tangent = pts[j] - pts[k]
            if abs(tangent) < 1e-12:
                tangent = 1 + 0j
            normal = 1j * tangent / abs(tangent)
            if (normal.real * p.real + normal.imag * p.imag) < 0:
                normal = -normal
            off = p + 0.24 * normal
        self.ax.annotate(
            text, xy=(p.real, p.imag), xytext=(off.real, off.imag),
            color=S.INK, fontsize=8.5, ha="center", va="center", zorder=7,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=1,
                            shrinkB=1),
            bbox=dict(boxstyle="round,pad=0.25", fc=S.SURFACE, ec=color,
                      lw=0.9, alpha=0.92),
        )
