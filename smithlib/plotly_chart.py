"""Interactive Smith chart built on Plotly's native ``Scattersmith`` trace.

Plotly draws the grid itself, so this module supplies the parts it does not:
the physically-correct arcs between impedances, direction arrowheads, VSWR
circles, and hover text that reports Z, Gamma, VSWR and return loss at every
sampled point.

The matplotlib renderer in :mod:`smithlib.chart` is the one to use for print
and for the Quarto docs; this one is for the app.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import geometry as geo
from . import style as S
from .core import gamma_from_z, return_loss_db, vswr, z_from_gamma

__all__ = ["SmithFigure"]


def _hover(z, z0):
    """Per-point hover strings with the numbers an RF engineer wants."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    g = gamma_from_z(z)
    Z = z * z0
    s = vswr(g)
    rl = return_loss_db(g)
    return [
        (f"Z = {Zi.real:.2f} {'+' if Zi.imag >= 0 else '-'} j{abs(Zi.imag):.2f} Ω<br>"
         f"z = {zi.real:.3f} {'+' if zi.imag >= 0 else '-'} j{abs(zi.imag):.3f}<br>"
         f"Γ = {abs(gi):.3f} ∠ {np.degrees(np.angle(gi)):.1f}°<br>"
         f"VSWR = {si:.2f}   RL = {rli:.1f} dB")
        for zi, Zi, gi, si, rli in zip(z, Z, g, np.atleast_1d(s),
                                       np.atleast_1d(rl))
    ]


def _smith_xy(g):
    """Convert reflection coefficients to the coordinates Plotly expects.

    ``Scattersmith`` is fed **normalized impedance** (real = r, imag = x) and
    does the mapping to the disc itself -- it does *not* take gamma.  All of
    this package's geometry is computed in the gamma plane, so it is converted
    here, at the boundary.

    Samples sitting exactly on the rim would become infinite impedances, so
    |gamma| is nudged just inside 1 first; the plotted point lands on the rim
    either way.
    """
    g = np.atleast_1d(np.asarray(g, dtype=complex))
    mag = np.abs(g)
    scale = np.where(mag > 0.99995, 0.99995 / np.maximum(mag, 1e-30), 1.0)
    z = z_from_gamma(g * scale)
    return np.real(z), np.imag(z)


class SmithFigure:
    """A Plotly figure wrapping one Smith chart, with the same verbs as
    :class:`smithlib.chart.SmithChart`."""

    def __init__(self, z0=50.0, title=None, height=620, palette=None):
        """``palette`` takes a snapshot from :func:`smithlib.style.palette`.

        Holding the colours on the instance rather than reading module globals
        keeps two figures on different themes independent, which is what the
        app needs when several browser sessions are open at once.
        """
        self.z0 = z0
        self.P = palette or S.palette()
        self.fig = go.Figure()
        self.fig.update_layout(
            title=title,
            height=height,
            margin=dict(l=10, r=10, t=48 if title else 16, b=54),
            paper_bgcolor=self.P["SURFACE"],
            plot_bgcolor=self.P["SURFACE"],
            font=dict(color=self.P["INK"], size=12),
            legend=dict(orientation="h", yanchor="top", y=-0.02,
                        xanchor="center", x=0.5,
                        bgcolor=self.P["SURFACE"], font=dict(size=11)),
            smith=dict(
                bgcolor=self.P["SURFACE"],
                realaxis=dict(gridcolor=self.P["GRID_MINOR"], linecolor=self.P["GRID_AXIS"],
                              tickcolor=self.P["GRID_MAJOR"], tickfont=dict(size=9)),
                imaginaryaxis=dict(gridcolor=self.P["GRID_MINOR"],
                                   linecolor=self.P["GRID_AXIS"],
                                   tickcolor=self.P["GRID_MAJOR"],
                                   tickfont=dict(size=9)),
            ),
        )

    # ------------------------------------------------------------------

    def point(self, z, name, color=None, size=11, symbol="circle",
              showlegend=True):
        """Mark one normalized impedance."""
        rr, xx = _smith_xy(gamma_from_z(z))
        self.fig.add_trace(go.Scattersmith(
            real=rr, imag=xx, mode="markers", name=name,
            marker=dict(size=size, color=color or self.P["C_LOAD"], symbol=symbol,
                        line=dict(width=1.6, color=self.P["SURFACE"])),
            hovertext=_hover(z, self.z0), hoverinfo="text+name",
            showlegend=showlegend,
        ))
        return self

    def arc(self, z_from, z_to, kind="resistance", name=None, color=None,
            width=3.0, arrows=1, dash=None, showlegend=True):
        """Draw the physical arc between two impedances, with direction arrows."""
        color = color or {"resistance": self.P["C_SERIES_EL"],
                          "conductance": self.P["C_SHUNT_EL"],
                          "gamma": self.P["C_LINE"]}.get(kind, self.P["SERIES"][0])
        pts = geo.arc_between(z_from, z_to, kind=kind, n=201)
        rr, xx = _smith_xy(pts)
        self.fig.add_trace(go.Scattersmith(
            real=rr, imag=xx, mode="lines",
            name=name or kind, line=dict(color=color, width=width, dash=dash),
            hovertext=_hover(z_from_gamma_safe(pts), self.z0),
            hoverinfo="text+name", showlegend=showlegend,
        ))
        self._arrows(pts, color, arrows)
        return self

    def _arrows(self, pts, color, n):
        """Direction arrowheads, drawn as auto-rotating arrow markers."""
        if n < 1 or len(pts) < 3:
            return
        idx = [max(1, min(len(pts) - 1, int(f * (len(pts) - 1))))
               for f in np.linspace(1, n, n) / (n + 1)]
        for i in idx:
            rr, xx = _smith_xy(pts[i - 1:i + 1])
            self.fig.add_trace(go.Scattersmith(
                real=rr, imag=xx, mode="markers",
                marker=dict(symbol="arrow", size=15, color=color,
                            angleref="previous"),
                hoverinfo="skip", showlegend=False,
            ))

    def path(self, points, labels=None, arcs=None, colors=None):
        """Draw a whole trajectory of normalized impedances in order.

        ``points`` is the list of impedances (load first); ``arcs`` says which
        arc joins each consecutive pair.
        """
        pts = [complex(p) for p in points]
        for i in range(len(pts) - 1):
            kind = "resistance" if arcs is None else arcs[i]
            name = None if labels is None else labels[i]
            color = None if colors is None else colors[i % len(colors)]
            self.arc(pts[i], pts[i + 1], kind=kind, name=name, color=color)
        return self

    def locus(self, z, name, color=None, width=3.0, dash=None, arrows=2,
              customdata=None, hovertemplate=None):
        """Plot a swept locus (e.g. impedance versus frequency)."""
        z = np.asarray(z, dtype=complex)
        g = gamma_from_z(z)
        rr, xx = _smith_xy(g)
        kw = {}
        if hovertemplate is not None:
            kw = dict(customdata=customdata, hovertemplate=hovertemplate)
        else:
            kw = dict(hovertext=_hover(z, self.z0), hoverinfo="text+name")
        self.fig.add_trace(go.Scattersmith(
            real=rr, imag=xx, mode="lines", name=name,
            line=dict(color=color or self.P["SERIES"][0], width=width, dash=dash),
            **kw,
        ))
        self._arrows(g, color or self.P["SERIES"][0], arrows)
        return self

    def vswr_circle(self, s, name=None, color=None, dash="dash", width=1.5):
        rr, xx = _smith_xy(geo.vswr_circle(s))
        self.fig.add_trace(go.Scattersmith(
            real=rr, imag=xx, mode="lines",
            name=name or f"VSWR {s:.2f}",
            line=dict(color=color or self.P["INK_MUTED"], width=width, dash=dash),
            hoverinfo="name",
        ))
        return self

    def q_contour(self, q, color=None):
        for half in ("upper", "lower"):
            rr, xx = _smith_xy(geo.q_contour(q, half=half))
            self.fig.add_trace(go.Scattersmith(
                real=rr, imag=xx, mode="lines", name=f"Q = {q:g}",
                line=dict(color=color or self.P["ACCENT"], width=1.3, dash="dot"),
                hoverinfo="name", showlegend=(half == "upper"),
                legendgroup=f"q{q}",
            ))
        return self

    def unity_circles(self):
        """The r = 1 and g = 1 circles -- the two runways into the centre."""
        for circ, color, name in (
            (geo.resistance_circle(1.0), self.P["C_SERIES_EL"], "r = 1"),
            (geo.conductance_circle(1.0), self.P["C_SHUNT_EL"], "g = 1"),
        ):
            rr, xx = _smith_xy(circ)
            self.fig.add_trace(go.Scattersmith(
                real=rr, imag=xx, mode="lines", name=name,
                line=dict(color=color, width=1.3, dash="dash"),
                opacity=0.55, hoverinfo="name",
            ))
        return self


def z_from_gamma_safe(g):
    """z from gamma, tolerating the |gamma| = 1 samples that hit infinity."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return z_from_gamma(g)
