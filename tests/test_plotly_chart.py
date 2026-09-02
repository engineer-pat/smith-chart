"""Regression tests for the Plotly renderer's coordinate handling.

Plotly's ``Scattersmith`` is fed *normalized impedance*, not the reflection
coefficient -- feeding it gamma silently produces a plausible-looking but
completely wrong chart (a matched load lands on the short-circuit point).
These tests pin the conversion.
"""

import numpy as np
import pytest

import smithlib as sm
from smithlib.plotly_chart import SmithFigure, _smith_xy


def test_smith_xy_returns_impedance_not_gamma():
    """Interior points convert exactly."""
    for g, z in [(0j, 1 + 0j), (0.5 + 0j, 3 + 0j), (0.5j, 0.6 + 0.8j)]:
        r, x = _smith_xy(g)
        assert complex(r[0], x[0]) == pytest.approx(z, abs=1e-9)


def test_rim_points_land_on_the_rim_within_the_nudge():
    """A short is |gamma| = 1, so it is pulled just inside to stay finite.

    The residual is the nudge itself (~2.5e-5 in z), which is far below one
    pixel on any real chart.
    """
    r, x = _smith_xy(-1 + 0j)
    assert complex(r[0], x[0]) == pytest.approx(0 + 0j, abs=1e-4)
    r, x = _smith_xy(1 + 0j)
    assert r[0] > 1e4                       # an open circuit, still huge


def test_rim_samples_stay_finite():
    """Points on |gamma| = 1 must not become inf/NaN coordinates."""
    rim = np.exp(1j * np.linspace(0, 2 * np.pi, 361))
    r, x = _smith_xy(rim)
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(x))


def test_matched_load_is_plotted_at_the_chart_centre():
    f = SmithFigure(z0=50.0)
    f.point(1 + 0j, "matched")
    tr = f.fig.data[0]
    assert float(tr.real[0]) == pytest.approx(1.0)
    assert float(tr.imag[0]) == pytest.approx(0.0)


def test_arc_endpoints_are_the_impedances_asked_for():
    z1, z2 = 0.5 - 0.8j, 0.5 + 0.3j
    f = SmithFigure(z0=50.0)
    f.arc(z1, z2, kind="resistance", arrows=0)
    tr = f.fig.data[0]
    assert complex(tr.real[0], tr.imag[0]) == pytest.approx(z1, abs=1e-6)
    assert complex(tr.real[-1], tr.imag[-1]) == pytest.approx(z2, abs=1e-6)


def test_vswr_circle_has_the_right_radius_in_gamma():
    f = SmithFigure(z0=50.0)
    f.vswr_circle(3.0)
    tr = f.fig.data[0]
    z = np.array(tr.real) + 1j * np.array(tr.imag)
    mags = np.abs(sm.gamma_from_z(z))
    assert np.allclose(mags, 0.5, atol=1e-6)     # VSWR 3 -> |gamma| = 0.5


def test_figure_builds_for_every_helper():
    f = SmithFigure(z0=50.0, title="t")
    f.unity_circles().q_contour(2).vswr_circle(2.0)
    f.point(0.5 - 0.8j, "load")
    f.arc(0.5 - 0.8j, 0.5 + 0.3j, "resistance", name="series L")
    f.locus(np.array([0.5 - 0.8j, 0.7 - 0.2j, 1 + 0j]), "sweep")
    f.fig.to_json()                               # must serialise cleanly
