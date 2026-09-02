"""Property-style checks on the RF maths.

These assert identities an RF engineer would recognise, rather than frozen
numbers, so they stay meaningful if the implementation is rewritten.
"""

import numpy as np
import pytest

import smithlib as sm
from smithlib import geometry as geo
from smithlib.matching import (l_match, quarter_wave_match, single_stub_match,
                               reactance_to_component, susceptance_to_component)

Z0 = 50.0
LOADS = [25 - 40j, 100 + 0j, 10 + 5j, 50 + 75j, 200 - 30j, 5 - 2j, 50 + 0j]


# --- core identities -------------------------------------------------------

def test_gamma_z_roundtrip():
    for ZL in LOADS:
        g = sm.gamma_from_Z(ZL, Z0)
        assert sm.Z_from_gamma(g, Z0) == pytest.approx(ZL, rel=1e-12)


def test_canonical_terminations():
    assert sm.gamma_from_z(1.0) == pytest.approx(0.0)          # matched
    assert sm.gamma_from_z(0.0) == pytest.approx(-1.0)         # short
    assert sm.gamma_from_z(np.inf) == pytest.approx(1.0)       # open


def test_passive_loads_stay_inside_the_chart():
    for ZL in LOADS:
        assert abs(sm.gamma_from_Z(ZL, Z0)) <= 1.0 + 1e-12


def test_vswr_and_return_loss_agree():
    for ZL in LOADS:
        g = sm.gamma_from_Z(ZL, Z0)
        s = sm.vswr(g)
        assert abs(g) == pytest.approx((s - 1) / (s + 1), abs=1e-12)
        if abs(g) == 0:
            assert np.isinf(sm.return_loss_db(g))    # a perfect match
        else:
            assert sm.return_loss_db(g) == pytest.approx(-20 * np.log10(abs(g)))


# --- transmission line -----------------------------------------------------

def test_half_wavelength_is_the_identity():
    for ZL in LOADS:
        assert sm.line_input_Z(ZL, 0.5, Z0) == pytest.approx(ZL, abs=1e-9)


def test_quarter_wavelength_inverts():
    for ZL in LOADS:
        zin = sm.line_input_Z(ZL, 0.25, Z0)
        assert zin == pytest.approx(Z0**2 / ZL, abs=1e-9)


def test_lossless_line_preserves_gamma_magnitude():
    g0 = sm.gamma_from_Z(25 - 40j, Z0)
    for d in np.linspace(0, 1, 17):
        assert abs(sm.rotate_gamma(g0, d)) == pytest.approx(abs(g0))


def test_lossy_line_spirals_inward():
    g0 = sm.gamma_from_Z(25 - 40j, Z0)
    g1 = sm.rotate_gamma(g0, 2.0, alpha_np_per_wl=0.05)
    assert abs(g1) < abs(g0)


def test_open_and_short_stubs_are_purely_reactive():
    for d in (0.05, 0.12, 0.3, 0.45):
        for kind in ("open", "short"):
            z = sm.stub_input_z(d, kind)
            assert z.real == pytest.approx(0.0, abs=1e-9)


def test_eighth_wave_shorted_stub_is_j():
    assert sm.stub_input_z(0.125, "short").imag == pytest.approx(1.0, abs=1e-9)


# --- matching networks -----------------------------------------------------

@pytest.mark.parametrize("ZL", LOADS)
def test_l_match_lands_on_the_origin(ZL):
    sols = l_match(ZL, Z0, freq_hz=1e9)
    assert sols
    for m in sols:
        assert m.path[-1] == pytest.approx(1 + 0j, abs=1e-9)


@pytest.mark.parametrize("ZL", LOADS)
@pytest.mark.parametrize("kind", ["open", "short"])
def test_single_stub_lands_on_the_origin(ZL, kind):
    sols = single_stub_match(ZL, Z0, stub_kind=kind, orientation="shunt")
    assert len(sols) >= 1
    for s in sols:
        assert s.path[-1] == pytest.approx(1 + 0j, abs=1e-6)
        assert 0.0 <= s.d_wl < 0.5 and 0.0 <= s.l_wl < 0.5


def test_stub_length_actually_produces_the_needed_susceptance():
    for ZL in LOADS:
        if ZL == Z0:            # already matched: no stub, nothing to check
            continue
        for s in single_stub_match(ZL, Z0, "open", "shunt"):
            b_needed = -(1.0 / s.z_at_stub).imag
            b_stub = (1.0 / sm.stub_input_z(s.l_wl, s.stub_kind)).imag
            assert b_stub == pytest.approx(b_needed, abs=1e-6)


@pytest.mark.parametrize("ZL", LOADS)
def test_quarter_wave_transformer_matches(ZL):
    q = quarter_wave_match(ZL, Z0)
    # The transformer sees a real R and converts it to Z0.
    zin = q.z0_transformer**2 / q.R_real
    assert zin == pytest.approx(Z0, rel=1e-9)


def test_component_conversions_round_trip():
    f = 2.4e9
    for X in (12.0, -30.0, 75.0):
        kind, val = reactance_to_component(X, f, Z0, normalized=False)
        w = 2 * np.pi * f
        back = w * val if kind == "L" else -1.0 / (w * val)
        assert back == pytest.approx(X)
    for B in (0.004, -0.01):
        kind, val = susceptance_to_component(B, f, Z0, normalized=False)
        w = 2 * np.pi * f
        back = w * val if kind == "C" else -1.0 / (w * val)
        assert back == pytest.approx(B)


# --- geometry --------------------------------------------------------------

def test_resistance_circles_pass_through_the_open_point():
    for r in (0.0, 0.5, 1.0, 5.0):
        c = geo.resistance_circle(r)
        assert np.min(np.abs(c - 1.0)) == pytest.approx(0.0, abs=1e-9)


def test_grid_arcs_stay_inside_the_unit_disc():
    for x in (0.2, 1.0, 5.0, -1.0):
        assert np.all(np.abs(geo.reactance_arc(x)) <= 1.0 + 1e-9)


def test_arc_between_starts_and_ends_where_told():
    z1 = 0.5 - 0.8j
    # Each kind gets an endpoint reachable by the move it represents.
    ends = {
        "resistance": z1 + 0.9j,                          # series reactance
        "conductance": 1.0 / (1.0 / z1 + 0.4j),           # shunt susceptance
        "gamma": sm.line_input_z(z1, 0.13),               # length of line
    }
    for kind, z2 in ends.items():
        pts = geo.arc_between(z1, z2, kind=kind)
        assert sm.z_from_gamma(pts[0]) == pytest.approx(z1, abs=1e-6)
        assert sm.z_from_gamma(pts[-1]) == pytest.approx(z2, abs=1e-6)


def test_series_element_walks_a_constant_resistance_arc():
    pts = geo.arc_between(0.5 - 0.8j, 0.5 + 0.3j, kind="resistance")
    assert np.allclose(sm.z_from_gamma(pts).real, 0.5, atol=1e-9)


def test_shunt_element_walks_a_constant_conductance_arc():
    z1 = 0.5 - 0.8j
    y1 = 1 / z1
    z2 = 1 / (y1 + 0.4j)
    pts = geo.arc_between(z1, z2, kind="conductance")
    assert np.allclose((1 / sm.z_from_gamma(pts)).real, y1.real, atol=1e-9)


def test_gamma_arc_is_the_clockwise_route():
    z1 = 0.5 - 0.8j
    z2 = sm.line_input_z(z1, 0.1)
    pts = geo.arc_between(z1, z2, kind="gamma")
    ang = np.unwrap(np.angle(pts))
    assert np.all(np.diff(ang) <= 1e-12)   # monotonically decreasing = clockwise
