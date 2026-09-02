"""Checks on the cascadable network model.

Each test pins an identity that has an independent closed form, so a bug in
``network.py`` cannot hide behind a bug in the solvers.
"""

import numpy as np
import pytest

import smithlib as sm
from smithlib.matching import l_match
from smithlib.network import (FixedLoad, Line, Network, ParallelRLC, SeriesC,
                              SeriesL, SeriesR, SeriesRLC, ShuntC, ShuntL,
                              ShuntR, Stub)
from smithlib.tline import C0

Z0 = 50.0
F0 = 1e9
LAM = C0 / F0


def test_series_and_shunt_elements_match_hand_algebra():
    w = 2 * np.pi * F0
    net = Network(load=FixedLoad(30 + 0j), elements=[SeriesL(2e-9)])
    assert net.Zin(F0) == pytest.approx(30 + 1j * w * 2e-9)

    net = Network(load=FixedLoad(30 + 0j), elements=[ShuntC(1e-12)])
    expected = 1 / (1 / 30 + 1j * w * 1e-12)
    assert net.Zin(F0) == pytest.approx(expected)

    net = Network(load=FixedLoad(30 + 0j), elements=[SeriesR(20.0)])
    assert net.Zin(F0) == pytest.approx(50 + 0j)

    net = Network(load=FixedLoad(100 + 0j), elements=[ShuntR(100.0)])
    assert net.Zin(F0) == pytest.approx(50 + 0j)


def test_quarter_wave_line_inverts():
    for ZL in (25 - 40j, 100 + 0j, 10 + 5j):
        net = Network(load=FixedLoad(ZL), elements=[Line(0.25 * LAM, Z0)])
        assert net.Zin(F0) == pytest.approx(Z0**2 / ZL, rel=1e-9)


def test_half_wave_line_is_the_identity():
    ZL = 25 - 40j
    net = Network(load=FixedLoad(ZL), elements=[Line(0.5 * LAM, Z0)])
    assert net.Zin(F0) == pytest.approx(ZL, rel=1e-9)


def test_line_with_a_different_z0_transforms_about_its_own_z0():
    # A quarter wave of 75 ohm line takes 25 ohm to 75^2/25 = 225 ohm.
    net = Network(load=FixedLoad(25 + 0j), elements=[Line(0.25 * LAM, 75.0)])
    assert net.Zin(F0) == pytest.approx(225 + 0j, rel=1e-9)


def test_lossy_line_reduces_the_reflection():
    ZL = 25 - 40j
    lossless = Network(load=FixedLoad(ZL), elements=[Line(2 * LAM, Z0)])
    lossy = Network(load=FixedLoad(ZL),
                    elements=[Line(2 * LAM, Z0, alpha_db_per_m=3.0 / LAM)])
    assert abs(sm.gamma_from_Z(lossy.Zin(F0), Z0)) < \
           abs(sm.gamma_from_Z(lossless.Zin(F0), Z0))


def test_eighth_wave_stubs_have_the_textbook_reactance():
    # Shorted lambda/8 stub: +jZ0.  Open lambda/8 stub: -jZ0.
    assert Stub(0.125 * LAM, Z0, kind="short").Z_stub(F0) == \
        pytest.approx(1j * Z0, abs=1e-6)
    assert Stub(0.125 * LAM, Z0, kind="open").Z_stub(F0) == \
        pytest.approx(-1j * Z0, abs=1e-6)


def test_quarter_wave_open_stub_shorts_the_line():
    net = Network(load=FixedLoad(50 + 0j),
                  elements=[Stub(0.25 * LAM, Z0, kind="open")])
    assert abs(net.Zin(F0)) < 1e-6


def test_trajectory_lines_up_with_zin():
    net = Network(load=FixedLoad(25 - 40j),
                  elements=[ShuntL(19.8e-9), SeriesL(7.03e-9)])
    traj = net.trajectory(F0)
    assert len(traj) == 3
    assert traj[0][1] == pytest.approx(25 - 40j)
    assert traj[-1][1] == pytest.approx(net.Zin(F0))


def test_sweep_is_vectorised_and_agrees_pointwise():
    net = Network(load=SeriesRLC(35.0, 8e-9, 3.17e-12),
                  elements=[ShuntC(1e-12), SeriesL(2e-9)])
    freqs = np.linspace(0.7e9, 1.3e9, 11)
    swept = net.sweep(freqs)
    assert swept.shape == freqs.shape
    for f, Z in zip(freqs, swept):
        assert complex(net.Zin(float(f))) == pytest.approx(Z, rel=1e-12)


def test_l_match_components_really_land_on_z0():
    """The solver and the network model must agree, end to end."""
    for ZL in (25 - 40j, 100 + 0j, 10 + 5j, 200 - 30j):
        for m in l_match(ZL, Z0, F0):
            els = []
            for role, kind, val, _ in m.components():
                cls = {("shunt", "L"): ShuntL, ("shunt", "C"): ShuntC,
                       ("series", "L"): SeriesL, ("series", "C"): SeriesC}[
                           (role.split()[0], kind)]
                els.append(cls(val))
            net = Network(load=FixedLoad(ZL), elements=els)
            assert net.Zin(F0) == pytest.approx(Z0 + 0j, abs=1e-6)


def test_resonant_loads_are_real_at_resonance():
    L, C = 8e-9, 3.17e-12
    fr = 1 / (2 * np.pi * np.sqrt(L * C))
    assert SeriesRLC(35.0, L, C).Z(fr).imag == pytest.approx(0.0, abs=1e-6)
    assert ParallelRLC(200.0, L, C).Z(fr).imag == pytest.approx(0.0, abs=1e-6)


def test_network_q_ignores_the_load_and_the_endpoint():
    """Network Q is a property of the corner, not of the load."""
    sols = l_match(25 - 40j, Z0, F0)
    load_q = sm.q_of_z((25 - 40j) / Z0)
    qs = {round(m.network_q, 6) for m in sols}
    assert len(qs) > 1, "Q must distinguish the topologies"
    assert all(q != pytest.approx(load_q) for q in qs)
