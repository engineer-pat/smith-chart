"""Checks on the theme layer.

The failure mode worth guarding against is *drift*: a colour role that exists
in one palette and not the other, or a theme switch that leaks out of the block
that asked for it.
"""

import pytest

from smithlib import style as S
from smithlib.chart import SmithChart
from smithlib.plotly_chart import SmithFigure


def test_both_palettes_define_the_same_roles():
    light, dark = S.palette("light"), S.palette("dark")
    assert set(light) == set(dark)
    for key in light:
        assert type(light[key]) is type(dark[key]), key


def test_semantic_roles_are_present_and_distinct():
    for name in ("light", "dark"):
        p = S.palette(name)
        roles = ["C_LOAD", "C_TARGET", "C_SERIES_EL", "C_SHUNT_EL", "C_LINE"]
        assert all(r in p for r in roles)
        # The three motion colours must not collide, or the chart is unreadable.
        assert len({p["C_SERIES_EL"], p["C_SHUNT_EL"], p["C_LINE"]}) == 3


def test_the_two_themes_actually_differ():
    assert S.palette("light")["SURFACE"] != S.palette("dark")["SURFACE"]
    assert S.palette("light")["INK"] != S.palette("dark")["INK"]


def test_theme_context_manager_restores_the_previous_theme():
    S.use_theme("light")
    before = S.SURFACE
    with S.theme("dark"):
        assert S.current_theme() == "dark"
        assert S.SURFACE != before
    assert S.current_theme() == "light"
    assert S.SURFACE == before


def test_theme_restores_even_if_the_block_raises():
    S.use_theme("light")
    with pytest.raises(RuntimeError):
        with S.theme("dark"):
            raise RuntimeError("boom")
    assert S.current_theme() == "light"


def test_unknown_theme_is_rejected():
    with pytest.raises(ValueError):
        S.use_theme("solarized")
    assert S.current_theme() in S.THEMES


def test_matplotlib_rc_tracks_the_active_theme():
    with S.theme("dark"):
        rc = S.matplotlib_rc()
        assert rc["figure.facecolor"] == S.palette("dark")["SURFACE"]
        assert rc["text.color"] == S.palette("dark")["INK"]


def test_plotly_figure_uses_the_palette_it_was_given_not_the_global():
    """Two figures on different themes must not interfere."""
    S.use_theme("light")
    dark_fig = SmithFigure(palette=S.palette("dark"))
    light_fig = SmithFigure(palette=S.palette("light"))
    assert dark_fig.fig.layout.paper_bgcolor == S.palette("dark")["SURFACE"]
    assert light_fig.fig.layout.paper_bgcolor == S.palette("light")["SURFACE"]
    assert S.current_theme() == "light"      # untouched by either figure


def test_matplotlib_chart_follows_the_active_theme():
    import matplotlib
    matplotlib.use("Agg")
    for name in ("light", "dark"):
        with S.theme(name):
            sc = SmithChart(figsize=(3, 3))
            sc.grid(labels=False)
            assert sc.ax.get_facecolor() == \
                matplotlib.colors.to_rgba(S.palette(name)["SURFACE"])
            sc.fig.clf()


# --- the app's Streamlit theme --------------------------------------------

def test_generated_streamlit_config_is_in_step_with_the_palette():
    """The committed config.toml must match what the palettes generate.

    Streamlit reads its theme from a file at server start, so the app frame
    cannot follow the palette at runtime; the file is generated instead, and
    this is what stops it drifting.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    r = subprocess.run(
        [sys.executable, str(root / "scripts" / "gen_streamlit_theme.py"), "--check"],
        capture_output=True, text=True, cwd=root,
    )
    assert r.returncode == 0, (
        r.stderr.strip() or "run: python scripts/gen_streamlit_theme.py")


def test_streamlit_chrome_matches_the_chart_surface():
    import streamlit as st
    for name in ("light", "dark"):
        assert st.get_option(f"theme.{name}.backgroundColor") == \
            S.palette(name)["SURFACE"]
        assert st.get_option(f"theme.{name}.textColor") == S.palette(name)["INK"]
