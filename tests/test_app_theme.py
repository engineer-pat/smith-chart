"""The app must paint its charts in whatever theme it is actually showing.

The bug these guard against is a light chart on dark chrome: Streamlit only
reports the browser's theme *after* the first script run, so a naive fallback
to "light" makes every fresh session flash the wrong palette.
"""

import json
from pathlib import Path

import pytest

from smithlib import style as S

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves relative paths against *this* file, not the cwd.
APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
THEME_RADIO = 2          # sidebar: Match Streamlit / Light / Dark


def chart_backgrounds(at):
    """Every Plotly figure's paper background in the rendered app."""
    out = set()
    for el in at.main:
        if type(el).__name__ == "UnknownElement" and hasattr(el.proto, "spec"):
            bg = json.loads(el.proto.spec).get("layout", {}).get("paper_bgcolor")
            if bg:
                out.add(bg)
    return out


def run(setup=None):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    if setup:
        setup(at)
        at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_app_runs_in_every_theme_setting():
    for choice in ("Match Streamlit", "Light", "Dark"):
        run(lambda at, c=choice: at.radio[THEME_RADIO].set_value(c))


def test_explicit_theme_choice_is_honoured():
    assert chart_backgrounds(run(lambda at: at.radio[THEME_RADIO].set_value("Light"))) \
        == {S.palette("light")["SURFACE"]}
    assert chart_backgrounds(run(lambda at: at.radio[THEME_RADIO].set_value("Dark"))) \
        == {S.palette("dark")["SURFACE"]}


def test_default_run_follows_the_configured_base_not_hardcoded_light():
    """With no browser report yet, charts must match the configured base."""
    import streamlit as st

    base = st.get_option("theme.base") or "light"
    assert chart_backgrounds(run()) == {S.palette(base)["SURFACE"]}


def test_every_chart_in_a_run_shares_one_theme():
    """A single run must not mix palettes across its figures."""
    for choice in ("Match Streamlit", "Light", "Dark"):
        bgs = chart_backgrounds(run(lambda at, c=choice: at.radio[THEME_RADIO].set_value(c)))
        assert len(bgs) == 1, f"{choice} produced mixed backgrounds: {bgs}"


def test_charts_opt_out_of_streamlits_plotly_template():
    """st.plotly_chart must be called with theme=None.

    The default, theme="streamlit", repaints the figure with Streamlit's own
    template and overrides the backgrounds set from the palette. That made
    charts render bright wherever Streamlit's own theme resolved light -- and
    the sidebar override could not win, because the repaint happens in the
    browser after the spec is sent.
    """
    at = run()
    seen = [el.proto.theme for el in at.main
            if type(el).__name__ == "UnknownElement" and hasattr(el.proto, "spec")]
    assert seen, "expected at least one Plotly chart"
    assert all(t != "streamlit" for t in seen), seen


def test_forced_dark_survives_all_the_way_to_the_wire():
    """A dark chart must stay dark in the spec actually sent to the browser."""
    at = run(lambda a: a.radio[THEME_RADIO].set_value("Dark"))
    assert chart_backgrounds(at) == {S.palette("dark")["SURFACE"]}
    for el in at.main:
        if type(el).__name__ == "UnknownElement" and hasattr(el.proto, "spec"):
            assert el.proto.theme != "streamlit"
