"""The load control must be usable, and reachable, from every tab.

The regression this guards against: the |Gamma| sliders were briefly moved onto
the Explore tab, which left the other three tabs with no way to change the load
at all. They belong in the sidebar, which renders alongside every tab.
"""

from pathlib import Path

import numpy as np
import pytest

import smithlib as sm

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
GAMMA_ENTRY = "|Γ| ∠ θ"


def fresh():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def sidebar_sliders(at):
    return [el for el in at.sidebar if type(el).__name__ == "Slider"]


def test_polar_entry_is_the_default():
    """The sliders are the easiest way in, so they are what you land on."""
    at = fresh()
    entry = next(s for s in at.selectbox if s.label == "Enter as")
    assert entry.value == GAMMA_ENTRY
    assert entry.options[0] == GAMMA_ENTRY


def test_the_sliders_live_in_the_sidebar_so_every_tab_can_reach_them():
    at = fresh()
    labels = [s.label for s in sidebar_sliders(at)]
    assert any(lbl.startswith("|Γ|") for lbl in labels), labels
    assert any(lbl.startswith("∠Γ") for lbl in labels), labels


def test_default_load_is_still_the_running_example():
    """The app should open on 25 - j40, the load the docs work through.

    Not exactly: the slider steps (0.001 and 0.1 deg) cannot express that
    load's polar form exactly, so 0.555 at -93.9 deg lands ~0.02 ohm away.
    Immaterial, but the assertion has to allow for it.
    """
    at = fresh()
    Z = complex(sm.Z_from_gamma(0.555 * np.exp(1j * np.radians(-93.9)), 50.0))
    assert Z.real == pytest.approx(25.0, abs=0.05)
    assert Z.imag == pytest.approx(-40.0, abs=0.05)
    shown = next(c.value for c in at.caption if "referenced to" in c.value)
    assert f"{Z.real:.2f}" in shown and f"{abs(Z.imag):.2f}" in shown


@pytest.mark.parametrize("mag", [0.2, 0.555, 0.8])
def test_magnitude_slider_sets_the_mismatch(mag):
    at = fresh()
    next(s for s in sidebar_sliders(at) if s.label.startswith("|Γ|")).set_value(mag)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.metric[0].value == f"{mag:.3f}"
    assert at.metric[2].value == f"{sm.vswr(mag):.2f}"
    assert at.metric[3].value == f"{sm.return_loss_db(mag):.1f} dB"


def test_angle_slider_rotates_without_changing_the_mismatch():
    at = fresh()
    next(s for s in sidebar_sliders(at) if s.label.startswith("|Γ|")).set_value(0.6)
    at.run()
    vswr_before = at.metric[2].value
    next(s for s in sidebar_sliders(at) if s.label.startswith("∠Γ")).set_value(120.0)
    at.run()
    assert at.metric[1].value == "120.0°"
    # Moving along a line changes phase only; VSWR is invariant.
    assert at.metric[2].value == vswr_before


def test_slider_load_matches_the_impedance_it_implies():
    at = fresh()
    mag, ang, z0 = 0.5, 60.0, 50.0
    next(s for s in sidebar_sliders(at) if s.label.startswith("|Γ|")).set_value(mag)
    next(s for s in sidebar_sliders(at) if s.label.startswith("∠Γ")).set_value(ang)
    at.run()
    want = complex(sm.Z_from_gamma(mag * np.exp(1j * np.radians(ang)), z0))
    shown = next(c.value for c in at.caption if "referenced to" in c.value)
    assert f"{want.real:.2f}" in shown and f"{abs(want.imag):.2f}" in shown


def test_explore_tab_points_newcomers_at_the_sliders():
    at = fresh()
    assert any("sliders" in c.value for c in at.caption)


def test_every_entry_mode_and_load_model_still_works():
    for mode in [GAMMA_ENTRY, "R + jX", "VSWR ∠ θ", "Series R–L / R–C"]:
        at = fresh()
        next(s for s in at.selectbox if s.label == "Enter as").set_value(mode)
        at.run()
        assert not at.exception, (mode, [str(e.value) for e in at.exception])
    for model in ["Fixed Z", "Series RLC", "Parallel RLC"]:
        at = fresh()
        next(r for r in at.radio if r.label == "Model").set_value(model)
        at.run()
        assert not at.exception, (model, [str(e.value) for e in at.exception])
