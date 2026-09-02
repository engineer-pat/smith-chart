"""The polar sliders on the Explore tab must actually drive the load.

They live on the Explore tab but are read by the sidebar, which runs earlier in
the script. That only works because Streamlit exposes a widget's value in
session_state from the start of the rerun it changed on -- so these tests pin
the round trip, not just that the widgets render.
"""

from pathlib import Path

import numpy as np
import pytest

import smithlib as sm

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "app.py")
GAMMA_ENTRY = "|Γ| ∠ θ"


def by_key(elements, key):
    return next((e for e in elements if getattr(e, "key", None) == key), None)


def fresh():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def in_gamma_mode():
    at = fresh()
    [b for b in at.button if "Switch to" in b.label][0].click()
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_sliders_are_hidden_until_polar_entry_is_selected():
    at = fresh()
    assert at.session_state.entry_mode == "R + jX"
    assert by_key(at.slider, "gamma_mag") is None
    assert any("Switch to" in b.label for b in at.button)


def test_the_switch_button_reveals_the_sliders():
    """Uses an on_click callback; a plain assignment would raise, because the
    sidebar instantiates the entry_mode widget before the tab body runs."""
    at = in_gamma_mode()
    assert at.session_state.entry_mode == GAMMA_ENTRY
    assert at.session_state.load_model == "Fixed Z"
    assert by_key(at.slider, "gamma_mag") is not None
    assert by_key(at.slider, "gamma_ang") is not None


@pytest.mark.parametrize("mag", [0.2, 0.555, 0.8])
def test_magnitude_slider_sets_the_mismatch(mag):
    at = in_gamma_mode()
    by_key(at.slider, "gamma_mag").set_value(mag)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.metric[0].value == f"{mag:.3f}"                       # |Γ|
    assert at.metric[2].value == f"{sm.vswr(mag):.2f}"              # VSWR
    assert at.metric[3].value == f"{sm.return_loss_db(mag):.1f} dB"  # return loss


def test_angle_slider_rotates_without_changing_the_mismatch():
    at = in_gamma_mode()
    by_key(at.slider, "gamma_mag").set_value(0.6)
    at.run()
    vswr_before = at.metric[2].value
    by_key(at.slider, "gamma_ang").set_value(120.0)
    at.run()
    assert at.metric[1].value == "120.0°"
    # Moving along a line changes the angle only; VSWR is invariant.
    assert at.metric[2].value == vswr_before


def test_the_sidebar_reports_what_the_sliders_are_set_to():
    at = in_gamma_mode()
    by_key(at.slider, "gamma_mag").set_value(0.75)
    by_key(at.slider, "gamma_ang").set_value(-30.0)
    at.run()
    echo = [c.value for c in at.caption if "Explore" in c.value]
    assert echo, "sidebar should say where the sliders are"
    assert "0.750" in echo[0] and "-30.0" in echo[0]


def test_slider_load_matches_the_impedance_it_implies():
    """The chart readout must agree with Z_from_gamma for the slider values."""
    at = in_gamma_mode()
    mag, ang, z0 = 0.5, 60.0, 50.0
    by_key(at.slider, "gamma_mag").set_value(mag)
    by_key(at.slider, "gamma_ang").set_value(ang)
    at.run()
    want = complex(sm.Z_from_gamma(mag * np.exp(1j * np.radians(ang)), z0))
    shown = next(c.value for c in at.caption if "referenced to" in c.value)
    assert f"{want.real:.2f}" in shown
    assert f"{abs(want.imag):.2f}" in shown
