#!/usr/bin/env python3
"""Generate ``.streamlit/config.toml`` from the palettes in ``smithlib.style``.

Streamlit's chrome is themed by a config file read at server start, so it
cannot ask the palette for colours at runtime the way the charts do.  Copying
hex values by hand would let the app frame drift away from the chart surface,
so the file is generated instead, and ``tests/test_style.py`` asserts the
committed copy is still in step.

    python scripts/gen_streamlit_theme.py          # write the file
    python scripts/gen_streamlit_theme.py --check  # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smithlib import style as S  # noqa: E402

CONFIG = ROOT / ".streamlit" / "config.toml"

#: Which theme the app opens in. Streamlit's Settings menu overrides it per
#: browser, and the sidebar control repoints the charts independently.
DEFAULT_BASE = "dark"

HEADER = f"""\
# GENERATED FILE -- do not edit by hand.
#
#     python scripts/gen_streamlit_theme.py
#
# Streamlit reads this once at server start, so its chrome cannot follow the
# palette at runtime the way the charts do. Generating it from
# smithlib/style.py keeps the app frame and the chart surface the same colour.
# Edit the palettes there, then re-run the command above (`make theme`).
#
# `base` sets which theme the app opens in; the Settings menu (under the app's
# top-right menu) switches it per browser, and the charts follow.

[theme]
base = "{DEFAULT_BASE}"
font = "sans-serif"
showWidgetBorder = true
"""


def block(name):
    """The ``[theme.light]`` / ``[theme.dark]`` section for one palette.

    Deliberately no ``chartCategoricalColors``: that key only themes
    Streamlit's *native* charts, and every chart here is a Plotly figure
    coloured explicitly from the palette. Setting it bought nothing and
    warned on Streamlit older than 1.59.
    """
    p = S.palette(name)
    return f"""
[theme.{name}]
backgroundColor = "{p['SURFACE']}"
secondaryBackgroundColor = "{p['PANEL']}"
textColor = "{p['INK']}"
primaryColor = "{p['SERIES'][0]}"
linkColor = "{p['SERIES'][0]}"
borderColor = "{p['GRID_MAJOR']}"
dataframeBorderColor = "{p['GRID_MAJOR']}"
dataframeHeaderBackgroundColor = "{p['PANEL']}"
codeBackgroundColor = "{p['PANEL']}"
"""


def render():
    return HEADER + block("light") + block("dark") + """
[browser]
gatherUsageStats = false
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed file is stale")
    args = ap.parse_args()

    want = render()
    if args.check:
        have = CONFIG.read_text() if CONFIG.exists() else ""
        if have != want:
            print(f"{CONFIG} is out of date; run: "
                  f"python scripts/gen_streamlit_theme.py", file=sys.stderr)
            return 1
        print(f"{CONFIG} is up to date")
        return 0

    CONFIG.parent.mkdir(exist_ok=True)
    CONFIG.write_text(want)
    print(f"wrote {CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
