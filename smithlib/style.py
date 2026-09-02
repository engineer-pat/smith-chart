"""Theme-aware colour and line-weight tokens for every renderer.

There are two palettes, light and dark, and one *active* set of module-level
names.  Renderers read those names at draw time (``S.SURFACE``, ``S.C_LOAD``,
...), so switching theme is a matter of rebinding them -- no renderer needs to
know a theme exists:

    from smithlib import style as S

    with S.theme("dark"):
        fig = draw_something()          # comes out dark

Highlight colours are the first three slots of the validated categorical
palette, which is the subset cleared for all-pairs use (path/scatter forms like
ours, where any two series can end up adjacent).  The dark column is the same
three hues re-stepped for a dark surface, not a separate palette, plus a
reserved violet accent for transmission-line motion.  Where a figure needs a
fourth distinguishable locus we change dash pattern and direct-label it rather
than inventing a fourth hue.
"""

from __future__ import annotations

from contextlib import contextmanager

__all__ = [
    "LIGHT", "DARK", "THEMES", "palette", "use_theme", "theme",
    "current_theme", "matplotlib_rc", "SURFACE", "INK",
]

LIGHT = dict(
    SURFACE="#fcfcfb",
    PANEL="#f2f1ec",
    INK="#0b0b0b",
    INK_SOFT="#52514e",
    INK_MUTED="#8a8983",

    GRID_MINOR="#dcdbd5",
    GRID_MAJOR="#bab9b1",
    GRID_AXIS="#8a8983",
    RIM="#52514e",

    SERIES=["#2a78d6", "#eb6834", "#1baf7a"],
    ACCENT="#4a3aa7",
)

DARK = dict(
    SURFACE="#1a1a19",
    PANEL="#242422",
    INK="#ffffff",
    INK_SOFT="#c3c2b7",
    INK_MUTED="#8f8e85",

    # Stepped for the dark surface: the grid must sit just above the
    # background, not just below the foreground.
    GRID_MINOR="#333230",
    GRID_MAJOR="#4d4c47",
    GRID_AXIS="#7a7972",
    RIM="#c3c2b7",

    SERIES=["#3987e5", "#d95926", "#199e70"],
    ACCENT="#9085e9",
)

THEMES = {"light": LIGHT, "dark": DARK}

# Line weights and marker sizes are theme-independent.
LW_GRID = 0.6
LW_GRID_MAJOR = 0.9
LW_RIM = 1.4
LW_PATH = 2.0
MARKER_SIZE = 7.0

_active = "light"


def current_theme():
    """Name of the palette currently in effect."""
    return _active


def palette(name=None):
    """The fully resolved palette for a theme, semantic roles included.

    Returned as a plain dict so a caller can hold on to a snapshot instead of
    reading the module globals -- which matters anywhere two themes might be
    live at once, such as two browser sessions of the app.
    """
    p = dict(THEMES[name or _active])
    p["C_LOAD"] = p["INK"]              # where a design starts
    p["C_TARGET"] = p["SERIES"][2]      # the matched centre, the goal
    p["C_SERIES_EL"] = p["SERIES"][0]   # motion from a series element
    p["C_SHUNT_EL"] = p["SERIES"][1]    # motion from a shunt element
    p["C_LINE"] = p["ACCENT"]           # motion from a length of line
    return p


def use_theme(name="light"):
    """Switch the active palette, rebinding the module-level colour names.

    Convenient for the docs and for scripts, where one theme is live at a
    time.  Code that must not depend on global state should take a snapshot
    from :func:`palette` instead.
    """
    global _active
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; expected one of {list(THEMES)}")
    _active = name
    globals().update(palette(name))


@contextmanager
def theme(name):
    """Use a palette for the duration of a block, then restore the previous one."""
    previous = current_theme()
    use_theme(name)
    try:
        yield THEMES[name]
    finally:
        use_theme(previous)


def matplotlib_rc():
    """rcParams matching the active palette, for ``plt.rcParams.update()``."""
    return {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID_MAJOR,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "grid.color": GRID_MINOR,
        "font.size": 10,
    }


use_theme("light")
