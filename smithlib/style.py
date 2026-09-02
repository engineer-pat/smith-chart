"""Shared colour and line-weight tokens for every renderer.

Highlight colours are the first three slots of the validated categorical
palette, which are the subset cleared for all-pairs use (scatter/path forms
like ours, where any two series can end up adjacent).  Where a figure needs a
fourth distinguishable locus we change dash pattern and direct-label it rather
than inventing a fourth hue.
"""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8983"

GRID_MINOR = "#dcdbd5"
GRID_MAJOR = "#bab9b1"
GRID_AXIS = "#8a8983"
RIM = "#52514e"

# Categorical slots 1-3 (all-pairs validated), then a reserved accent.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
ACCENT = "#4a3aa7"

# Semantic roles built from those slots, so figures stay consistent.
C_LOAD = INK              # where you start
C_TARGET = "#1baf7a"      # the matched centre / goal
C_SERIES_EL = "#2a78d6"   # motion caused by a series element
C_SHUNT_EL = "#eb6834"    # motion caused by a shunt element
C_LINE = "#4a3aa7"        # motion caused by a length of transmission line

LW_GRID = 0.6
LW_GRID_MAJOR = 0.9
LW_RIM = 1.4
LW_PATH = 2.0
MARKER_SIZE = 7.0
