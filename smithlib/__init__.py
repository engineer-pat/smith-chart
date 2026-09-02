"""smithlib -- Smith chart mathematics, plotting, and matching-network design.

Layers, lightest first:

* :mod:`smithlib.core`      -- gamma <-> z, VSWR, return loss, Q
* :mod:`smithlib.tline`     -- motion along transmission lines and stubs
* :mod:`smithlib.matching`  -- L-network, single-stub and quarter-wave solvers
* :mod:`smithlib.geometry`  -- the chart's circles and arcs, as plain arrays
* :mod:`smithlib.chart`     -- a matplotlib Smith chart with arrow annotations

Only :mod:`smithlib.chart` needs matplotlib; everything else is numpy alone.
"""

from .core import (Z_from_gamma, gamma_from_Z, gamma_from_z, mismatch_loss_db,
                   q_of_z, reflected_power_fraction, return_loss_db, vswr,
                   y_from_z, z_from_gamma)
from .matching import (format_component, l_match, quarter_wave_match,
                       reactance_to_component, single_stub_match,
                       susceptance_to_component)
from .tline import (electrical_length_deg, line_input_Z, line_input_z,
                    rotate_gamma, stub_input_z, wavelength)

__version__ = "0.1.0"

__all__ = [
    "gamma_from_z", "z_from_gamma", "gamma_from_Z", "Z_from_gamma",
    "vswr", "return_loss_db", "mismatch_loss_db", "reflected_power_fraction",
    "q_of_z", "y_from_z",
    "rotate_gamma", "line_input_z", "line_input_Z", "stub_input_z",
    "wavelength", "electrical_length_deg",
    "l_match", "single_stub_match", "quarter_wave_match",
    "reactance_to_component", "susceptance_to_component", "format_component",
    "__version__",
]


def SmithChart(*args, **kwargs):
    """Lazily construct a :class:`smithlib.chart.SmithChart` (imports matplotlib)."""
    from .chart import SmithChart as _SC
    return _SC(*args, **kwargs)
