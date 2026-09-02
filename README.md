# smithChart

Learning and working with Smith charts: a written explanation, a small RF
library, and an interactive app.

Three pieces, in the order they are useful:

| Piece | What it is |
|---|---|
| [`docs/smith-chart.qmd`](docs/smith-chart.qmd) | The tutorial. What the chart is, why it exists, and worked examples with the moves drawn as arrows on the chart. |
| [`smithlib/`](smithlib/) | The math. Reflection coefficients, transmission lines, matching-network synthesis, and two chart renderers. |
| [`app/app.py`](app/app.py) | An interactive workbench built on `smithlib`. |

Both the document and the app have a dark mode — see [Themes](#themes).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[app,docs,dev]"

.venv/bin/streamlit run app/app.py            # the interactive app
.venv/bin/python -m pytest                    # the test suite

cd docs && quarto render smith-chart.qmd      # the tutorial (needs Quarto)
```

Rendering writes both themes' figures into `docs/figures/` (git-ignored).

```bash
make venv && make test && make docs && make app
```

Rendering the docs needs the venv's Python. If Quarto picks the wrong one:

```bash
QUARTO_PYTHON=$PWD/.venv/bin/python quarto render docs/smith-chart.qmd
```

> Quarto keeps a **persistent Jupyter kernel** between renders. If a render
> behaves as though it is running stale code, add `--execute-daemon-restart`.

## Why it is laid out this way

The math is deliberately separate from every user interface. `core`, `tline`,
`matching` and `network` import nothing but numpy, so they work equally well in
a measurement script, a notebook, the Quarto document, or the app. The chart
renderers sit on top and are interchangeable:

```
core.py      gamma <-> z, VSWR, return loss, Q            numpy only
tline.py     motion along lines, stubs                    numpy only
matching.py  L-network / stub / quarter-wave solvers      numpy only
network.py   cascadable element chain + frequency sweep   numpy only
geometry.py  the chart's circles and arcs, as arrays      numpy only
style.py     light and dark palettes, one set of roles    no deps
  chart.py         matplotlib renderer  -> print, docs
  plotly_chart.py  Plotly renderer      -> the app
```

The app is Streamlit rather than PyQt because the interesting part here is the
library, and Streamlit costs almost nothing to iterate on and nothing to share
with a colleague. Since no UI code holds any of the maths, swapping in a Qt or
web front end later touches nothing under `smithlib/`.

## Using the library directly

```python
import numpy as np
import smithlib as sm
from smithlib.matching import l_match
from smithlib.network import FixedLoad, Network, ShuntL, SeriesL

ZL, Z0, f0 = 25 - 40j, 50.0, 1e9

g = sm.gamma_from_Z(ZL, Z0)
sm.vswr(g), sm.return_loss_db(g)              # 3.49, 5.1 dB

for m in l_match(ZL, Z0, freq_hz=f0):
    print(m.topology, [t for *_, t in m.components()], f"Q={m.network_q:.2f}")

net = Network(load=FixedLoad(ZL), elements=[ShuntL(19.8e-9), SeriesL(7.03e-9)])
net.Zin(f0)                                    # ~50 ohm (rounded part values)
net.sweep(np.linspace(0.8e9, 1.2e9, 401))      # vectorised over frequency
```

Drawing a match, with direction arrows on every step:

```python
from smithlib.chart import SmithChart

sc = SmithChart(title="L-network match", z0=Z0)
sc.grid()
sc.vswr_circle(sm.vswr(g), label="load VSWR")
sc.point(ZL / Z0, "load", show_value=True)
sc.path(m.steps, ZL / Z0)                      # arcs + arrowheads + labels
sc.point(1 + 0j, "matched")
```

## Moving the load around the chart

The Explore tab carries two sliders — |Γ| and ∠Γ — that walk the load around
the chart while every readout, matching solution and sweep follows. Radius is
the mismatch; angle is where a length of line puts you, with half a wavelength
being one full turn. It is the closest thing to dragging a point around the
chart, and it is precise, which dragging would not be.

They appear once the load is entered as **|Γ| ∠ θ** (`Load → Enter as` in the
sidebar); the tab offers a one-click switch when it is not. The sidebar then
echoes the current values, since the sliders themselves live on the tab so you
can watch the chart while moving them.

Actual point-and-drag is not on the table in Streamlit: every interaction is a
server round trip (~90 ms for this app before the browser even redraws), where
dragging needs sub-frame feedback. Plotly's Smith subplots also have no
box/lasso `dragmode`, so there is no way to read a coordinate out of empty
chart space — only points belonging to a trace. Doing it properly would mean a
custom bidirectional component; the sliders get the same intuition for a
fraction of the cost.

## Themes

Colour lives in one place, [`smithlib/style.py`](smithlib/style.py), as two
palettes over one set of semantic roles — `C_LOAD` (where a design starts),
`C_TARGET` (the matched centre), and one colour each for the three moves a
Smith chart can express. Renderers ask for roles, never for hex, so neither
theme can drift out of step with the other.

```python
from smithlib import style as S

with S.theme("dark"):          # for scripts and the docs
    fig = draw_something()

P = S.palette("dark")          # a snapshot, for code that must not touch globals
SmithFigure(palette=P, z0=50)
```

**The document** gets a light/dark toggle in the Quarto navbar. Baked PNGs
cannot restyle themselves, so every figure is rendered twice — once per palette
— and emitted as a pair of images that `docs/styles.css` swaps on Quarto's
`quarto-dark` body class. The `dual()` helper in the document's setup chunk
does this, which is why each figure is written as a function returning a figure
rather than as loose script: the drawing has to be repeatable.

Note that Quarto's toggle **ignores the OS colour-scheme preference** and starts
light until the reader chooses otherwise. The swap CSS is therefore keyed only
on the body class — honouring `prefers-color-scheme` there would put dark
figures on a light page.

**The app** is dark by default, chrome and charts together. Streamlit themes
its own frame from `.streamlit/config.toml`, read once at server start, so it
cannot follow the palette at runtime the way the charts do — the file is
generated from `smithlib/style.py` instead:

```bash
make theme            # regenerate after editing the palettes
```

A test fails if the committed file drifts out of step, so the app frame and the
chart surface stay the same colour.

Switching themes under *Settings* (the app's top-right ⋮ menu) restyles the
chrome, and the charts follow via `st.context.theme`. That is only reported
back *after* the first script run, so the fallback is the configured
`theme.base` rather than a hardcoded light — otherwise every fresh session
would flash light charts onto dark chrome. The sidebar also has an explicit
override if you want the charts pinned regardless of the frame.

To open light instead, change `base` in `scripts/gen_streamlit_theme.py`
(`DEFAULT_BASE`) and re-run `make theme`. Note that Streamlit only picks the
config up when launched from the project root, which is what `make app` does.

Each script run takes one palette snapshot and passes it explicitly into every
figure, so two browser sessions on different themes stay independent.

## A note on bandwidth

The app and the docs report a **network Q** for each L-network solution: the Q
at the intermediate node, excluding the load's own Q. It is worth knowing what
that number does and does not tell you.

- It ranks the *geometry*. A load far from the centre forces a sharp corner and
  a narrow match, and no two-element network escapes that.
- It does **not** rank the variants. The high-pass and low-pass versions of one
  topology share a corner, so they share a Q — but their real 10 dB bandwidths
  can differ by a factor of three, because component reactances drift in
  opposite directions with frequency.

So use Q to understand why a match is hard, and the sweep to choose between
candidates. `docs/smith-chart.qmd` works this through with the numbers.

## Extending it

Natural next steps, roughly in order of usefulness:

- **Touchstone / measured data.** Feed real S-parameters in place of an ideal
  load. [`scikit-rf`](https://scikit-rf.org) reads `.s1p`/`.s2p` and would slot
  in behind the `Load` interface in `network.py`.
- **Two-port networks.** Everything here is one-port; S/ABCD cascades would
  bring in gain circles, stability circles and noise circles.
- **Component realism.** Finite-Q inductors, capacitor ESR/SRF, and E-series
  value snapping, so the sweep reflects parts you can actually buy.
- **Three-element networks.** Pi and T topologies, where Q becomes a design
  input rather than an outcome.

## Testing

```bash
.venv/bin/python -m pytest -q
```

The suite asserts identities an RF engineer would recognise — a half-wave line
is the identity, a quarter-wave line inverts, an eighth-wave shorted stub is
+jZ0, every synthesised match lands on Z0 — rather than frozen numbers, so it
stays meaningful if the internals are rewritten.

## References

Pozar, *Microwave Engineering*, chapters 2 and 5. Smith's original is
*Transmission Line Calculator*, Electronics, January 1939.

## Licence

MIT — see [LICENSE](LICENSE).
