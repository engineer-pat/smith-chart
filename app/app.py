"""Interactive Smith chart workbench.

Run with:  streamlit run app/app.py

The app is deliberately a thin shell over :mod:`smithlib` -- every number it
shows comes from the library, so anything you work out here can be reproduced
in a script with the same three or four calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import smithlib as sm
from smithlib import style as S
from smithlib.matching import (format_component, l_match, quarter_wave_match,
                               single_stub_match)
from smithlib.network import (FixedLoad, Line, Network, ParallelRLC, SeriesC,
                              SeriesL, SeriesR, SeriesRLC, ShuntC, ShuntL,
                              ShuntR, Stub)
from smithlib.plotly_chart import SmithFigure
from smithlib.tline import C0

st.set_page_config(page_title="Smith Chart Workbench", page_icon="📡",
                   layout="wide")

PLOT_CFG = {"displaylogo": False,
            "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def detect_streamlit_theme():
    """Whatever theme Streamlit itself is currently showing.

    ``st.context.theme`` is reported back by the browser, so it is unset on the
    very first script run of a session and on older Streamlit versions.  The
    fallback is the configured ``theme.base`` -- which is exactly what
    Streamlit paints until the browser reports in -- rather than a hardcoded
    "light", which would flash light charts onto dark chrome.
    """
    try:
        reported = st.context.theme.type
    except Exception:
        reported = None
    if reported in ("light", "dark"):
        return reported
    base = st.get_option("theme.base")
    return base if base in ("light", "dark") else "light"


# ==========================================================================
# session state
# ==========================================================================

if "elements" not in st.session_state:
    st.session_state.elements = []


def reset_elements():
    st.session_state.elements = []


# ==========================================================================
# sidebar: system, frequency, load
# ==========================================================================

with st.sidebar:
    st.header("System")
    z0 = st.number_input("Reference $Z_0$ (Ω)", 1.0, 1000.0, 50.0, 1.0)
    f0_ghz = st.number_input("Design frequency (GHz)", 0.001, 200.0, 1.0,
                             0.05, format="%.4f")
    f0 = f0_ghz * 1e9

    st.divider()
    st.header("Load")
    load_kind = st.radio(
        "Model", ["Fixed Z", "Series RLC", "Parallel RLC"],
        help="A fixed impedance is enough for a single-frequency design. "
             "The RLC models give a load that actually changes with "
             "frequency, so the sweep tab means something.",
    )

    if load_kind == "Fixed Z":
        entry = st.selectbox("Enter as", ["R + jX", "|Γ| ∠ θ", "VSWR ∠ θ",
                                          "Series R–L / R–C"])
        if entry == "R + jX":
            c1, c2 = st.columns(2)
            R = c1.number_input("R (Ω)", 0.0, 1e6, 25.0, 1.0)
            X = c2.number_input("X (Ω)", -1e6, 1e6, -40.0, 1.0)
            ZL = complex(R, X)
        elif entry == "|Γ| ∠ θ":
            c1, c2 = st.columns(2)
            mag = c1.slider("|Γ|", 0.0, 0.999, 0.555, 0.001)
            ang = c2.slider("∠Γ (deg)", -180.0, 180.0, -93.9, 0.1)
            ZL = complex(sm.Z_from_gamma(mag * np.exp(1j * np.radians(ang)), z0))
        elif entry == "VSWR ∠ θ":
            c1, c2 = st.columns(2)
            s_in = c1.number_input("VSWR", 1.0, 100.0, 3.5, 0.1)
            ang = c2.slider("∠Γ (deg)", -180.0, 180.0, -93.9, 0.1)
            mag = (s_in - 1) / (s_in + 1)
            ZL = complex(sm.Z_from_gamma(mag * np.exp(1j * np.radians(ang)), z0))
        else:
            c1, c2 = st.columns(2)
            R = c1.number_input("R (Ω)", 0.0, 1e6, 25.0, 1.0)
            which = c2.selectbox("with", ["L (nH)", "C (pF)"])
            val = c2.number_input("value", 0.0001, 1e6, 6.37, 0.01)
            w = 2 * np.pi * f0
            X = w * val * 1e-9 if which.startswith("L") else -1 / (w * val * 1e-12)
            ZL = complex(R, X)
        load = FixedLoad(ZL)

    elif load_kind == "Series RLC":
        c1, c2, c3 = st.columns(3)
        R = c1.number_input("R (Ω)", 0.01, 1e5, 35.0, 1.0)
        Lh = c2.number_input("L (nH)", 0.001, 1e6, 8.0, 0.1)
        Cf = c3.number_input("C (pF)", 0.0001, 1e6, 3.17, 0.01)
        load = SeriesRLC(R, Lh * 1e-9, Cf * 1e-12)
        ZL = complex(load.Z(f0))
    else:
        c1, c2, c3 = st.columns(3)
        R = c1.number_input("R (Ω)", 0.01, 1e6, 200.0, 1.0)
        Lh = c2.number_input("L (nH)", 0.001, 1e6, 8.0, 0.1)
        Cf = c3.number_input("C (pF)", 0.0001, 1e6, 3.17, 0.01)
        load = ParallelRLC(R, Lh * 1e-9, Cf * 1e-12)
        ZL = complex(load.Z(f0))

    st.divider()
    st.header("Sweep band")
    c1, c2 = st.columns(2)
    f_lo = c1.number_input("start (GHz)", 0.0001, 200.0,
                           max(f0_ghz * 0.5, 1e-4), 0.01, format="%.4f")
    f_hi = c2.number_input("stop (GHz)", 0.0002, 400.0, f0_ghz * 1.5, 0.01,
                           format="%.4f")
    n_pts = st.slider("points", 51, 2001, 401, 50)
    freqs = np.linspace(f_lo * 1e9, max(f_hi * 1e9, f_lo * 1e9 * 1.001), n_pts)

    st.divider()
    st.header("Appearance")
    theme_choice = st.radio(
        "Chart theme", ["Match Streamlit", "Light", "Dark"], horizontal=True,
        help="Charts follow Streamlit's own theme by default — change that "
             "under Settings in the ⋮ menu and the charts follow.")
    mode = (detect_streamlit_theme() if theme_choice == "Match Streamlit"
            else theme_choice.lower())

    st.divider()
    st.header("Chart overlays")
    show_unity_circles = st.checkbox(
        "$r=1$ and $g=1$ circles", value=True,
        help="The only two circles that pass through the centre — every "
             "two-element match has to reach one of them.")
    show_q_contours = st.checkbox(
        "constant-$Q$ contours", value=False,
        help="Bandwidth guides. Useful, but they clutter the chart, so they "
             "are off by default.")

    st.divider()
    eps_eff = st.number_input(
        "Line $\\varepsilon_{eff}$", 1.0, 20.0, 1.0, 0.1,
        help="Effective permittivity for physical line lengths. "
             "1.0 = air/coax with vf 1; FR-4 microstrip is roughly 3.0.",
    )
    lam0 = C0 / (f0 * np.sqrt(eps_eff))
    st.caption(f"λ at {f0_ghz:g} GHz = {lam0 * 1000:.2f} mm")


# One snapshot per script run, passed explicitly into every figure, so two
# browser sessions on different themes cannot tread on each other.
P = S.palette(mode)

gL = sm.gamma_from_Z(ZL, z0)
zL = ZL / z0

# ==========================================================================
# header readout
# ==========================================================================

st.title("Smith Chart Workbench")
st.caption(f"{load.describe()}  →  "
           f"Z = {ZL.real:.2f} {'+' if ZL.imag >= 0 else '-'} "
           f"j{abs(ZL.imag):.2f} Ω at {f0_ghz:g} GHz, referenced to {z0:g} Ω")

k = st.columns(6)
k[0].metric("|Γ|", f"{abs(gL):.3f}")
k[1].metric("∠Γ", f"{np.degrees(np.angle(gL)):.1f}°")
k[2].metric("VSWR", f"{sm.vswr(gL):.2f}")
k[3].metric("Return loss", f"{sm.return_loss_db(gL):.1f} dB")
k[4].metric("Mismatch loss", f"{sm.mismatch_loss_db(gL):.2f} dB")
k[5].metric("Power reflected", f"{100 * sm.reflected_power_fraction(gL):.1f} %")


# ==========================================================================
# helpers
# ==========================================================================

def sweep_panel(net, title, extra=None):
    """VSWR and return loss versus frequency for a network."""
    Zin = net.sweep(freqs)
    g = sm.gamma_from_Z(Zin, z0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs / 1e9, y=sm.return_loss_db(g), name="return loss",
        line=dict(color=P["SERIES"][0], width=2.5),
        hovertemplate="%{x:.4f} GHz<br>RL %{y:.2f} dB<extra></extra>"))
    fig.add_hline(y=10, line=dict(color=P["C_TARGET"], width=1.4, dash="dash"),
                  annotation_text="10 dB", annotation_position="top left",
                  annotation_font_color=P["INK_SOFT"])
    fig.add_vline(x=f0 / 1e9,
                  line=dict(color=P["INK_MUTED"], width=1, dash="dot"))
    fig.update_layout(
        title=title, height=300, margin=dict(l=10, r=10, t=44, b=10),
        paper_bgcolor=P["SURFACE"], plot_bgcolor=P["SURFACE"],
        font=dict(color=P["INK"], size=12), showlegend=False,
        xaxis=dict(title="frequency (GHz)", gridcolor=P["GRID_MINOR"],
                   zeroline=False),
        yaxis=dict(title="return loss (dB)", gridcolor=P["GRID_MINOR"],
                   zeroline=False, autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG,
                    key=f"sweep-{title}")

    # Usable bandwidth = the contiguous run around f0 that stays above 10 dB.
    ok = sm.return_loss_db(g) >= 10
    if ok.any():
        i0 = int(np.argmin(np.abs(freqs - f0)))
        if ok[i0]:
            lo = i0
            while lo > 0 and ok[lo - 1]:
                lo -= 1
            hi = i0
            while hi < len(ok) - 1 and ok[hi + 1]:
                hi += 1
            bw = freqs[hi] - freqs[lo]
            st.caption(f"≥10 dB return loss from {freqs[lo]/1e9:.4f} to "
                       f"{freqs[hi]/1e9:.4f} GHz — {bw/1e6:.1f} MHz "
                       f"({100 * bw / f0:.1f} % fractional)")
        else:
            st.caption("Does not meet 10 dB return loss at the design "
                       "frequency.")
    else:
        st.caption("Never reaches 10 dB return loss in this band.")


def chart_for_trajectory(traj, title, show_unity=None, q_marks=None):
    """Smith chart of a load-to-source trajectory with arrows on every step.

    Overlays default to the sidebar toggles; every trace is also clickable in
    the Plotly legend, so anything can be hidden without a rerun.
    """
    f = SmithFigure(palette=P, z0=z0, title=title, height=620)
    if show_unity if show_unity is not None else show_unity_circles:
        f.unity_circles()
    for q in (q_marks if q_marks is not None
              else ((1, 2, 3) if show_q_contours else ())):
        f.q_contour(q)
    f.vswr_circle(sm.vswr(gL), name="load VSWR")
    pts = [Z / z0 for _, Z, _ in traj]
    labels = [lab for lab, _, _ in traj[1:]]
    f.point(pts[0], "load", color=P["C_LOAD"], size=13)
    for i in range(len(pts) - 1):
        f.arc(pts[i], pts[i + 1], kind=traj[i + 1][2], name=labels[i])
    f.point(pts[-1], "input", color=P["C_TARGET"], size=13, symbol="star")
    f.point(1 + 0j, "matched", color=P["INK_MUTED"], size=8, symbol="x",
            showlegend=False)
    return f


def traj_with_arcs(net):
    """Network trajectory as (label, Z, arc_kind) triples."""
    t = net.trajectory(f0)
    out = [(t[0][0], t[0][1], "gamma")]
    for (lab, Z), el in zip(t[1:], net.elements):
        out.append((lab, Z, el.arc))
    return out


def elements_from_lmatch(m):
    """Turn an LMatch solution into concrete network elements."""
    els = []
    for role, kind, val, _ in m.components():
        cls = {("shunt", "L"): ShuntL, ("shunt", "C"): ShuntC,
               ("series", "L"): SeriesL, ("series", "C"): SeriesC}[
                   (role.split()[0], kind)]
        els.append(cls(val))
    return els


# ==========================================================================
# tabs
# ==========================================================================

tab_explore, tab_match, tab_build, tab_line = st.tabs(
    ["Explore", "Auto-match", "Build a network", "Line & stub calculator"])


# --- Explore --------------------------------------------------------------
with tab_explore:
    c1, c2 = st.columns([3, 2])
    with c1:
        f = SmithFigure(palette=P, z0=z0, title="the load, and where it goes with frequency")
        f.vswr_circle(sm.vswr(gL), name="VSWR at design freq")
        f.unity_circles()
        if load_kind != "Fixed Z":
            Zsw = load.Z(freqs)
            f.locus(Zsw / z0, "load vs frequency", color=P["SERIES"][0],
                    customdata=np.stack([
                        freqs / 1e9, Zsw.real, Zsw.imag,
                        sm.vswr(sm.gamma_from_Z(Zsw, z0))], axis=-1),
                    hovertemplate=("%{customdata[0]:.4f} GHz<br>"
                                   "Z = %{customdata[1]:.2f} %{customdata[2]:+.2f}j Ω"
                                   "<br>VSWR %{customdata[3]:.2f}<extra></extra>"))
        f.point(zL, f"load @ {f0_ghz:g} GHz", color=P["C_LOAD"], size=13)
        f.point(1 + 0j, "matched", color=P["C_TARGET"], size=10, symbol="x")
        st.plotly_chart(f.fig, width="stretch", config=PLOT_CFG,
                        key="explore-smith")
    with c2:
        st.subheader("At the design frequency")
        yL = 1 / zL
        st.dataframe(
            {
                "quantity": ["Z", "z (normalized)", "Y", "y (normalized)",
                             "Γ", "VSWR", "return loss", "mismatch loss",
                             "Q = |X|/R"],
                "value": [
                    f"{ZL.real:.3f} {ZL.imag:+.3f}j Ω",
                    f"{zL.real:.4f} {zL.imag:+.4f}j",
                    f"{(1/ZL).real*1e3:.4f} {(1/ZL).imag*1e3:+.4f}j mS",
                    f"{yL.real:.4f} {yL.imag:+.4f}j",
                    f"{abs(gL):.4f} ∠ {np.degrees(np.angle(gL)):.2f}°",
                    f"{sm.vswr(gL):.3f}",
                    f"{sm.return_loss_db(gL):.2f} dB",
                    f"{sm.mismatch_loss_db(gL):.3f} dB",
                    f"{sm.q_of_z(zL):.3f}",
                ],
            },
            hide_index=True, width="stretch",
        )
        st.caption("Hover any point on the chart for the same numbers there.")
        if load_kind != "Fixed Z":
            sweep_panel(Network(load=load, z0=z0), "bare load")


# --- Auto-match -----------------------------------------------------------
with tab_match:
    method = st.radio("Method", ["L-network (lumped)",
                                 "Single stub (distributed)",
                                 "Quarter-wave transformer"],
                      horizontal=True)

    if method == "L-network (lumped)":
        try:
            sols = l_match(ZL, z0, freq_hz=f0)
        except ValueError as e:
            st.error(str(e))
            sols = []
        if sols:
            rows = []
            for i, m in enumerate(sols):
                rows.append(f"[{i}] {m.topology}: "
                            + ", ".join(t for _, _, _, t in m.components())
                            + f"   (network Q {m.network_q:.2f})")
            pick = st.selectbox(
                "Solution", rows,
                help="Network Q is the Q at the corner the path turns at. It "
                     "tells you how hard the match is, but it is the same for "
                     "both variants of a topology — the sweep below is what "
                     "actually decides between them.")
            m = sols[rows.index(pick)]
            net = Network(load=load, elements=elements_from_lmatch(m), z0=z0)
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = chart_for_trajectory(traj_with_arcs(net),
                                           "matching path")
                st.plotly_chart(fig.fig, width="stretch",
                                config=PLOT_CFG, key="lmatch-smith")
            with c2:
                st.subheader("Bill of materials")
                for role, kind, val, txt in m.components():
                    st.write(f"**{role}** — {txt}")
                st.caption("Listed load-side first.")
                Zin = complex(net.Zin(f0))
                st.metric("Z at design frequency",
                          f"{Zin.real:.3f} {Zin.imag:+.3f}j Ω")
                sweep_panel(net, "matched response")

    elif method == "Single stub (distributed)":
        c1, c2, c3 = st.columns(3)
        kind = c1.selectbox("Stub termination", ["open", "short"])
        orient = c2.selectbox("Orientation", ["shunt", "series"])
        z0_stub = c3.number_input("Stub/line $Z_0$ (Ω)", 1.0, 1000.0, z0, 1.0)
        sols = single_stub_match(ZL, z0, stub_kind=kind, orientation=orient)
        if not sols:
            st.error("No stub solution found for this load.")
        else:
            rows = [f"[{i}] line {s_.d_wl:.4f} λ ({s_.d_wl*lam0*1000:.2f} mm), "
                    f"stub {s_.l_wl:.4f} λ ({s_.l_wl*lam0*1000:.2f} mm)"
                    for i, s_ in enumerate(sols)]
            pick = st.selectbox("Solution", rows)
            s_ = sols[rows.index(pick)]
            net = Network(load=load, z0=z0, elements=[
                Line(length_m=s_.d_wl * lam0, z0=z0, eps_eff=eps_eff),
                Stub(length_m=s_.l_wl * lam0, z0=z0_stub, eps_eff=eps_eff,
                     kind=kind, orientation=orient),
            ])
            c1, c2 = st.columns([3, 2])
            with c1:
                fig = chart_for_trajectory(traj_with_arcs(net), "stub tuner")
                st.plotly_chart(fig.fig, width="stretch",
                                config=PLOT_CFG, key="stub-smith")
            with c2:
                st.subheader("Cut list")
                st.write(f"**Line from load** — {s_.d_wl:.4f} λ  "
                         f"= {s_.d_wl * lam0 * 1000:.3f} mm")
                st.write(f"**{orient} {kind} stub** — {s_.l_wl:.4f} λ  "
                         f"= {s_.l_wl * lam0 * 1000:.3f} mm")
                y_at = 1 / s_.z_at_stub
                st.caption(f"At the stub, y = {y_at.real:.3f} "
                           f"{y_at.imag:+.3f}j — the stub cancels the "
                           f"{y_at.imag:+.3f}j.")
                Zin = complex(net.Zin(f0))
                st.metric("Z at design frequency",
                          f"{Zin.real:.3f} {Zin.imag:+.3f}j Ω")
                sweep_panel(net, "matched response")

    else:
        branch = st.radio("Which real-axis crossing", [0, 1], horizontal=True,
                          format_func=lambda b: f"branch {b}")
        qw = quarter_wave_match(ZL, z0, branch=branch)
        net = Network(load=load, z0=z0, elements=[
            Line(length_m=qw.d_wl * lam0, z0=z0, eps_eff=eps_eff),
            Line(length_m=0.25 * C0 / (f0 * np.sqrt(eps_eff)),
                 z0=qw.z0_transformer, eps_eff=eps_eff),
        ])
        c1, c2 = st.columns([3, 2])
        with c1:
            fig = chart_for_trajectory(traj_with_arcs(net), "quarter-wave")
            st.plotly_chart(fig.fig, width="stretch",
                            config=PLOT_CFG, key="qw-smith")
        with c2:
            st.subheader("Cut list")
            st.write(f"**Line from load** — {qw.d_wl:.4f} λ = "
                     f"{qw.d_wl * lam0 * 1000:.3f} mm")
            st.write(f"**λ/4 transformer** — {qw.z0_transformer:.2f} Ω, "
                     f"{0.25 * lam0 * 1000:.3f} mm")
            st.caption(f"The transformer sees a real {qw.R_real:.2f} Ω.")
            st.info("Inside the transformer the locus is a circle about "
                    f"{qw.z0_transformer:.1f} Ω, not about {z0:g} Ω — only its "
                    "endpoints are meaningful on this chart.")
            Zin = complex(net.Zin(f0))
            st.metric("Z at design frequency",
                      f"{Zin.real:.3f} {Zin.imag:+.3f}j Ω")
            sweep_panel(net, "matched response")


# --- Build a network ------------------------------------------------------
with tab_build:
    st.caption("Add elements one at a time, load-side first, and watch the "
               "path grow. This is the chart used the way it was designed "
               "to be used.")
    with st.form("add_element", clear_on_submit=False):
        c = st.columns([2, 2, 2, 2, 1])
        etype = c[0].selectbox("Element", [
            "series L", "series C", "series R",
            "shunt L", "shunt C", "shunt R",
            "line", "shunt open stub", "shunt short stub"])
        if etype.endswith(("L",)):
            val = c[1].number_input("nH", 0.0001, 1e6, 5.0, 0.1)
        elif etype.endswith("C"):
            val = c[1].number_input("pF", 0.0001, 1e6, 2.0, 0.01)
        elif etype.endswith("R"):
            val = c[1].number_input("Ω", 0.0001, 1e6, 50.0, 1.0)
        else:
            val = c[1].number_input("length (λ)", 0.0, 2.0, 0.125, 0.001,
                                    format="%.4f")
        el_z0 = c[2].number_input("element $Z_0$ (Ω)", 1.0, 1000.0, z0, 1.0,
                                  disabled=etype in (
                                      "series L", "series C", "series R",
                                      "shunt L", "shunt C", "shunt R"))
        c[3].markdown("&nbsp;", unsafe_allow_html=True)
        added = c[4].form_submit_button("Add", width="stretch")

    if added:
        mk = {
            "series L": lambda: SeriesL(val * 1e-9),
            "series C": lambda: SeriesC(val * 1e-12),
            "series R": lambda: SeriesR(val),
            "shunt L": lambda: ShuntL(val * 1e-9),
            "shunt C": lambda: ShuntC(val * 1e-12),
            "shunt R": lambda: ShuntR(val),
            "line": lambda: Line(val * lam0, el_z0, eps_eff),
            "shunt open stub": lambda: Stub(val * lam0, el_z0, eps_eff,
                                            "open", "shunt"),
            "shunt short stub": lambda: Stub(val * lam0, el_z0, eps_eff,
                                             "short", "shunt"),
        }[etype]
        st.session_state.elements.append(mk())

    cc = st.columns([1, 1, 6])
    if cc[0].button("Undo last", disabled=not st.session_state.elements):
        st.session_state.elements.pop()
    cc[1].button("Clear all", on_click=reset_elements,
                 disabled=not st.session_state.elements)

    net = Network(load=load, elements=list(st.session_state.elements), z0=z0)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = chart_for_trajectory(traj_with_arcs(net), "network path")
        st.plotly_chart(fig.fig, width="stretch", config=PLOT_CFG,
                        key="build-smith")
    with c2:
        st.subheader("Chain, load-side first")
        traj = net.trajectory(f0)
        st.dataframe(
            {
                "step": [lab for lab, _ in traj],
                "Z (Ω)": [f"{Z.real:.3f} {Z.imag:+.3f}j" for _, Z in traj],
                "VSWR": [f"{sm.vswr(sm.gamma_from_Z(Z, z0)):.3f}"
                         for _, Z in traj],
            },
            hide_index=True, width="stretch",
        )
        if st.session_state.elements:
            sweep_panel(net, "network response")


# --- Line & stub calculator ----------------------------------------------
with tab_line:
    c1, c2 = st.columns([3, 2])
    with c2:
        st.subheader("Move along a line")
        d_wl = st.slider("length (wavelengths)", 0.0, 1.0, 0.25, 0.001)
        loss = st.slider("loss (dB per wavelength)", 0.0, 3.0, 0.0, 0.01)
        z0_line = st.number_input("line $Z_0$ (Ω)", 1.0, 1000.0, z0, 1.0,
                                  key="lc_z0")
        alpha_np = loss / 8.685889638
        if abs(z0_line - z0) < 1e-9:
            z_in = sm.z_from_gamma(sm.rotate_gamma(gL, d_wl, "generator",
                                                   alpha_np))
            Zin = z_in * z0
        else:
            net_l = Network(load=load, z0=z0, elements=[
                Line(d_wl * lam0, z0_line, eps_eff,
                     alpha_db_per_m=loss / lam0)])
            Zin = complex(net_l.Zin(f0))
            z_in = Zin / z0
        st.metric("Input impedance",
                  f"{Zin.real:.3f} {Zin.imag:+.3f}j Ω")
        gi = sm.gamma_from_z(z_in)
        st.write(f"|Γ| = {abs(gi):.4f}, VSWR = {sm.vswr(gi):.3f}, "
                 f"RL = {sm.return_loss_db(gi):.2f} dB")
        st.caption(f"{d_wl:.4f} λ = {d_wl * lam0 * 1000:.3f} mm "
                   f"at {f0_ghz:g} GHz with εeff = {eps_eff:g}")

        st.divider()
        st.subheader("Stub reactance")
        sk = st.selectbox("Termination", ["open", "short"], key="lc_stub")
        sl = st.slider("stub length (λ)", 0.0, 0.5, 0.125, 0.001)
        zs = sm.stub_input_z(sl, sk)
        if np.isfinite(zs.imag):
            st.metric("Stub impedance", f"j{zs.imag * z0:+.3f} Ω")
            st.write(f"normalized  z = j{zs.imag:+.4f},  "
                     f"y = j{(1/zs).imag if abs(zs) > 1e-12 else float('inf'):+.4f}")
        st.caption(f"{sl:.4f} λ = {sl * lam0 * 1000:.3f} mm")

    with c1:
        f = SmithFigure(palette=P, z0=z0, title="rotation along the line")
        f.vswr_circle(sm.vswr(gL), name="lossless VSWR")
        d = np.linspace(0, max(d_wl, 1e-6), 400)
        gtrace = gL * np.exp(-2 * alpha_np * d) * np.exp(-1j * 4 * np.pi * d)
        f.locus(sm.z_from_gamma(gtrace), "along the line", color=P["C_LINE"],
                arrows=3,
                customdata=np.stack([d, d * lam0 * 1000], axis=-1),
                hovertemplate=("%{customdata[0]:.4f} λ  "
                               "(%{customdata[1]:.2f} mm)<extra></extra>"))
        f.point(zL, "load", color=P["C_LOAD"], size=13)
        f.point(z_in, "input", color=P["C_TARGET"], size=13, symbol="star")
        f.point(1 + 0j, "matched", color=P["INK_MUTED"], size=8, symbol="x",
                showlegend=False)
        st.plotly_chart(f.fig, width="stretch", config=PLOT_CFG,
                        key="line-smith")

st.divider()
st.caption("Built on `smithlib` — the maths is plain numpy in "
           "`smithlib/core.py`, `tline.py`, `matching.py` and `network.py`, "
           "with no dependency on this UI. See `docs/smith-chart.qmd` for the "
           "explanation of every move drawn here.")
