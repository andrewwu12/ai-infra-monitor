"""
AI Infrastructure Capex Cycle Monitor
Streamlit Dashboard

Reads data from:
  - data/latest.json   (current snapshot)
  - data/history.jsonl  (historical time series)
"""

import json
import pathlib
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = pathlib.Path(__file__).parent / "data"
LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.jsonl"

PHASE_DISPLAY = {
    "acceleration": "Acceleration",
    "peak_intensity": "Peak Intensity",
    "digestion": "Digestion",
    "contraction": "Contraction",
}

PHASE_COLORS = {
    "acceleration": "#4fc3f7",
    "peak_intensity": "#ef5350",
    "digestion": "#ffa726",
    "contraction": "#78909c",
}

LAYER_META = {
    "capex_acceleration": {
        "name": "Hyperscaler Capex Intensity",
        "subtitle": "MSFT, AMZN, GOOG, META",
        "icon": "1",
    },
    "bottleneck_power": {
        "name": "Infrastructure Supplier Tightness",
        "subtitle": "NVDA + supply chain",
        "icon": "2",
    },
    "monetization": {
        "name": "Monetization Signals",
        "subtitle": "Cloud growth, AI revenue",
        "icon": "3",
    },
    "macro_constraint": {
        "name": "Macro Stress Valve",
        "subtitle": "Yields, spreads, labor",
        "icon": "4",
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_latest() -> dict | None:
    if not LATEST_FILE.exists():
        return None
    with open(LATEST_FILE) as f:
        return json.load(f)


def load_history() -> pd.DataFrame | None:
    if not HISTORY_FILE.exists():
        return None
    records = []
    with open(HISTORY_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helper: score color
# ---------------------------------------------------------------------------


def score_color(score: float) -> str:
    if score > 70:
        return "#ef5350"
    if score > 50:
        return "#ffa726"
    if score > 30:
        return "#4fc3f7"
    return "#78909c"


def score_label(score: float) -> str:
    if score > 70:
        return "Hot"
    if score > 50:
        return "Warm"
    if score > 30:
        return "Cool"
    return "Cold"


# ---------------------------------------------------------------------------
# Gauge chart for Heat Index
# ---------------------------------------------------------------------------


def make_heat_gauge(value: float) -> go.Figure:
    color = score_color(value)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"size": 54, "color": color}, "suffix": ""},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#555",
                    "dtick": 10,
                    "tickfont": {"color": "#888", "size": 10},
                },
                "bar": {"color": color, "thickness": 0.6},
                "bgcolor": "#1a1d23",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(120,144,156,0.15)"},
                    {"range": [30, 50], "color": "rgba(79,195,247,0.15)"},
                    {"range": [50, 70], "color": "rgba(255,167,38,0.15)"},
                    {"range": [70, 100], "color": "rgba(239,83,80,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    return fig


# ---------------------------------------------------------------------------
# Phase probability bars
# ---------------------------------------------------------------------------


def make_phase_bars(probs: dict) -> go.Figure:
    phases = list(probs.keys())
    values = [probs[p] * 100 for p in phases]
    labels = [PHASE_DISPLAY.get(p, p) for p in phases]
    colors = [PHASE_COLORS.get(p, "#4fc3f7") for p in phases]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}%" for v in values],
            textposition="auto",
            textfont={"color": "white", "size": 13},
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont={"size": 12, "color": "#ccc"},
        ),
        bargap=0.35,
    )
    return fig


# ---------------------------------------------------------------------------
# Yield curve chart
# ---------------------------------------------------------------------------


def make_yield_curve(yc_data: dict) -> go.Figure:
    maturities = yc_data.get("maturities", {})
    if not maturities:
        return None

    # Sort by numeric maturity (handle labels like "3mo", "1yr", "2yr", etc.)
    def sort_key(label: str) -> float:
        label_lower = label.lower().replace(" ", "")
        if "mo" in label_lower:
            num = float(
                "".join(
                    c for c in label_lower.replace("mo", "") if c.isdigit() or c == "."
                )
                or "0"
            )
            return num / 12.0
        if "yr" in label_lower or "y" in label_lower:
            cleaned = label_lower.replace("yr", "").replace("y", "")
            num = float("".join(c for c in cleaned if c.isdigit() or c == ".") or "0")
            return num
        try:
            return float(label)
        except ValueError:
            return 0.0

    sorted_items = sorted(maturities.items(), key=lambda x: sort_key(x[0]))
    labels = [item[0] for item in sorted_items]
    yields = [item[1] for item in sorted_items]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=yields,
            mode="lines+markers",
            line=dict(color="#4fc3f7", width=2.5),
            marker=dict(size=7, color="#4fc3f7"),
            fill="tozeroy",
            fillcolor="rgba(79,195,247,0.08)",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Maturity",
            gridcolor="rgba(255,255,255,0.05)",
            tickfont={"color": "#aaa"},
            titlefont={"color": "#aaa"},
        ),
        yaxis=dict(
            title="Yield (%)",
            gridcolor="rgba(255,255,255,0.08)",
            tickfont={"color": "#aaa"},
            titlefont={"color": "#aaa"},
            ticksuffix="%",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Historical charts
# ---------------------------------------------------------------------------


def make_history_heat_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["heat_index"],
            mode="lines+markers",
            line=dict(color="#4fc3f7", width=2),
            marker=dict(size=5),
            name="Heat Index",
            fill="tozeroy",
            fillcolor="rgba(79,195,247,0.07)",
        )
    )

    # Color zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor="rgba(255,167,38,0.04)", line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor="rgba(79,195,247,0.04)", line_width=0)

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.08)",
            tickfont={"color": "#aaa"},
            title="Heat Index",
            titlefont={"color": "#aaa"},
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            tickfont={"color": "#aaa"},
        ),
        showlegend=False,
    )
    return fig


def make_history_layers_chart(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    layer_colors = {
        "capex_acceleration": "#4fc3f7",
        "bottleneck_power": "#ffa726",
        "monetization": "#66bb6a",
        "macro_constraint": "#ef5350",
    }

    fig = go.Figure()
    for layer_key, meta in LAYER_META.items():
        scores = []
        for _, row in df.iterrows():
            layers = row.get("layers", {})
            if isinstance(layers, str):
                layers = json.loads(layers)
            layer_data = layers.get(layer_key, {})
            scores.append(layer_data.get("score", None))

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=scores,
                mode="lines+markers",
                name=meta["name"],
                line=dict(color=layer_colors.get(layer_key, "#888"), width=2),
                marker=dict(size=4),
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.08)",
            tickfont={"color": "#aaa"},
            title="Layer Score",
            titlefont={"color": "#aaa"},
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            tickfont={"color": "#aaa"},
        ),
        legend=dict(
            font={"color": "#ccc", "size": 11},
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Layer card renderer
# ---------------------------------------------------------------------------


def render_layer_card(layer_key: str, layer_data: dict):
    meta = LAYER_META.get(layer_key, {"name": layer_key, "subtitle": "", "icon": "?"})
    score = layer_data.get("score", 0)
    details = layer_data.get("details", {})
    interpretation = layer_data.get("interpretation", "")
    color = score_color(score)

    st.markdown(
        f"""
        <div style="
            border: 1px solid {color}33;
            border-radius: 8px;
            padding: 16px 18px 12px 18px;
            background: linear-gradient(135deg, {color}08 0%, rgba(0,0,0,0) 60%);
            margin-bottom: 8px;
            min-height: 200px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div>
                    <span style="color:#e0e0e0; font-size:15px; font-weight:600;">
                        Layer {meta['icon']}: {meta['name']}
                    </span><br/>
                    <span style="color:#888; font-size:12px;">{meta['subtitle']}</span>
                </div>
                <div style="
                    background: {color}22;
                    border: 1px solid {color}55;
                    border-radius: 6px;
                    padding: 4px 14px;
                    text-align:center;
                ">
                    <span style="color:{color}; font-size:24px; font-weight:700;">{score}</span>
                    <span style="color:{color}; font-size:11px; display:block; margin-top:-4px;">{score_label(score)}</span>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Details table
    if details:
        rows_html = ""
        for k, v in details.items():
            display_key = k.replace("_", " ").title()
            if isinstance(v, float):
                display_val = f"{v:.2f}" if abs(v) < 100 else f"{v:,.0f}"
            elif isinstance(v, (int,)):
                display_val = f"{v:,}"
            else:
                display_val = str(v)
            rows_html += f"""
                <tr>
                    <td style="color:#aaa; padding:2px 8px 2px 0; font-size:12px; border:none;">{display_key}</td>
                    <td style="color:#e0e0e0; padding:2px 0; font-size:12px; font-weight:500; border:none; text-align:right;">{display_val}</td>
                </tr>
            """
        st.markdown(
            f'<table style="width:100%; border-collapse:collapse; margin-top:4px;">{rows_html}</table>',
            unsafe_allow_html=True,
        )

    if interpretation:
        st.markdown(
            f'<div style="color:#aaa; font-size:11px; font-style:italic; margin-top:8px; border-top:1px solid #333; padding-top:6px;">{interpretation}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Signal list renderer
# ---------------------------------------------------------------------------


def render_signal_list(items: list, color: str = "#4fc3f7"):
    if not items:
        st.caption("No data available.")
        return
    for item in items:
        if isinstance(item, dict):
            title = item.get("title", item.get("signal", item.get("name", str(item))))
            desc = item.get("description", item.get("detail", item.get("desc", "")))
            severity = item.get("severity", item.get("level", ""))
            severity_html = ""
            if severity:
                sev_color = {
                    "high": "#ef5350",
                    "medium": "#ffa726",
                    "low": "#66bb6a",
                }.get(severity.lower(), "#888")
                severity_html = f' <span style="color:{sev_color}; font-size:11px; font-weight:600;">[{severity.upper()}]</span>'
            st.markdown(
                f"""<div style="padding:6px 0; border-bottom:1px solid #222;">
                    <span style="color:{color}; font-size:13px; font-weight:500;">{title}</span>{severity_html}
                    <br/><span style="color:#999; font-size:12px;">{desc}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div style="padding:4px 0; border-bottom:1px solid #222;">
                    <span style="color:#ccc; font-size:13px;">{item}</span>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Infra Capex Monitor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown(
    """
    <style>
        /* Remove default padding */
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

        /* Metric styling */
        [data-testid="stMetricValue"] { font-size: 1.1rem; }
        [data-testid="stMetricDelta"] { font-size: 0.85rem; }

        /* Expander styling */
        .streamlit-expanderHeader { font-size: 14px; font-weight: 600; }

        /* Hide the hamburger menu */
        #MainMenu { visibility: hidden; }

        /* Section dividers */
        hr { border-color: #333; }

        /* Subtle card-like containers */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    data = load_latest()

    if data is None:
        st.markdown(
            """
            <div style="text-align:center; padding:60px 20px;">
                <h1 style="color:#4fc3f7; margin-bottom:8px;">AI Infrastructure Capex Cycle Monitor</h1>
                <p style="color:#888; font-size:16px; margin-bottom:30px;">Waiting for data...</p>
                <div style="
                    border:1px dashed #444;
                    border-radius:12px;
                    padding:40px;
                    max-width:600px;
                    margin:0 auto;
                    background:#1a1d23;
                ">
                    <p style="color:#aaa; font-size:14px;">
                        No data file found at:<br/>
                        <code style="color:#4fc3f7;">data/latest.json</code>
                    </p>
                    <p style="color:#666; font-size:13px; margin-top:16px;">
                        The dashboard will populate automatically once the data pipeline
                        writes its first snapshot.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    timestamp_str = data.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(timestamp_str)
        display_ts = ts.strftime("%b %d, %Y  %H:%M UTC")
    except (ValueError, TypeError):
        display_ts = timestamp_str or "Unknown"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:4px;">
            <h1 style="color:#e0e0e0; font-size:28px; margin:0; font-weight:700;">
                AI Infrastructure Capex Cycle Monitor
            </h1>
            <span style="color:#666; font-size:13px;">Last updated: {display_ts}</span>
        </div>
        <hr style="margin-top:8px; margin-bottom:20px; border-color:#2a2a2a;"/>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Top Row: Hero Metrics
    # ------------------------------------------------------------------
    heat_index = data.get("heat_index", 0)
    phase_probs = data.get("phase_probabilities", {})
    current_phase = max(phase_probs, key=phase_probs.get) if phase_probs else "unknown"
    current_phase_pct = phase_probs.get(current_phase, 0) * 100

    col_gauge, col_phase, col_bars = st.columns([1.2, 0.8, 1.5])

    with col_gauge:
        st.markdown(
            '<p style="color:#888; font-size:13px; margin-bottom:2px; font-weight:600; letter-spacing:0.5px;">AI INFRA HEAT INDEX</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_heat_gauge(heat_index),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with col_phase:
        phase_color = PHASE_COLORS.get(current_phase, "#4fc3f7")
        st.markdown(
            f"""
            <div style="padding-top:12px;">
                <p style="color:#888; font-size:13px; margin-bottom:12px; font-weight:600; letter-spacing:0.5px;">CURRENT PHASE</p>
                <div style="
                    background: {phase_color}15;
                    border: 1px solid {phase_color}44;
                    border-radius: 10px;
                    padding: 20px 16px;
                    text-align: center;
                ">
                    <span style="color:{phase_color}; font-size:26px; font-weight:700;">
                        {PHASE_DISPLAY.get(current_phase, current_phase)}
                    </span><br/>
                    <span style="color:#aaa; font-size:14px;">{current_phase_pct:.0f}% probability</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_bars:
        st.markdown(
            '<p style="color:#888; font-size:13px; margin-bottom:2px; font-weight:600; letter-spacing:0.5px;">PHASE PROBABILITIES</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_phase_bars(phase_probs),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        '<hr style="margin:12px 0; border-color:#2a2a2a;"/>', unsafe_allow_html=True
    )

    # ------------------------------------------------------------------
    # 4-Layer Breakdown (2x2)
    # ------------------------------------------------------------------
    st.markdown(
        '<p style="color:#888; font-size:14px; font-weight:600; letter-spacing:0.5px; margin-bottom:12px;">LAYER BREAKDOWN</p>',
        unsafe_allow_html=True,
    )

    layers = data.get("layers", {})
    layer_keys = list(LAYER_META.keys())

    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)

    grid_positions = [row1_left, row1_right, row2_left, row2_right]

    for i, key in enumerate(layer_keys):
        with grid_positions[i]:
            render_layer_card(key, layers.get(key, {"score": 0, "details": {}}))

    st.markdown(
        '<hr style="margin:16px 0; border-color:#2a2a2a;"/>', unsafe_allow_html=True
    )

    # ------------------------------------------------------------------
    # Yield Curve
    # ------------------------------------------------------------------
    yc_data = data.get("yield_curve", {})
    if yc_data and yc_data.get("maturities"):
        st.markdown(
            '<p style="color:#888; font-size:14px; font-weight:600; letter-spacing:0.5px; margin-bottom:12px;">YIELD CURVE</p>',
            unsafe_allow_html=True,
        )

        yc_col1, yc_col2 = st.columns([2.5, 1])

        with yc_col1:
            fig = make_yield_curve(yc_data)
            if fig:
                yc_date = yc_data.get("date", "")
                if yc_date:
                    st.caption(f"As of {yc_date}")
                st.plotly_chart(
                    fig, use_container_width=True, config={"displayModeBar": False}
                )

        with yc_col2:
            spreads = yc_data.get("spreads", {})
            if spreads:
                st.markdown(
                    '<p style="color:#aaa; font-size:13px; font-weight:600; margin-bottom:8px;">Key Spreads</p>',
                    unsafe_allow_html=True,
                )
                for spread_name, spread_val in spreads.items():
                    display_name = (
                        spread_name.replace("_", " ")
                        .replace("2s10s", "2s10s")
                        .replace("3mo10yr", "3mo10yr")
                    )
                    if isinstance(spread_val, (int, float)):
                        spread_bps = (
                            spread_val * 100 if abs(spread_val) < 10 else spread_val
                        )
                        val_color = "#ef5350" if spread_bps < 0 else "#66bb6a"
                        st.markdown(
                            f"""
                            <div style="
                                background:#1a1d23;
                                border:1px solid #333;
                                border-radius:6px;
                                padding:12px;
                                margin-bottom:8px;
                            ">
                                <span style="color:#888; font-size:12px; display:block;">{display_name}</span>
                                <span style="color:{val_color}; font-size:22px; font-weight:700;">{spread_bps:+.0f} bps</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.metric(display_name, str(spread_val))

        st.markdown(
            '<hr style="margin:16px 0; border-color:#2a2a2a;"/>', unsafe_allow_html=True
        )

    # ------------------------------------------------------------------
    # Key Drivers / Stress Signals / Early Warnings
    # ------------------------------------------------------------------
    st.markdown(
        '<p style="color:#888; font-size:14px; font-weight:600; letter-spacing:0.5px; margin-bottom:12px;">SIGNALS & DRIVERS</p>',
        unsafe_allow_html=True,
    )

    sig_col1, sig_col2, sig_col3 = st.columns(3)

    with sig_col1:
        with st.expander("Key Drivers", expanded=True):
            render_signal_list(data.get("key_drivers", []), color="#4fc3f7")

    with sig_col2:
        with st.expander("Stress Signals", expanded=True):
            render_signal_list(data.get("stress_signals", []), color="#ef5350")

    with sig_col3:
        with st.expander("Early Warnings", expanded=True):
            render_signal_list(data.get("early_warnings", []), color="#ffa726")

    st.markdown(
        '<hr style="margin:16px 0; border-color:#2a2a2a;"/>', unsafe_allow_html=True
    )

    # ------------------------------------------------------------------
    # Historical Trends
    # ------------------------------------------------------------------
    history_df = load_history()

    if history_df is not None and len(history_df) > 1:
        st.markdown(
            '<p style="color:#888; font-size:14px; font-weight:600; letter-spacing:0.5px; margin-bottom:12px;">HISTORICAL TRENDS</p>',
            unsafe_allow_html=True,
        )

        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            st.markdown(
                '<p style="color:#aaa; font-size:12px; margin-bottom:4px;">Heat Index Over Time</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                make_history_heat_chart(history_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        with hist_col2:
            st.markdown(
                '<p style="color:#aaa; font-size:12px; margin-bottom:4px;">Layer Scores Over Time</p>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                make_history_layers_chart(history_df),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown(
            '<hr style="margin:16px 0; border-color:#2a2a2a;"/>', unsafe_allow_html=True
        )

    # ------------------------------------------------------------------
    # Outlook
    # ------------------------------------------------------------------
    outlook = data.get("outlook_6_12m", {})
    if outlook:
        st.markdown(
            '<p style="color:#888; font-size:14px; font-weight:600; letter-spacing:0.5px; margin-bottom:12px;">6-12 MONTH OUTLOOK</p>',
            unsafe_allow_html=True,
        )

        summary = outlook.get("summary", "")
        scenarios = outlook.get("scenarios", [])
        base_case = outlook.get("base_case", "")
        risks = outlook.get("risks", [])

        if summary:
            st.markdown(
                f'<p style="color:#ccc; font-size:14px; line-height:1.6; margin-bottom:16px;">{summary}</p>',
                unsafe_allow_html=True,
            )

        if base_case and not scenarios:
            st.markdown(
                f"""
                <div style="
                    background:#1a1d23;
                    border-left:3px solid #4fc3f7;
                    padding:12px 16px;
                    border-radius:4px;
                    margin-bottom:12px;
                ">
                    <span style="color:#888; font-size:12px; font-weight:600;">BASE CASE</span><br/>
                    <span style="color:#e0e0e0; font-size:13px;">{base_case}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if scenarios:
            scenario_cols = st.columns(len(scenarios))
            for idx, scenario in enumerate(scenarios):
                with scenario_cols[idx]:
                    s_name = scenario.get(
                        "name", scenario.get("scenario", f"Scenario {idx+1}")
                    )
                    s_prob = scenario.get("probability", 0)
                    s_desc = scenario.get("description", scenario.get("detail", ""))

                    if isinstance(s_prob, float) and s_prob <= 1:
                        s_prob_display = f"{s_prob * 100:.0f}%"
                    else:
                        s_prob_display = f"{s_prob}%"

                    # Pick border color based on index
                    s_colors = ["#4fc3f7", "#ffa726", "#ef5350", "#66bb6a"]
                    s_color = s_colors[idx % len(s_colors)]

                    st.markdown(
                        f"""
                        <div style="
                            background:#1a1d23;
                            border:1px solid {s_color}44;
                            border-top:3px solid {s_color};
                            padding:14px;
                            border-radius:6px;
                            min-height:120px;
                        ">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                                <span style="color:#e0e0e0; font-size:14px; font-weight:600;">{s_name}</span>
                                <span style="color:{s_color}; font-size:18px; font-weight:700;">{s_prob_display}</span>
                            </div>
                            <p style="color:#999; font-size:12px; line-height:1.5;">{s_desc}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if risks:
            with st.expander("Key Risks"):
                for risk in risks:
                    if isinstance(risk, dict):
                        r_name = risk.get("name", risk.get("risk", str(risk)))
                        r_desc = risk.get("description", risk.get("detail", ""))
                        st.markdown(
                            f"""<div style="padding:4px 0; border-bottom:1px solid #222;">
                                <span style="color:#ef5350; font-size:13px; font-weight:500;">{r_name}</span><br/>
                                <span style="color:#999; font-size:12px;">{r_desc}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"- {risk}")

    # Footer
    st.markdown(
        """
        <div style="text-align:center; padding:24px 0 8px 0; border-top:1px solid #222; margin-top:24px;">
            <span style="color:#555; font-size:11px;">
                AI Infrastructure Capex Cycle Monitor | Data refreshed periodically | For informational purposes only
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
