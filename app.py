import streamlit as st
import pandas as pd
import os
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from math_engine import calculate_physics_hazard_score, get_partial_gravity_uncertainty, train_ai_model

st.set_page_config(
    page_title="EigenMind | NASA Microgravity Fire Safety SI Dashboard",
    page_icon="🔥",
    layout="wide"
)

# ============================================================
# EIGENMIND | MISSION OPERATIONS CENTER VISUAL THEME
# Styling only: backend, calculations, data, controls and tabs
# remain unchanged.
# ============================================================

st.markdown("""
<style>
    :root {
        --mc-bg: #0b111a;
        --mc-panel: #172331;
        --mc-panel-dark: #101b28;
        --mc-panel-light: #213246;
        --mc-border: #34495d;
        --mc-blue: #58b9ed;
        --mc-cyan: #55d8e8;
        --mc-amber: #f2bd55;
        --mc-red: #ef6868;
        --mc-green: #63d39b;
        --mc-text: #e8f0f7;
        --mc-muted: #9eb0c1;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% -15%, rgba(53, 104, 145, 0.20), transparent 55%),
            linear-gradient(180deg, #101923 0%, #0b111a 58%, #080e15 100%);
        color: var(--mc-text);
        font-family: "Segoe UI", "Roboto", sans-serif;
    }

    .block-container {
        max-width: 98%;
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
    }

    h1 {
        color: #f4f8fc !important;
        font-size: clamp(1.8rem, 3vw, 2.65rem) !important;
        font-weight: 800 !important;
        letter-spacing: 0.045em !important;
        text-transform: uppercase;
        text-shadow: 0 2px 14px rgba(0, 0, 0, 0.65);
        border-bottom: 1px solid #49647b;
        padding-bottom: 0.75rem;
    }

    h2, h3 {
        color: #c8e8f8 !important;
        font-weight: 700 !important;
        letter-spacing: 0.025em;
    }

    h2 {
        border-bottom: 1px solid rgba(104, 151, 184, 0.28);
        padding-bottom: 0.35rem;
    }

    p, li, label, .stMarkdown {
        color: var(--mc-text);
    }

    [data-testid="stCaptionContainer"] {
        color: var(--mc-muted) !important;
    }

    /* Sidebar: Mission Operations Console */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #172433 0%, #101a26 100%);
        border-right: 1px solid #40566a;
        box-shadow: 8px 0 24px rgba(0, 0, 0, 0.30);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #b9e5fa !important;
        text-transform: uppercase;
        letter-spacing: 0.055em;
        font-size: 1rem !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #3b5267;
    }

    /* Mission Telemetry KPI Panels */
    div[data-testid="stMetric"] {
        position: relative;
        min-height: 118px;
        background:
            linear-gradient(145deg, rgba(34, 53, 72, 0.98), rgba(20, 33, 47, 0.98));
        border: 1px solid #405a70;
        border-top: 3px solid #5aa9d6;
        border-radius: 5px;
        padding: 15px 17px;
        box-shadow:
            0 8px 20px rgba(0, 0, 0, 0.28),
            inset 0 1px 0 rgba(255, 255, 255, 0.07);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #6bbfe9;
        box-shadow:
            0 12px 24px rgba(0, 0, 0, 0.38),
            0 0 12px rgba(88, 185, 237, 0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #a9c1d3 !important;
        font-size: 0.73rem !important;
        font-weight: 750 !important;
        letter-spacing: 0.085em;
        text-transform: uppercase;
    }

    div[data-testid="stMetricValue"] {
        color: #f5f9fc !important;
        font-family: "Consolas", "Courier New", monospace;
        font-size: clamp(1.45rem, 2.2vw, 2rem) !important;
        font-weight: 800 !important;
        text-shadow: 0 0 12px rgba(88, 185, 237, 0.14);
    }

    div[data-testid="stMetricDelta"] {
        font-family: "Consolas", "Courier New", monospace;
        font-size: 0.78rem !important;
    }

    /* Mission Control Tabs */
    div[data-baseweb="tab-list"] {
        gap: 5px;
        background: #101a26;
        padding: 6px 6px 0 6px;
        border: 1px solid #354b60;
        border-radius: 5px 5px 0 0;
    }

    button[data-baseweb="tab"] {
        background: #1a2a3a !important;
        color: #b4c6d5 !important;
        border: 1px solid #3a5268 !important;
        border-bottom: 0 !important;
        border-radius: 4px 4px 0 0 !important;
        padding: 10px 15px !important;
        font-weight: 700 !important;
        letter-spacing: 0.015em;
        transition: background 160ms ease, color 160ms ease;
    }

    button[data-baseweb="tab"]:hover {
        background: #263d52 !important;
        color: #ffffff !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #24445e !important;
        color: #ffffff !important;
        border-color: #63b9e7 !important;
        border-top: 3px solid #69c5ee !important;
        box-shadow: inset 0 -2px 0 #24445e, 0 -3px 12px rgba(88, 185, 237, 0.10);
    }

    /* Recessed Telemetry Chart Monitors */
    div[data-testid="stPlotlyChart"] {
        background: linear-gradient(145deg, #111d2a, #0d1722);
        border: 1px solid #3a5268;
        border-top: 2px solid #4f88ad;
        border-radius: 5px;
        padding: 8px;
        box-shadow:
            0 10px 24px rgba(0, 0, 0, 0.30),
            inset 0 1px 0 rgba(255, 255, 255, 0.045);
    }

    /* Data Tables */
    div[data-testid="stDataFrame"] {
        background: #111d29;
        border: 1px solid #3b5267;
        border-radius: 5px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }

    /* Expanders and Information Panels */
    div[data-testid="stExpander"] {
        background: #142231;
        border: 1px solid #3a5268;
        border-radius: 5px;
    }

    div[data-testid="stExpander"] summary {
        color: #c6e8f8 !important;
        font-weight: 700;
    }

    div[data-testid="stAlert"] {
        border-radius: 5px;
        border: 1px solid #405a70;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.20);
    }

    div[data-testid="stAlert"][kind="success"] {
        border-left: 4px solid var(--mc-green);
    }

    div[data-testid="stAlert"][kind="warning"] {
        border-left: 4px solid var(--mc-amber);
    }

    div[data-testid="stAlert"][kind="error"] {
        border-left: 4px solid var(--mc-red);
    }

    /* Environmental Selectors and Text Inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div,
    .stNumberInput > div > div {
        background: #101d2a !important;
        border: 1px solid #405970 !important;
        border-radius: 4px !important;
        color: #edf6fc !important;
    }

    div[data-baseweb="select"] span {
        color: #edf6fc !important;
    }

    .stSelectbox label,
    .stSlider label,
    .stRadio label,
    .stTextInput label {
        color: #b8d1e2 !important;
        font-weight: 650 !important;
    }

    div[data-testid="stSlider"] [role="slider"] {
        background: #65bce8 !important;
        border: 2px solid #eaf7ff !important;
        box-shadow: 0 0 8px rgba(88, 185, 237, 0.38);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(180deg, #315d7d, #23445f);
        border: 1px solid #5b8fb1;
        border-radius: 4px;
        color: #f4f8fc;
        font-weight: 700;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 160ms ease, background 160ms ease;
    }

    .stButton > button:hover {
        background: linear-gradient(180deg, #3e7398, #2a5575);
        color: #ffffff;
        border-color: #8acdf0;
        transform: translateY(-2px);
    }

    hr {
        border-color: #354b60 !important;
    }

    code {
        color: #9ce2ff !important;
        background: #142b3d !important;
        border: 1px solid #35566e;
        border-radius: 3px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 9px;
        height: 9px;
    }

    ::-webkit-scrollbar-track {
        background: #0a121c;
    }

    ::-webkit-scrollbar-thumb {
        background: #405c73;
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #6398b8;
    }

    /* Responsive Layout */
    @media screen and (max-width: 768px) {
        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        h1 {
            font-size: 1.55rem !important;
        }

        button[data-baseweb="tab"] {
            padding: 8px 9px !important;
            font-size: 0.72rem;
        }

        div[data-testid="stMetric"] {
            min-height: 100px;
            padding: 11px;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# Auto-build dataset & model if not yet generated
try:
    base_df = pd.read_csv("nasa_combustion_data.csv")
    grid_df = pd.read_csv("nasa_parametric_grid.csv")
    model = joblib.load("eigenmind_fire_model.pkl")
except FileNotFoundError:
    train_ai_model()
    base_df = pd.read_csv("nasa_combustion_data.csv")
    grid_df = pd.read_csv("nasa_parametric_grid.csv")
    model = joblib.load("eigenmind_fire_model.pkl")

st.title("🔥 EigenMind: SI-Powered Microgravity Fire Safety Dashboard")
st.markdown(
    "**2026 NASA Space Apps Challenge — Flame in Freefall** | "
    "Mathematical & Super-Intelligence (SI) Synthesis of NASA Physical Sciences Informatics (PSI) Data"
)

# Sidebar: Mission Presets & Environmental Controls

# Sidebar: Mission Presets & Environmental Controls
st.sidebar.header("🚀 Mission Environment Presets")

preset = st.sidebar.selectbox(
    "Select Spacecraft / Habitat Profile:",
    [
        "Custom Simulation",
        "ISS Cabin Standard (21% O2, 14.7 psi, 0g)",
        "Moon Base / Pressurized Rover (30% O2, 10.2 psi, 1/6g)",
        "Mars Habitat Concept (26% O2, 11.6 psi, 3/8g)",
        "Illustrative EVA-Related Oxygen Scenario (85% O2, 4.3 psi, 0g)"
    ]
)

if preset == "ISS Cabin Standard (21% O2, 14.7 psi, 0g)":
    def_o2, def_p, def_flow, def_dir, def_g = 21.0, 101.3, 15.0, "Concurrent", 0.0

elif preset == "Moon Base / Pressurized Rover (30% O2, 10.2 psi, 1/6g)":
    def_o2, def_p, def_flow, def_dir, def_g = 30.0, 70.3, 12.0, "Concurrent", 0.165

elif preset == "Mars Habitat Concept (26% O2, 11.6 psi, 3/8g)":
    def_o2, def_p, def_flow, def_dir, def_g = 26.0, 80.0, 12.0, "Concurrent", 0.38

elif preset == "Illustrative EVA-Related Oxygen Scenario (85% O2, 4.3 psi, 0g)":
    def_o2, def_p, def_flow, def_dir, def_g = 85.0, 29.6, 5.0, "Opposed", 0.0

else:
    def_o2, def_p, def_flow, def_dir, def_g = 21.6, 101.3, 20.0, "Concurrent", 0.0

st.sidebar.divider()
st.sidebar.header("🎛️ Fine-Tune Cabin Parameters")

o2 = st.sidebar.slider(
    "Oxygen Concentration (%)", 14.0, 85.0, float(def_o2), step=0.5
)

pressure = st.sidebar.slider(
    "Cabin Pressure (kPa) [101.3 = 14.7 psi]",
    29.6, 120.0, float(def_p), step=0.5
)

airflow = st.sidebar.slider(
    "Ventilation Airflow Speed (cm/s)",
    0.5, 40.0, float(def_flow), step=0.5
)

flow_dir = st.sidebar.radio(
    "Airflow Direction (SAFFIRE-I Factor)",
    ["Concurrent", "Opposed"],
    index=0 if def_dir == "Concurrent" else 1
)

gravity_label = st.sidebar.selectbox(
    "Gravity Regime",
    [
        "0g (Microgravity - ISS/Orion)",
        "0.165g (Moon Surface 1/6g)",
        "0.38g (Mars Surface 3/8g)",
        "1.0g (Earth Benchmark)"
    ],
    index=0 if def_g == 0.0 else (
        1 if def_g == 0.165 else (
            2 if def_g == 0.38 else 3
        )
    )
)

gravity_map = {
    "0g (Microgravity - ISS/Orion)": 0.0,
    "0.165g (Moon Surface 1/6g)": 0.165,
    "0.38g (Mars Surface 3/8g)": 0.38,
    "1.0g (Earth Benchmark)": 1.0
}

gravity_g = gravity_map[gravity_label]
is_concurrent = 1 if flow_dir == "Concurrent" else 0


# Compute ML Surrogate + Physics Engine Predictions
ml_input = pd.DataFrame(list(({
    "Oxygen_Percent": o2,
    "Pressure_kPa": pressure,
    "Airflow_cm_s": airflow,
    "Is_Concurrent": is_concurrent,
    "Gravity_g": gravity_g
},)))
ml_score = float(model.predict(ml_input).item())
physics_score = calculate_physics_hazard_score(o2, pressure, airflow, is_concurrent, gravity_g)
hybrid_score = 0.5 * ml_score + 0.5 * physics_score
conf_text, uncertainty_margin = get_partial_gravity_uncertainty(gravity_g)


# Main 5-Tab Dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(list((
    "🎯 Mission Fire-Risk Simulator",
    "📊 Rank & Compare NASA PSI Studies",
    "🧠 SI Research Summarizer & Q&A",
    "📚 NASA PSI Citations & Earth Impact",
    "🔥 PSI-26 BASS Experimental Data"
)))


# ---------------- TAB 1: SIMULATOR ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hybrid Fire Hazard Index", f"{hybrid_score:.1f} / 100", f"±{uncertainty_margin:.1f} uncertainty")
    c2.metric("ML Surrogate Prediction", f"{ml_score:.1f} / 100", "Random Forest (Synthetic-Target Model)")
    c3.metric("Physics Scaling Score", f"{physics_score:.1f} / 100", f"{flow_dir} Flow")
    c4.metric("Partial-O2 Pressure (ppO2)", f"{(o2/100.0)*pressure:.1f} kPa", f"{pressure*0.145038:.1f} psi total")

    if hybrid_score < 35:
        st.success(f"🟢 **LOW IGNITION RISK ({hybrid_score:.1f}/100):** Atmosphere within controllable suppression limits. Confidence: {conf_text}.")
    elif hybrid_score < 62:
        st.warning(f"🟡 **ELEVATED FLAMMABILITY RISK ({hybrid_score:.1f}/100):** Elevated oxygen and the selected {flow_dir.lower()} airflow configuration may influence flame behaviour. Confidence: {conf_text}.")
    else:
        st.error(
    f"🔴 **HIGH SCREENING BAND ({hybrid_score:.1f}/100):** "
    f"The scenario falls in the upper project-defined score band. "
    f"Flag for detailed engineering review and material-specific "
    f"fire-safety assessment. This is an illustrative screening output, "
    f"not a validated fire prediction or operational suppression instruction. "
    f"Confidence: {conf_text}."
)

    col_left, col_right = st.columns(list((1.3, 1)))

    with col_left:
        st.subheader("🌐 3D Synthetic Microgravity Fire-Hazard Visualization")

        fig_scatter = px.scatter_3d(
            grid_df,
            x="Oxygen_Percent",
            y="Airflow_cm_s",
            z="Fire_Hazard_Score",
            color="Investigation",
            size="Pressure_kPa",
            hover_data=list(("PSI_ID", "Pressure_kPa", "Gravity_g")),
            labels={
                "Oxygen_Percent": "O2 (%)",
                "Airflow_cm_s": "Airflow (cm/s)",
                "Fire_Hazard_Score": "Hazard (0-100)"
            },
            title="EigenMind Synthetic Fire-Hazard Scenario Envelope (Drag to Rotate • Scroll to Zoom)"
        )

        fig_scatter.add_trace(go.Scatter3d(
            x=list((o2,)),
            y=list((airflow,)),
            z=list((hybrid_score,)),
            mode="markers+text",
            marker=dict(
                size=10,
                color="#FC3D21",
                symbol="diamond",
                line=dict(color="white", width=2)
            ),
            text=list(("ACTIVE SIM",)),
            textposition="top center",
            name="Active Mission Sim"
        ))

        # Mission-control styling for the existing scientific 3D plot.
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Segoe UI, Arial, sans-serif",
                color="#C8D8E5",
                size=11
            ),
            title=dict(
                font=dict(size=15, color="#D8ECF8"),
                x=0.5,
                xanchor="center"
            ),
            margin=dict(l=0, r=0, b=0, t=48),
            height=570,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.08,
                xanchor="left",
                x=0,
                bgcolor="rgba(12,24,36,0.88)",
                bordercolor="rgba(105,153,184,0.45)",
                borderwidth=1,
                font=dict(color="#C8D8E5", size=10)
            ),
            hoverlabel=dict(
                bgcolor="#172838",
                bordercolor="#66BCE8",
                font=dict(color="#F3F8FC", family="Segoe UI, Arial", size=12)
            ),
            scene=dict(
                bgcolor="rgba(12,22,33,0.45)",
                aspectmode="cube",
                camera=dict(
                    eye=dict(x=1.65, y=1.65, z=1.25),
                    projection=dict(type="perspective")
                ),
                xaxis=dict(
                    title=dict(
                        text="OXYGEN CONCENTRATION (%)",
                        font=dict(color="#75C7ED", size=10)
                    ),
                    backgroundcolor="rgba(20,36,51,0.85)",
                    gridcolor="rgba(117,165,194,0.28)",
                    zerolinecolor="rgba(117,165,194,0.55)",
                    linecolor="#5D9ABF",
                    tickfont=dict(color="#B9CBD8", size=10),
                    showbackground=True,
                    showgrid=True,
                    showspikes=True,
                    spikecolor="#75C7ED",
                    spikethickness=2
                ),
                yaxis=dict(
                    title=dict(
                        text="AIRFLOW VELOCITY (cm/s)",
                        font=dict(color="#75C7ED", size=10)
                    ),
                    backgroundcolor="rgba(20,36,51,0.85)",
                    gridcolor="rgba(117,165,194,0.28)",
                    zerolinecolor="rgba(117,165,194,0.55)",
                    linecolor="#5D9ABF",
                    tickfont=dict(color="#B9CBD8", size=10),
                    showbackground=True,
                    showgrid=True,
                    showspikes=True,
                    spikecolor="#75C7ED",
                    spikethickness=2
                ),
                zaxis=dict(
                    title=dict(
                        text="FIRE HAZARD INDEX (0-100)",
                        font=dict(color="#F0A5A5", size=10)
                    ),
                    backgroundcolor="rgba(20,36,51,0.85)",
                    gridcolor="rgba(194,116,116,0.25)",
                    zerolinecolor="rgba(194,116,116,0.48)",
                    linecolor="#C87979",
                    tickfont=dict(color="#D8C2C2", size=10),
                    showbackground=True,
                    showgrid=True,
                    showspikes=True,
                    spikecolor="#EF8585",
                    spikethickness=2
                )
            )
        )

        # Keep the active simulation point visually distinct.
        fig_scatter.data[-1].update(
            marker=dict(
                size=11,
                color="#F06464",
                symbol="diamond",
                opacity=1,
                line=dict(color="#FFFFFF", width=2)
            ),
            textfont=dict(color="#FFFFFF", size=12, family="Arial")
        )

        st.plotly_chart(fig_scatter, width="stretch")

    with col_right:
        st.subheader("⚖️ Concurrent vs. Opposed Airflow (PSI-98 SAFFIRE-I)")

        saffire_compare = pd.DataFrame({
            "Test Run": list((
                "S1: Concurrent Flow (20 cm/s)",
                "S2: Opposed Flow (20 cm/s)"
            )),
            "Burn Duration (seconds)": list((420, 70)),
            "Material": list((
                "SIBAL Fabric (Cotton/Fiberglass)",
                "SIBAL Fabric (Cotton/Fiberglass)"
            ))
        })

        fig_bar = px.bar(
            saffire_compare,
            x="Test Run",
            y="Burn Duration (seconds)",
            color="Test Run",
            text="Burn Duration (seconds)",
            title="SAFFIRE-I Experimental Burn Time (DOI: 10.60555/0t15-1z43)",
            color_discrete_sequence=["#62BDEB", "#E87979"]
        )

        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,22,33,0.45)",
            font=dict(
                family="Segoe UI, Arial, sans-serif",
                color="#C8D8E5",
                size=11
            ),
            title=dict(
                font=dict(size=14, color="#D8ECF8"),
                x=0.5,
                xanchor="center"
            ),
            height=430,
            margin=dict(l=20, r=15, b=90, t=60),
            showlegend=False,
            hoverlabel=dict(
                bgcolor="#172838",
                bordercolor="#66BCE8",
                font=dict(color="#F3F8FC", size=12)
            ),
            xaxis=dict(
                title=None,
                tickfont=dict(color="#B9CBD8", size=10),
                gridcolor="rgba(117,165,194,0.10)",
                linecolor="#49677E"
            ),
            yaxis=dict(
                title=dict(
                    text="BURN DURATION (SECONDS)",
                    font=dict(color="#75C7ED", size=10)
                ),
                tickfont=dict(color="#B9CBD8", size=10),
                gridcolor="rgba(117,165,194,0.22)",
                zerolinecolor="rgba(117,165,194,0.40)",
                linecolor="#49677E"
            )
        )

        fig_bar.update_traces(
            marker=dict(
                line=dict(color="rgba(235,245,252,0.75)", width=1)
            ),
            textposition="outside",
            textfont=dict(color="#F0F5F9", size=13, family="Arial"),
            hovertemplate="<b>%{x}</b><br>Burn Duration: %{y} seconds<extra></extra>"
        )

        st.plotly_chart(fig_bar, width="stretch")

        st.info(
            "**Key Insight:** In NASA's SAFFIRE-I experiment aboard Cygnus (`PSI-98`), "
            "the reported concurrent-flow burning phase lasted 420 seconds, "
            "while the subsequent opposed-flow phase lasted 70 seconds. "
            "These were sequential phases of the experiment, not a controlled "
            "comparison of two independent fresh samples."
        )

    # ============================================================
    # PHASE 1: 3D PHYSICAL FLAME & AIRFLOW VISUALIZER
    # Illustrative parameter-driven visualization, not validated CFD
    # ============================================================

    st.markdown("---")
    st.subheader("🌐 3D Synthetic Fire-Hazard Scenario Space")

    st.caption(
        "ILLUSTRATIVE SIMULATION | Interactive flame morphology, "
        "airflow particles, soot distribution and time evolution"
    )

    flame_col, telemetry_col = st.columns([1.65, 1])

    with telemetry_col:
        st.markdown("#### MISSION SIMULATION CONTROLS")

        flame_time = st.slider(
            "Simulation Timeline (seconds)",
            min_value=0,
            max_value=100,
            value=35,
            step=1,
            help="Scrub through the illustrative ignition, growth and decay sequence."
        )

        st.markdown("#### LIVE ENVIRONMENT")

        st.metric("Oxygen Concentration", f"{o2:.1f}%")
        st.metric("Airflow Velocity", f"{airflow:.1f} cm/s")
        st.metric("Gravity Regime", f"{gravity_g:.3f} g")

        # Illustrative timeline envelope.
        if flame_time < 15:
            flame_phase = "IGNITION INITIATION"
            flame_intensity = flame_time / 15.0
        elif flame_time < 55:
            flame_phase = "FLAME DEVELOPMENT"
            flame_intensity = 1.0
        elif flame_time < 85:
            flame_phase = "FLAME DECAY"
            flame_intensity = (85 - flame_time) / 30.0
        else:
            flame_phase = "POST-EXTINCTION / RESIDUAL SHELL"
            flame_intensity = 0.0

        flame_intensity = float(np.clip(flame_intensity, 0.0, 1.0))

        st.markdown("#### COMBUSTION EVENT")

        if flame_intensity > 0.65:
            st.error(f"● {flame_phase}")
        elif flame_intensity > 0:
            st.warning(f"● {flame_phase}")
        else:
            st.success(f"● {flame_phase}")

        st.progress(flame_intensity)

        st.caption(
            "Timeline is a conceptual animation parameter. "
            "It does not represent experimentally measured burn duration."
        )

    with flame_col:

        # Spherical coordinates for an illustrative flame geometry.
        theta = np.linspace(0, 2 * np.pi, 72)
        phi = np.linspace(0, np.pi, 48)

        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Parameter-driven illustrative flame dimensions.
        flame_base_radius = 0.65 + 0.15 * (o2 / 85.0)
        flame_height = 1.15 + 0.30 * (airflow / 40.0)

        # Elongation varies with the selected gravity regime.
        gravity_shape = 1.0 + (0.35 * gravity_g)

        # Flame surface.
        radial_shape = (
            flame_base_radius
            * (1.0 + 0.20 * np.sin(phi_grid))
            * flame_intensity
        )

        X_flame = (
            radial_shape
            * np.sin(phi_grid)
            * np.cos(theta_grid)
        )

        Y_flame = (
            radial_shape
            * np.sin(phi_grid)
            * np.sin(theta_grid)
        )

        Z_flame = (
            radial_shape
            * np.cos(phi_grid)
            * flame_height
            * gravity_shape
        )

        # Cool-flame shell: conceptual outer boundary.
        shell_radius = radial_shape + 0.18

        X_shell = (
            shell_radius
            * np.sin(phi_grid)
            * np.cos(theta_grid)
        )

        Y_shell = (
            shell_radius
            * np.sin(phi_grid)
            * np.sin(theta_grid)
        )

        Z_shell = (
            shell_radius
            * np.cos(phi_grid)
            * flame_height
            * gravity_shape
        )

        fig_flame = go.Figure()

        # Translucent outer cool-flame visualization.
        fig_flame.add_trace(go.Surface(
            x=X_shell,
            y=Y_shell,
            z=Z_shell,
            surfacecolor=np.ones_like(Z_shell),
            colorscale=[
                [0, "#176B9B"],
                [1, "#66D9FF"]
            ],
            opacity=0.16,
            showscale=False,
            name="Illustrative Cool-Flame Shell",
            hoverinfo="skip"
        ))

        # Main flame surface.
        fig_flame.add_trace(go.Surface(
            x=X_flame,
            y=Y_flame,
            z=Z_flame,
            surfacecolor=Z_flame,
            colorscale=[
                [0.0, "#8B1E18"],
                [0.25, "#E34A24"],
                [0.55, "#FF8C32"],
                [0.8, "#FFD166"],
                [1.0, "#FFF0A6"]
            ],
            opacity=max(0.12, flame_intensity),
            showscale=False,
            name="Illustrative Flame Surface",
            hoverinfo="skip"
        ))

        # Deterministic particle distribution for repeatable visualization.
        particle_rng = np.random.default_rng(42)

        particle_count = 180

        particle_angle = particle_rng.uniform(
            0, 2 * np.pi, particle_count
        )

        particle_radius = particle_rng.uniform(
            0.75, 1.65, particle_count
        )

        particle_z = particle_rng.uniform(
            -1.2, 1.7, particle_count
        )

        # Illustrative airflow displacement.
        airflow_shift = (airflow / 40.0) * (flame_time / 100.0)

        particle_x = (
            particle_radius * np.cos(particle_angle)
            + airflow_shift * 0.4
        )

        particle_y = (
            particle_radius * np.sin(particle_angle)
        )

        particle_z = (
            particle_z + airflow_shift
        )

        # Soot / aerosol particle cloud.
        fig_flame.add_trace(go.Scatter3d(
            x=particle_x,
            y=particle_y,
            z=particle_z,
            mode="markers",
            marker=dict(
                size=2.5,
                color=particle_z,
                colorscale=[
                    [0, "#52677A"],
                    [0.5, "#9AAAB8"],
                    [1, "#D8E7F0"]
                ],
                opacity=0.55
            ),
            name="Illustrative Soot / Aerosol",
            hoverinfo="skip"
        ))

        # Airflow streamlines around the flame.
        stream_z = np.linspace(-1.5, 2.0, 55)

        for stream_index in range(5):
            stream_offset = -0.8 + stream_index * 0.4

            stream_x = (
                np.full_like(stream_z, stream_offset)
                + 0.12 * np.sin(stream_z * 2.0 + flame_time / 12.0)
            )

            stream_y = (
                np.full_like(stream_z, 1.35)
                + 0.08 * np.cos(stream_z * 2.0)
            )

            fig_flame.add_trace(go.Scatter3d(
                x=stream_x,
                y=stream_y,
                z=stream_z,
                mode="lines",
                line=dict(
                    color="#54C8F2",
                    width=3
                ),
                opacity=0.55,
                showlegend=(stream_index == 0),
                name="Airflow Streamlines",
                hoverinfo="skip"
            ))

        # Mission-control chart styling.
        fig_flame.update_layout(
            title=dict(
                text=(
                    "MICROGRAVITY FLAME MORPHOLOGY"
                    f"<br><sup>SIMULATION TIME: {flame_time}s | "
                    f"PHASE: {flame_phase}</sup>"
                ),
                x=0.5,
                xanchor="center",
                font=dict(
                    family="Arial",
                    size=15,
                    color="#D8ECF8"
                )
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Arial",
                color="#C8D8E5"
            ),
            height=600,
            margin=dict(
                l=0,
                r=0,
                t=75,
                b=0
            ),
            legend=dict(
                orientation="h",
                y=-0.08,
                x=0,
                bgcolor="rgba(10,22,35,0.7)",
                font=dict(size=9, color="#C8D8E5")
            ),
            scene=dict(
                bgcolor="rgba(8,17,29,0.3)",
                aspectmode="manual",
                aspectratio=dict(
                    x=1,
                    y=1,
                    z=1.3
                ),
                camera=dict(
                    eye=dict(
                        x=1.7,
                        y=1.7,
                        z=1.35
                    )
                ),
                xaxis=dict(
                    title="X POSITION",
                    backgroundcolor="rgba(10,25,40,0.6)",
                    gridcolor="rgba(75,185,225,0.2)",
                    zerolinecolor="#397D9C",
                    color="#AFC6D6"
                ),
                yaxis=dict(
                    title="Y POSITION",
                    backgroundcolor="rgba(10,25,40,0.6)",
                    gridcolor="rgba(75,185,225,0.2)",
                    zerolinecolor="#397D9C",
                    color="#AFC6D6"
                ),
                zaxis=dict(
                    title="HEIGHT",
                    backgroundcolor="rgba(10,25,40,0.6)",
                    gridcolor="rgba(75,185,225,0.2)",
                    zerolinecolor="#397D9C",
                    color="#AFC6D6"
                )
            )
        )

        st.plotly_chart(
            fig_flame,
            width="stretch",
            key="eigenmind_physical_flame"
        )

    st.caption(
        "VISUALIZATION NOTICE: Flame geometry, particle trajectories, "
        "cool-flame shell and airflow paths are illustrative. "
        "They are not validated CFD outputs or direct NASA measurements."
    )

    # ============================================================
    # PHASE 2: 3D ISS-STYLE COMBUSTION CHAMBER DIGITAL TWIN
    # Conceptual habitat geometry and illustrative sensor telemetry
    # ============================================================

    st.markdown("---")
    st.subheader("🛰️ Conceptual Spacecraft Combustion Chamber | Habitat Digital Twin")

    st.caption(
        "CONCEPTUAL SPACECRAFT LABORATORY MODEL | "
        "Interactive chamber geometry, sensor locations and aerosol zones"
    )

    habitat_left, habitat_right = st.columns([1.65, 1])

    with habitat_right:
        st.markdown("#### HABITAT TELEMETRY CONTROLS")

        habitat_show_sensors = st.checkbox(
            "Show environmental sensors",
            value=True,
            key="habitat_show_sensors"
        )

        habitat_show_smoke = st.checkbox(
            "Show smoke / aerosol particles",
            value=True,
            key="habitat_show_smoke"
        )

        habitat_show_zones = st.checkbox(
            "Show conceptual detection zones",
            value=True,
            key="habitat_show_zones"
        )

        habitat_smoke_level = st.slider(
            "Illustrative aerosol concentration",
            min_value=0,
            max_value=100,
            value=45,
            key="habitat_smoke_level"
        )

        habitat_camera = st.selectbox(
            "Chamber viewing angle",
            ["Isometric", "Front", "Side", "Top"],
            key="habitat_camera"
        )

        st.markdown("#### ENVIRONMENTAL STATUS")

        st.metric(
            "Cabin Oxygen",
            f"{o2:.1f}%"
        )

        st.metric(
            "Cabin Pressure",
            f"{pressure:.1f} kPa"
        )

        st.metric(
            "Gravity",
            f"{gravity_g:.3f} g"
        )

        st.metric(
            "Airflow",
            f"{airflow:.1f} cm/s"
        )

        st.caption(
            "Sensor readings and smoke distributions below are "
            "illustrative, not actual ISS telemetry."
        )

    # Build the conceptual combustion chamber.
    with habitat_left:

        habitat_fig = go.Figure()

        # Chamber dimensions: normalized visualization units.
        habitat_x = [-2.5, 2.5]
        habitat_y = [-1.5, 1.5]
        habitat_z = [-1.5, 1.5]

        # Eight vertices of the chamber.
        habitat_vertices = [
            [-2.5, -1.5, -1.5],
            [ 2.5, -1.5, -1.5],
            [ 2.5,  1.5, -1.5],
            [-2.5,  1.5, -1.5],
            [-2.5, -1.5,  1.5],
            [ 2.5, -1.5,  1.5],
            [ 2.5,  1.5,  1.5],
            [-2.5,  1.5,  1.5]
        ]

        habitat_vx = [v[0] for v in habitat_vertices]
        habitat_vy = [v[1] for v in habitat_vertices]
        habitat_vz = [v[2] for v in habitat_vertices]

        # Transparent chamber enclosure.
        habitat_fig.add_trace(go.Mesh3d(
            x=habitat_vx,
            y=habitat_vy,
            z=habitat_vz,
            i=[0, 0, 4, 4, 0, 1, 2, 3, 0, 4, 5, 6],
            j=[1, 2, 5, 6, 1, 5, 6, 7, 4, 7, 6, 7],
            k=[2, 3, 6, 7, 5, 6, 7, 4, 3, 6, 2, 3],
            color="#2D7194",
            opacity=0.09,
            flatshading=True,
            name="Habitat Enclosure",
            hoverinfo="skip",
            showscale=False
        ))

        # Chamber structural edges.
        habitat_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for habitat_edge in habitat_edges:
            habitat_fig.add_trace(go.Scatter3d(
                x=[
                    habitat_vertices[habitat_edge[0]][0],
                    habitat_vertices[habitat_edge[1]][0]
                ],
                y=[
                    habitat_vertices[habitat_edge[0]][1],
                    habitat_vertices[habitat_edge[1]][1]
                ],
                z=[
                    habitat_vertices[habitat_edge[0]][2],
                    habitat_vertices[habitat_edge[1]][2]
                ],
                mode="lines",
                line=dict(
                    color="#65BCE8",
                    width=4
                ),
                showlegend=False,
                hoverinfo="skip"
            ))

        # Equipment rack geometry.
        habitat_racks = [
            (-1.85, -1.05, -0.85, 0.65, 0.9, 1.1, "CIR RACK"),
            (0.95, -1.05, -0.85, 0.65, 0.9, 1.1, "CONTROL RACK"),
            (-0.45, 0.75, -0.95, 0.9, 0.45, 0.9, "EXPERIMENT BAY")
        ]

        for habitat_rack in habitat_racks:

            rack_x, rack_y, rack_z, rack_w, rack_d, rack_h, rack_name = habitat_rack

            habitat_fig.add_trace(go.Mesh3d(
                x=[
                    rack_x, rack_x + rack_w,
                    rack_x + rack_w, rack_x,
                    rack_x, rack_x + rack_w,
                    rack_x + rack_w, rack_x
                ],
                y=[
                    rack_y, rack_y,
                    rack_y + rack_d, rack_y + rack_d,
                    rack_y, rack_y,
                    rack_y + rack_d, rack_y + rack_d
                ],
                z=[
                    rack_z, rack_z,
                    rack_z, rack_z,
                    rack_z + rack_h, rack_z + rack_h,
                    rack_z + rack_h, rack_z + rack_h
                ],
                i=[0, 0, 4, 4, 0, 1, 2, 3, 0, 4, 5, 6],
                j=[1, 2, 5, 6, 1, 5, 6, 7, 4, 7, 6, 7],
                k=[2, 3, 6, 7, 5, 6, 7, 4, 3, 6, 2, 3],
                color="#34566F",
                opacity=0.88,
                flatshading=True,
                name=rack_name,
                showlegend=False,
                hovertemplate=rack_name + "<extra></extra>"
            ))

        # Sensor positions inside the habitat.
        habitat_sensor_positions = {
            "Smoke Detector A": [-1.7, 0.0, 1.15],
            "Smoke Detector B": [1.7, 0.0, 1.15],
            "Temperature Sensor": [0.0, -0.7, 0.7],
            "Airflow Sensor": [0.0, 1.0, 0.2],
            "Oxygen Sensor": [-1.8, -0.8, -0.5],
            "Pressure Sensor": [1.8, -0.8, -0.5]
        }

        habitat_sensor_colors = [
            "#F05D5E",
            "#F05D5E",
            "#FFD166",
            "#56CCF2",
            "#63D39B",
            "#B69CFF"
        ]

        if habitat_show_sensors:

            for habitat_index, (habitat_sensor_name, habitat_position) in enumerate(
                habitat_sensor_positions.items()
            ):

                habitat_fig.add_trace(go.Scatter3d(
                    x=[habitat_position[0]],
                    y=[habitat_position[1]],
                    z=[habitat_position[2]],
                    mode="markers+text",
                    marker=dict(
                        size=8,
                        color=habitat_sensor_colors[habitat_index],
                        line=dict(
                            color="#FFFFFF",
                            width=1.5
                        )
                    ),
                    text=[habitat_sensor_name],
                    textposition="top center",
                    textfont=dict(
                        size=9,
                        color="#E5F3FF"
                    ),
                    name=habitat_sensor_name,
                    hovertemplate=(
                        "<b>" + habitat_sensor_name +
                        "</b><br>Conceptual sensor location<extra></extra>"
                    )
                ))

        # Illustrative aerosol / smoke cloud.
        habitat_rng = np.random.default_rng(2026)

        habitat_particle_count = int(
            20 + habitat_smoke_level * 3
        )

        habitat_smoke_x = habitat_rng.normal(
            0.0,
            0.45 + habitat_smoke_level / 160.0,
            habitat_particle_count
        )

        habitat_smoke_y = habitat_rng.normal(
            0.0,
            0.35 + habitat_smoke_level / 200.0,
            habitat_particle_count
        )

        habitat_smoke_z = habitat_rng.uniform(
            -0.5,
            1.2,
            habitat_particle_count
        )

        if habitat_show_smoke:

            habitat_fig.add_trace(go.Scatter3d(
                x=habitat_smoke_x,
                y=habitat_smoke_y,
                z=habitat_smoke_z,
                mode="markers",
                marker=dict(
                    size=3 + habitat_smoke_level / 35.0,
                    color=habitat_smoke_z,
                    colorscale=[
                        [0, "#718096"],
                        [0.5, "#B0BEC5"],
                        [1, "#F0F4F8"]
                    ],
                    opacity=0.45
                ),
                name="Illustrative Aerosol Cloud",
                hoverinfo="skip"
            ))

        # Conceptual detector coverage zones.
        if habitat_show_zones:

            for habitat_zone_index, habitat_zone_center in enumerate([
                [-1.7, 0.0, 1.15],
                [1.7, 0.0, 1.15]
            ]):

                habitat_zone_u = np.linspace(
                    0, 2 * np.pi, 24
                )

                habitat_zone_v = np.linspace(
                    0, np.pi, 16
                )

                habitat_uu, habitat_vv = np.meshgrid(
                    habitat_zone_u,
                    habitat_zone_v
                )

                habitat_zone_radius = 0.65

                habitat_fig.add_trace(go.Surface(
                    x=(
                        habitat_zone_center[0]
                        + habitat_zone_radius
                        * np.cos(habitat_uu)
                        * np.sin(habitat_vv)
                    ),
                    y=(
                        habitat_zone_center[1]
                        + habitat_zone_radius
                        * np.sin(habitat_uu)
                        * np.sin(habitat_vv)
                    ),
                    z=(
                        habitat_zone_center[2]
                        + habitat_zone_radius
                        * np.cos(habitat_vv)
                    ),
                    colorscale=[
                        [0, "#E35D6A"],
                        [1, "#F59E9E"]
                    ],
                    opacity=0.10,
                    showscale=False,
                    showlegend=(habitat_zone_index == 0),
                    name="Conceptual Detector Coverage",
                    hoverinfo="skip"
                ))

        # Camera selection.
        habitat_camera_positions = {
            "Isometric": dict(x=1.7, y=1.7, z=1.4),
            "Front": dict(x=0.0, y=3.5, z=0.2),
            "Side": dict(x=3.5, y=0.0, z=0.2),
            "Top": dict(x=0.0, y=0.1, z=4.0)
        }

        habitat_fig.update_layout(
            title=dict(
                text="ISS-STYLE COMBUSTION LABORATORY DIGITAL TWIN",
                x=0.5,
                xanchor="center",
                font=dict(
                    size=14,
                    color="#D8ECF8"
                )
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Arial",
                color="#C8D8E5"
            ),
            height=610,
            margin=dict(
                l=0,
                r=0,
                t=55,
                b=0
            ),
            legend=dict(
                orientation="h",
                y=-0.08,
                x=0,
                bgcolor="rgba(10,22,35,0.8)",
                font=dict(
                    size=9,
                    color="#D8ECF8"
                )
            ),
            scene=dict(
                bgcolor="rgba(8,17,29,0.35)",
                aspectmode="manual",
                aspectratio=dict(
                    x=1.7,
                    y=1,
                    z=1
                ),
                camera=dict(
                    eye=habitat_camera_positions[habitat_camera]
                ),
                xaxis=dict(
                    title="CHAMBER X",
                    backgroundcolor="rgba(10,25,40,0.5)",
                    gridcolor="rgba(75,185,225,0.18)",
                    color="#AFC6D6",
                    range=[-3, 3]
                ),
                yaxis=dict(
                    title="CHAMBER Y",
                    backgroundcolor="rgba(10,25,40,0.5)",
                    gridcolor="rgba(75,185,225,0.18)",
                    color="#AFC6D6",
                    range=[-2, 2]
                ),
                zaxis=dict(
                    title="CHAMBER Z",
                    backgroundcolor="rgba(10,25,40,0.5)",
                    gridcolor="rgba(75,185,225,0.18)",
                    color="#AFC6D6",
                    range=[-2, 2]
                )
            )
        )

        st.plotly_chart(
            habitat_fig,
            width="stretch",
            key="eigenmind_habitat_digital_twin"
        )

    st.caption(
        "MODEL LIMITATION: Chamber dimensions, equipment racks, sensor "
        "positions, detector coverage and aerosol cloud are conceptual. "
        "They are not exact ISS engineering drawings, calibrated smoke "
        "measurements or verified detector performance."
    )


    # ============================================================
    # PHASE 3: NASA PSI CLICK-TO-INSPECT INTELLIGENCE
    # Interactive investigation selection + research intelligence
    # ============================================================

    st.markdown("---")
    st.subheader("🧠 NASA PSI Investigation Intelligence")
    st.caption(
        "INTERACTIVE RESEARCH EXPLORER | Select a point to inspect "
        "NASA investigation parameters, hardware, materials and findings"
    )

    # Required dataset fields are already used by your Tab 2 comparator.
    phase3_required_columns = [
        "PSI_ID",
        "Investigation",
        "Hardware",
        "Material",
        "Fuel_State",
        "Oxygen_Percent",
        "Pressure_kPa",
        "Airflow_cm_s",
        "Flow_Direction",
        "Gravity_g",
        "Burn_Time_s",
        "DOI",
        "Key_Finding"
    ]

    phase3_missing_columns = [
        col for col in phase3_required_columns
        if col not in base_df.columns
    ]

    if phase3_missing_columns:
        st.error(
            "Phase 3 cannot start because these dataset columns are missing: "
            + ", ".join(phase3_missing_columns)
        )

    elif base_df.empty:
        st.warning("No NASA PSI investigation records are available.")

    else:
        # Work on a clean copy; do not modify the original dataset.
        phase3_df = base_df.copy().reset_index(drop=True)

        # Convert numerical plotting fields safely.
        phase3_numeric_columns = [
            "Oxygen_Percent",
            "Airflow_cm_s",
            "Gravity_g",
            "Pressure_kPa",
            "Burn_Time_s"
        ]

        for phase3_col in phase3_numeric_columns:
            phase3_df[phase3_col] = pd.to_numeric(
                phase3_df[phase3_col],
                errors="coerce"
            )

        # Keep only records with valid coordinates for the 3D plot.
        phase3_plot_df = phase3_df.dropna(
            subset=[
                "Oxygen_Percent",
                "Airflow_cm_s",
                "Gravity_g"
            ]
        ).reset_index(drop=True)

        if phase3_plot_df.empty:
            st.warning(
                "No records have valid oxygen, airflow and gravity "
                "values for the interactive 3D plot."
            )

        else:
            st.markdown("#### 🔭 INVESTIGATION PHASE-SPACE")

            phase3_left, phase3_right = st.columns([1.7, 1])

            with phase3_right:
                st.markdown("##### PLOT CONFIGURATION")

                phase3_color = st.selectbox(
                    "Color points by",
                    [
                        "Investigation",
                        "Flow_Direction",
                        "Material",
                        "Fuel_State"
                    ],
                    key="phase3_color_by"
                )

                phase3_size = st.checkbox(
                    "Scale marker size by burn duration",
                    value=False,
                    key="phase3_size_by_burn"
                )

                st.info(
                    "Select a point in the 3D plot to load its "
                    "investigation intelligence below."
                )

                st.metric(
                    "Available investigations",
                    len(phase3_plot_df)
                )

                st.metric(
                    "Unique PSI identifiers",
                    phase3_plot_df["PSI_ID"].nunique()
                )

            with phase3_left:
                # Build an interactive 3D investigation map.
                phase3_fig = px.scatter_3d(
                    phase3_plot_df,
                    x="Oxygen_Percent",
                    y="Airflow_cm_s",
                    z="Gravity_g",
                    color=phase3_color,
                    size=(
                        "Burn_Time_s"
                        if phase3_size
                        and phase3_plot_df["Burn_Time_s"].notna().any()
                        else None
                    ),
                    hover_name="Investigation",
                    hover_data={
                        "PSI_ID": True,
                        "Hardware": True,
                        "Material": True,
                        "Pressure_kPa": True,
                        "Flow_Direction": True,
                        "Burn_Time_s": True
                    },
                    custom_data=[
                        "PSI_ID",
                        "Investigation"
                    ],
                    title="NASA PSI | 3D Investigation Phase-Space"
                )

                phase3_fig.update_traces(
                    marker=dict(
                        size=7,
                        opacity=0.88,
                        line=dict(
                            width=0.7,
                            color="#D7F2FF"
                        )
                    )
                )

                phase3_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        family="Arial",
                        color="#D8ECF8"
                    ),
                    height=590,
                    margin=dict(
                        l=0,
                        r=0,
                        t=65,
                        b=0
                    ),
                    legend=dict(
                        bgcolor="rgba(10,22,35,0.75)",
                        font=dict(
                            color="#D8ECF8",
                            size=9
                        )
                    ),
                    scene=dict(
                        bgcolor="rgba(8,17,29,0.35)",
                        xaxis=dict(
                            title="OXYGEN (%)",
                            gridcolor="rgba(75,185,225,0.2)",
                            color="#AFC6D6"
                        ),
                        yaxis=dict(
                            title="AIRFLOW (cm/s)",
                            gridcolor="rgba(75,185,225,0.2)",
                            color="#AFC6D6"
                        ),
                        zaxis=dict(
                            title="GRAVITY (g)",
                            gridcolor="rgba(75,185,225,0.2)",
                            color="#AFC6D6"
                        ),
                        camera=dict(
                            eye=dict(x=1.6, y=1.6, z=1.25)
                        )
                    )
                )

                # Streamlit selection event.
                phase3_event = st.plotly_chart(
                    phase3_fig,
                    width="stretch",
                    key="eigenmind_psi_click_inspect",
                    on_select="rerun",
                    selection_mode="points"
                )

            # ----------------------------------------------------
            # SELECTED INVESTIGATION
            # ----------------------------------------------------

            phase3_selected_points = (
                phase3_event.selection.points
                if phase3_event
                and hasattr(phase3_event, "selection")
                else []
            )

            if phase3_selected_points:
                phase3_point = phase3_selected_points[0]

                phase3_point_index = phase3_point.get(
                    "point_index",
                    None
                )

                if phase3_point_index is not None:
                    phase3_selected = phase3_plot_df.iloc[
                        int(phase3_point_index)
                    ]

                else:
                    phase3_selected = None

            else:
                phase3_selected = None

            # Allow selection through a fallback dropdown.
            # This is useful when a device cannot select Plotly points.
            if phase3_selected is None:
                st.markdown("---")
                st.markdown("#### 🔎 SELECT AN INVESTIGATION")

                phase3_options = phase3_plot_df[
                    ["PSI_ID", "Investigation"]
                ].copy()

                phase3_options["Display"] = (
                    phase3_options["PSI_ID"].astype(str)
                    + " | "
                    + phase3_options["Investigation"].astype(str)
                )

                phase3_choice = st.selectbox(
                    "Choose a NASA PSI investigation",
                    phase3_options["Display"].tolist(),
                    key="phase3_fallback_selector"
                )

                phase3_selected = phase3_plot_df.iloc[
                    phase3_options.index[
                        phase3_options["Display"] == phase3_choice
                    ][0]
                ]

            # ----------------------------------------------------
            # INTELLIGENCE DISPLAY
            # ----------------------------------------------------

            st.markdown("---")
            st.markdown("### 🛰️ SELECTED INVESTIGATION INTELLIGENCE")

            phase3_id = str(phase3_selected["PSI_ID"])
            phase3_name = str(phase3_selected["Investigation"])

            st.markdown(
                f"## {phase3_name}"
            )

            st.markdown(
                f"**NASA PSI Identifier:** `{phase3_id}`"
            )

            phase3_kpi1, phase3_kpi2, phase3_kpi3, phase3_kpi4 = (
                st.columns(4)
            )

            phase3_kpi1.metric(
                "Oxygen",
                f"{phase3_selected['Oxygen_Percent']:.2f}%"
                if pd.notna(phase3_selected["Oxygen_Percent"])
                else "N/A"
            )

            phase3_kpi2.metric(
                "Pressure",
                f"{phase3_selected['Pressure_kPa']:.2f} kPa"
                if pd.notna(phase3_selected["Pressure_kPa"])
                else "N/A"
            )

            phase3_kpi3.metric(
                "Airflow",
                f"{phase3_selected['Airflow_cm_s']:.2f} cm/s"
                if pd.notna(phase3_selected["Airflow_cm_s"])
                else "N/A"
            )

            phase3_kpi4.metric(
                "Gravity",
                f"{phase3_selected['Gravity_g']:.3f} g"
                if pd.notna(phase3_selected["Gravity_g"])
                else "N/A"
            )

            phase3_detail_left, phase3_detail_right = st.columns(2)

            with phase3_detail_left:
                st.markdown("#### ⚙️ EXPERIMENTAL CONFIGURATION")

                st.markdown(
                    f"**Hardware:** {phase3_selected['Hardware']}"
                )

                st.markdown(
                    f"**Material / Fuel:** {phase3_selected['Material']}"
                )

                st.markdown(
                    f"**Fuel State:** {phase3_selected['Fuel_State']}"
                )

                st.markdown(
                    f"**Flow Direction:** {phase3_selected['Flow_Direction']}"
                )

                st.markdown(
                    f"**Burn Duration:** "
                    f"{phase3_selected['Burn_Time_s']} seconds"
                )

            with phase3_detail_right:
                st.markdown("#### 🧪 RESEARCH FINDINGS")

                phase3_finding = phase3_selected["Key_Finding"]

                if pd.notna(phase3_finding) and str(phase3_finding).strip():
                    st.success(str(phase3_finding))
                else:
                    st.warning(
                        "No scientific finding is recorded for this row."
                    )

                st.markdown("#### 📄 SOURCE METADATA")

                phase3_doi = phase3_selected["DOI"]

                if pd.notna(phase3_doi) and str(phase3_doi).strip():
                    phase3_doi_text = str(phase3_doi).strip()

                    if phase3_doi_text.startswith("http"):
                        st.markdown(
                            f"[Open DOI source]({phase3_doi_text})"
                        )
                    else:
                        st.markdown(
                            f"[Open DOI source](https://doi.org/{phase3_doi_text})"
                        )
                else:
                    st.caption("DOI not available in the current dataset.")

            # ----------------------------------------------------
            # EIGENMIND RESEARCH SUMMARY
            # ----------------------------------------------------

            st.markdown("---")
            st.markdown("### 🧠 EIGENMIND RESEARCH INTERPRETATION")

            phase3_summary = (
                f"The selected investigation, {phase3_name}, "
                f"is associated with the hardware "
                f"{phase3_selected['Hardware']} and material "
                f"{phase3_selected['Material']}. "
                f"The recorded experimental environment includes "
                f"{phase3_selected['Oxygen_Percent']}% oxygen, "
                f"{phase3_selected['Pressure_kPa']} kPa pressure, "
                f"{phase3_selected['Airflow_cm_s']} cm/s airflow, "
                f"and {phase3_selected['Gravity_g']} g gravity. "
                f"The recorded flow direction is "
                f"{phase3_selected['Flow_Direction']}. "
            )

            st.info(phase3_summary)

            st.caption(
                "INTERPRETATION NOTICE: This summary is generated from "
                "the selected dataset row. It is not an independent "
                "NASA-authored conclusion or a new experimental result."
            )

            # Download selected investigation record.
            phase3_export = phase3_selected.to_frame().T

            st.download_button(
                label="⬇️ Export Selected Investigation (CSV)",
                data=phase3_export.to_csv(index=False),
                file_name=f"EigenMind_{phase3_id}_investigation.csv",
                mime="text/csv",
                key="phase3_export_selected"
            )


    # ============================================================
    # PHASE 3B: SOURCE-LINKED NASA PSI INVESTIGATION CATALOGUE
    # Metadata catalogue is separate from numerical experiment data.
    # ============================================================

    st.markdown("---")
    st.subheader("?? NASA PSI Investigation Catalogue")
    st.caption(
        "SOURCE-LINKED RESEARCH CATALOGUE | Includes investigations "
        "that do not yet have numerical parameters in the master dataset."
    )

    phase3b_catalogue_path = "data/nasa_investigation_catalogue.csv"

    if os.path.exists(phase3b_catalogue_path):
        try:
            phase3b_catalogue = pd.read_csv(
                phase3b_catalogue_path,
                dtype=str,
                keep_default_na=False
            )

            phase3b_required = [
                "PSI_ID",
                "Investigation",
                "Research_Area",
                "Platform",
                "Data_Type",
                "DOI",
                "NASA_Source",
                "Verification_Status"
            ]

            phase3b_missing = [
                c for c in phase3b_required
                if c not in phase3b_catalogue.columns
            ]

            if phase3b_missing:
                st.error(
                    "Catalogue is missing required columns: "
                    + ", ".join(phase3b_missing)
                )
            elif phase3b_catalogue.empty:
                st.warning("The investigation catalogue is empty.")
            else:
                phase3b_verified_mask = (
                    phase3b_catalogue["Verification_Status"]
                    .str.contains("Verified NASA PSI record", case=False)
                )

                phase3b_verified_count = int(
                    phase3b_verified_mask.sum()
                )
                phase3b_pending_count = (
                    len(phase3b_catalogue) - phase3b_verified_count
                )

                phase3b_k1, phase3b_k2, phase3b_k3 = st.columns(3)

                phase3b_k1.metric(
                    "Catalogue investigations",
                    len(phase3b_catalogue)
                )
                phase3b_k2.metric(
                    "Verified PSI records",
                    phase3b_verified_count
                )
                phase3b_k3.metric(
                    "Identifiers pending verification",
                    phase3b_pending_count
                )

                st.dataframe(
                    phase3b_catalogue[phase3b_required],
                    width="stretch",
                    hide_index=True
                )

                phase3b_options = (
                    phase3b_catalogue["PSI_ID"].astype(str)
                    + " | "
                    + phase3b_catalogue["Investigation"].astype(str)
                )

                phase3b_choice = st.selectbox(
                    "Inspect a catalogue investigation",
                    phase3b_options.tolist(),
                    key="phase3b_catalogue_selector"
                )

                phase3b_selected_index = phase3b_options[
                    phase3b_options == phase3b_choice
                ].index[0]

                phase3b_selected = phase3b_catalogue.iloc[
                    phase3b_selected_index
                ]

                st.markdown("#### Selected investigation")

                st.markdown(
                    "**Investigation:** "
                    + str(phase3b_selected["Investigation"])
                )
                st.markdown(
                    "**PSI identifier:** "
                    + str(phase3b_selected["PSI_ID"])
                )
                st.markdown(
                    "**Research area:** "
                    + str(phase3b_selected["Research_Area"])
                )
                st.markdown(
                    "**Platform:** "
                    + str(phase3b_selected["Platform"])
                )
                st.markdown(
                    "**Available data classification:** "
                    + str(phase3b_selected["Data_Type"])
                )
                st.caption(
                    "Verification: "
                    + str(phase3b_selected["Verification_Status"])
                )

                phase3b_doi = str(
                    phase3b_selected["DOI"]
                ).strip()

                phase3b_source = str(
                    phase3b_selected["NASA_Source"]
                ).strip()

                phase3b_link_left, phase3b_link_right = st.columns(2)

                with phase3b_link_left:
                    if (
                        phase3b_doi
                        and phase3b_doi.lower() not in ["nan", "none"]
                    ):
                        if phase3b_doi.startswith("http"):
                            phase3b_doi_url = phase3b_doi
                        else:
                            phase3b_doi_url = (
                                "https://doi.org/" + phase3b_doi
                            )

                        st.markdown(
                            f"[Open DOI source]({phase3b_doi_url})"
                        )
                    else:
                        st.caption("No verified DOI supplied.")

                with phase3b_link_right:
                    if phase3b_source.startswith("https://"):
                        st.markdown(
                            f"[Open NASA source record]({phase3b_source})"
                        )
                    else:
                        st.caption("NASA source link unavailable.")

                st.info(
                    "Catalogue metadata is not equivalent to numerical "
                    "experimental measurements. Records with unverified "
                    "identifiers must not be treated as verified PSI datasets."
                )

        except Exception as phase3b_error:
            st.error(
                "Could not load the NASA PSI investigation catalogue: "
                + str(phase3b_error)
            )
    else:
        st.warning(
            "Catalogue file not found: "
            + phase3b_catalogue_path
        )

    # PSI-26 dataset connection: test-index data, not sensor telemetry.
    phase3b_bass_path = (
        "data/psi_26_bass_experimental_table.csv"
    )

    if os.path.exists(phase3b_bass_path):
        try:
            phase3b_bass = pd.read_csv(phase3b_bass_path)

            st.markdown("#### PSI-26 BASS dataset connection")

            phase3b_bass_a, phase3b_bass_b = st.columns(2)

            phase3b_bass_a.metric(
                "PSI-26 test-index records",
                len(phase3b_bass)
            )

            if "Fuel Sample Material" in phase3b_bass.columns:
                phase3b_material_count = (
                    phase3b_bass["Fuel Sample Material"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .nunique()
                )
            else:
                phase3b_material_count = 0

            phase3b_bass_b.metric(
                "Recorded material labels",
                phase3b_material_count
            )

            st.caption(
                "This file contains test-index metadata. It is not "
                "a table of measured flame-spread rates, oxygen, pressure "
                "or suppression outcomes."
            )

            with st.expander("Preview PSI-26 test-index records"):
                st.dataframe(
                    phase3b_bass.head(20),
                    width="stretch",
                    hide_index=True
                )

        except Exception as phase3b_bass_error:
            st.warning(
                "PSI-26 catalogue connection could not be loaded: "
                + str(phase3b_bass_error)
            )

    # ============================================================
    # PHASE 4: 4D TIMELINE PLAYBACK ENGINE
    # 3D NASA PSI phase-space + 4th dimension (burn duration)
    # ============================================================

    import plotly.graph_objects as go

    st.markdown("---")
    st.subheader("⏱️ 4D NASA PSI Timeline Playback")
    st.caption(
        "3D PHASE SPACE + TIME DIMENSION | Explore how NASA PSI "
        "investigations are distributed across recorded burn durations"
    )

    phase4_required = [
        "PSI_ID",
        "Investigation",
        "Oxygen_Percent",
        "Airflow_cm_s",
        "Gravity_g",
        "Burn_Time_s"
    ]

    phase4_missing = [
        col for col in phase4_required
        if col not in base_df.columns
    ]

    if phase4_missing:
        st.error(
            "Phase 4 cannot run. Missing dataset columns: "
            + ", ".join(phase4_missing)
        )

    else:
        phase4_df = base_df.copy()

        # Convert numerical fields safely
        for phase4_col in [
            "Oxygen_Percent",
            "Airflow_cm_s",
            "Gravity_g",
            "Burn_Time_s"
        ]:
            phase4_df[phase4_col] = pd.to_numeric(
                phase4_df[phase4_col],
                errors="coerce"
            )

        # Keep records with valid 3D coordinates and burn duration
        phase4_plot_df = phase4_df.dropna(
            subset=[
                "Oxygen_Percent",
                "Airflow_cm_s",
                "Gravity_g",
                "Burn_Time_s"
            ]
        ).copy()

        phase4_plot_df = phase4_plot_df[
            phase4_plot_df["Burn_Time_s"] >= 0
        ].reset_index(drop=True)

        if phase4_plot_df.empty:
            st.warning(
                "No valid NASA PSI records are available "
                "for the timeline visualization."
            )

        else:
            # ----------------------------------------------------
            # CONTROLS
            # ----------------------------------------------------

            phase4_col1, phase4_col2 = st.columns(2)

            with phase4_col1:
                phase4_investigations = [
                    "All Investigations"
                ] + sorted(
                    phase4_plot_df["Investigation"]
                    .fillna("Unknown")
                    .astype(str)
                    .unique()
                    .tolist()
                )

                phase4_selected = st.selectbox(
                    "Select Investigation",
                    phase4_investigations,
                    key="phase4_investigation_filter"
                )

            with phase4_col2:
                phase4_show_labels = st.checkbox(
                    "Show PSI identifiers",
                    value=True,
                    key="phase4_show_labels"
                )

            # Apply investigation filter
            if phase4_selected != "All Investigations":
                phase4_plot_df = phase4_plot_df[
                    phase4_plot_df["Investigation"]
                    .fillna("Unknown")
                    .astype(str) == phase4_selected
                ].reset_index(drop=True)

            # ----------------------------------------------------
            # TIMELINE STATISTICS
            # ----------------------------------------------------

            phase4_min_time = float(
                phase4_plot_df["Burn_Time_s"].min()
            )

            phase4_max_time = float(
                phase4_plot_df["Burn_Time_s"].max()
            )

            phase4_total = len(phase4_plot_df)

            phase4_k1, phase4_k2, phase4_k3 = st.columns(3)

            phase4_k1.metric(
                "NASA PSI Records",
                phase4_total
            )

            phase4_k2.metric(
                "Minimum Burn Duration",
                f"{phase4_min_time:.2f} s"
            )

            phase4_k3.metric(
                "Maximum Burn Duration",
                f"{phase4_max_time:.2f} s"
            )

            st.markdown("### 4D Phase-Space Playback")

            st.info(
                "Use Play to progress through recorded burn durations. "
                "The highlighted points represent investigations whose "
                "recorded Burn_Time_s is at or below the current timeline "
                "position. This is a duration-based data visualization, "
                "not a reconstruction of continuous flame telemetry."
            )

            # ----------------------------------------------------
            # PREPARE TIMELINE FRAMES
            # ----------------------------------------------------

            if phase4_max_time > phase4_min_time:
                phase4_thresholds = np.linspace(
                    phase4_min_time,
                    phase4_max_time,
                    25
                )
            else:
                phase4_thresholds = np.array(
                    [phase4_min_time]
                )

            phase4_ids = (
                phase4_plot_df["PSI_ID"]
                .fillna("Unknown")
                .astype(str)
                .to_numpy()
            )

            phase4_names = (
                phase4_plot_df["Investigation"]
                .fillna("Unknown")
                .astype(str)
                .to_numpy()
            )

            phase4_hardware = (
                phase4_plot_df["Hardware"]
                .fillna("Unknown").astype(str).to_numpy()
                if "Hardware" in phase4_plot_df.columns
                else np.array(["Unknown"] * phase4_total)
            )

            phase4_material = (
                phase4_plot_df["Material"]
                .fillna("Unknown").astype(str).to_numpy()
                if "Material" in phase4_plot_df.columns
                else np.array(["Unknown"] * phase4_total)
            )

            phase4_x = phase4_plot_df[
                "Oxygen_Percent"
            ].to_numpy()

            phase4_y = phase4_plot_df[
                "Airflow_cm_s"
            ].to_numpy()

            phase4_z = phase4_plot_df[
                "Gravity_g"
            ].to_numpy()

            phase4_burn = phase4_plot_df[
                "Burn_Time_s"
            ].to_numpy()

            phase4_customdata = np.column_stack([
                phase4_ids,
                phase4_names,
                phase4_burn,
                phase4_hardware,
                phase4_material
            ])

            # ----------------------------------------------------
            # BASE 3D FIGURE
            # ----------------------------------------------------

            phase4_fig = go.Figure()

            # Trace 0: all investigations in background
            phase4_fig.add_trace(
                go.Scatter3d(
                    x=phase4_x,
                    y=phase4_y,
                    z=phase4_z,
                    mode="markers",
                    name="All NASA PSI Records",
                    marker=dict(
                        size=5,
                        color="lightgrey",
                        opacity=0.30
                    ),
                    customdata=phase4_customdata,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]}<br>"
                        "Burn Duration: %{customdata[2]:.2f} s"
                        "<extra>All Records</extra>"
                    )
                )
            )

            # Trace 1: records active at current timeline position
            phase4_initial_mask = (
                phase4_burn <= phase4_thresholds[0]
            )

            phase4_fig.add_trace(
                go.Scatter3d(
                    x=phase4_x[phase4_initial_mask],
                    y=phase4_y[phase4_initial_mask],
                    z=phase4_z[phase4_initial_mask],
                    mode="markers+text" if phase4_show_labels else "markers",
                    name="Timeline Progress",
                    marker=dict(
                        size=8,
                        color=phase4_burn[phase4_initial_mask],
                        colorscale="Turbo",
                        showscale=True,
                        colorbar=dict(
                            title="Burn Time (s)"
                        ),
                        opacity=0.95
                    ),
                    text=phase4_ids[phase4_initial_mask],
                    textposition="top center",
                    customdata=phase4_customdata[
                        phase4_initial_mask
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]}<br>"
                        "Burn Duration: %{customdata[2]:.2f} s<br>"
                        "Hardware: %{customdata[3]}<br>"
                        "Material: %{customdata[4]}"
                        "<extra>Timeline</extra>"
                    )
                )
            )

            # ----------------------------------------------------
            # GENERATE ANIMATION FRAMES
            # ----------------------------------------------------

            phase4_frames = []

            for phase4_i, phase4_time in enumerate(
                phase4_thresholds
            ):

                phase4_mask = (
                    phase4_burn <= phase4_time
                )

                phase4_active_custom = (
                    phase4_customdata[phase4_mask]
                )

                phase4_active_ids = (
                    phase4_ids[phase4_mask]
                )

                phase4_active_burn = (
                    phase4_burn[phase4_mask]
                )

                phase4_frames.append(
                    go.Frame(
                        name=str(phase4_i),
                        data=[
                            go.Scatter3d(
                                x=phase4_x,
                                y=phase4_y,
                                z=phase4_z
                            ),
                            go.Scatter3d(
                                x=phase4_x[phase4_mask],
                                y=phase4_y[phase4_mask],
                                z=phase4_z[phase4_mask],
                                text=phase4_active_ids,
                                customdata=phase4_active_custom,
                                mode=(
                                    "markers+text"
                                    if phase4_show_labels
                                    else "markers"
                                ),
                                marker=dict(
                                    size=8,
                                    color=phase4_active_burn,
                                    colorscale="Turbo",
                                    showscale=True,
                                    opacity=0.95
                                )
                            )
                        ],
                        traces=[0, 1]
                    )
                )

            phase4_fig.frames = phase4_frames

            # ----------------------------------------------------
            # PLAYBACK CONTROLS
            # ----------------------------------------------------

            phase4_fig.update_layout(
                height=650,
                title=(
                    "EigenMind | 4D NASA PSI Investigation Timeline"
                ),
                scene=dict(
                    xaxis_title="Oxygen Concentration (%)",
                    yaxis_title="Airflow (cm/s)",
                    zaxis_title="Gravity (g)",
                    aspectmode="cube"
                ),
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=60
                ),
                legend=dict(
                    orientation="h",
                    y=1.02
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        showactive=False,
                        x=0.05,
                        y=1.12,
                        buttons=[
                            dict(
                                label="▶ Play",
                                method="animate",
                                args=[
                                    None,
                                    {
                                        "frame": {
                                            "duration": 450,
                                            "redraw": True
                                        },
                                        "transition": {
                                            "duration": 150
                                        },
                                        "fromcurrent": True
                                    }
                                ]
                            ),
                            dict(
                                label="⏸ Pause",
                                method="animate",
                                args=[
                                    [None],
                                    {
                                        "frame": {
                                            "duration": 0,
                                            "redraw": False
                                        },
                                        "mode": "immediate",
                                        "transition": {
                                            "duration": 0
                                        }
                                    }
                                ]
                            )
                        ]
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        x=0.12,
                        y=0.02,
                        len=0.82,
                        currentvalue=dict(
                            prefix="Burn-duration threshold: "
                        ),
                        steps=[
                            dict(
                                method="animate",
                                label=f"{t:.1f} s",
                                args=[
                                    [str(i)],
                                    {
                                        "frame": {
                                            "duration": 0,
                                            "redraw": True
                                        },
                                        "mode": "immediate",
                                        "transition": {
                                            "duration": 0
                                        }
                                    }
                                ]
                            )
                            for i, t in enumerate(
                                phase4_thresholds
                            )
                        ]
                    )
                ]
            )

            st.plotly_chart(
                phase4_fig,
                width="stretch",
                key="eigenmind_phase4_timeline"
            )

            # ----------------------------------------------------
            # TIMELINE DATA TABLE
            # ----------------------------------------------------

            st.markdown("### 📋 Investigation Timeline Data")

            phase4_table = phase4_plot_df[
                [
                    "PSI_ID",
                    "Investigation",
                    "Oxygen_Percent",
                    "Airflow_cm_s",
                    "Gravity_g",
                    "Burn_Time_s"
                ]
            ].sort_values(
                by="Burn_Time_s"
            )

            st.dataframe(
                phase4_table,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                label="⬇️ Download Timeline Data (CSV)",
                data=phase4_table.to_csv(index=False).encode("utf-8"),
                file_name="EigenMind_Phase4_Timeline.csv",
                mime="text/csv",
                key="phase4_download_timeline"
            )

            st.caption(
                "Scientific limitation: The animation orders and highlights "
                "investigation records using their reported burn durations. "
                "It does not represent time-resolved flame measurements, "
                "validated CFD, or actual NASA sensor telemetry."
            )

    # ============================================================
    # END OF PHASE 4
    # ============================================================


    # ============================================================
    # PHASE 5: EIGENMIND INTELLIGENCE & MISSION DECISION SUPPORT
    # Integrates existing model outputs and current mission settings.
    # This is decision support, not a flight-safety certification.
    # ============================================================

    st.markdown("---")
    st.subheader("🧠 Phase 5 | Mission Intelligence & Decision Support")
    st.caption(
        "INTEGRATED MISSION BRIEF | Combines the existing hybrid index, "
        "ML estimate, physics score, uncertainty indicator and current "
        "environment settings"
    )

    phase5_score = float(np.clip(hybrid_score, 0, 100))
    phase5_model_gap = abs(float(ml_score) - float(physics_score))

    if phase5_score < 35:
        phase5_level = "LOW"
        phase5_color = "success"
        phase5_summary = (
            "The current modelled conditions fall in the lower score band. "
            "Continue material-specific screening and review the assumptions "
            "before making any operational decision."
        )
    elif phase5_score < 62:
        phase5_level = "ELEVATED"
        phase5_color = "warning"
        phase5_summary = (
            "The current modelled conditions fall in the intermediate score "
            "band. Review oxygen concentration, airflow configuration and "
            "material flammability evidence together."
        )
    else:
        phase5_level = "HIGH"
        phase5_color = "error"
        phase5_summary = (
            "The current modelled conditions fall in the upper score band. "
            "Flag this scenario for detailed engineering review and "
            "material-specific fire-safety assessment."
        )

    phase5_left, phase5_right = st.columns([1.15, 1])

    with phase5_left:
        st.markdown("#### 🎯 Integrated Mission Assessment")
        st.metric(
            "Hybrid Fire Hazard Index",
            f"{phase5_score:.1f} / 100",
            f"{phase5_level} SCREENING BAND"
        )
        if phase5_color == "success":
            st.success(phase5_summary)
        elif phase5_color == "warning":
            st.warning(phase5_summary)
        else:
            st.error(phase5_summary)

        phase5_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=phase5_score,
            number={"suffix": " / 100", "font": {"color": "#E8F0F7"}},
            title={"text": "MODEL-BASED SCREENING INDEX", "font": {"color": "#C8E8F8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#B9CBD8"},
                "bar": {"color": "#62BDEB"},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "#49677E",
                "steps": [
                    {"range": [0, 35], "color": "rgba(99,211,155,0.22)"},
                    {"range": [35, 62], "color": "rgba(242,189,85,0.22)"},
                    {"range": [62, 100], "color": "rgba(239,104,104,0.22)"}
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.8,
                    "value": phase5_score
                }
            }
        ))
        phase5_gauge.update_layout(
            height=270,
            margin=dict(l=20, r=20, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#D8ECF8")
        )
        st.plotly_chart(
            phase5_gauge,
            width="stretch",
            key="eigenmind_phase5_integrated_gauge"
        )

    with phase5_right:
        st.markdown("#### 📡 Current Mission Telemetry")
        phase5_telemetry = pd.DataFrame([
            {"Parameter": "Mission preset", "Current value": str(preset)},
            {"Parameter": "Oxygen concentration", "Current value": f"{o2:.1f} %"},
            {"Parameter": "Cabin pressure", "Current value": f"{pressure:.1f} kPa"},
            {"Parameter": "Airflow speed", "Current value": f"{airflow:.1f} cm/s"},
            {"Parameter": "Airflow direction", "Current value": str(flow_dir)},
            {"Parameter": "Gravity regime", "Current value": f"{gravity_g:.3f} g"},
            {"Parameter": "Partial oxygen pressure", "Current value": f"{(o2 / 100.0) * pressure:.2f} kPa"},
            {"Parameter": "Uncertainty indicator", "Current value": str(conf_text)}
        ])
        st.dataframe(
            phase5_telemetry,
            width="stretch",
            hide_index=True
        )

        st.markdown("#### ⚖️ Model Agreement Check")
        st.metric("ML vs Physics score difference", f"{phase5_model_gap:.1f} points")
        if phase5_model_gap >= 20:
            st.warning(
                "The two model components differ noticeably. Treat the "
                "combined score cautiously and inspect the inputs, model "
                "domain and underlying PSI evidence."
            )
        else:
            st.info(
                "The two model components are relatively close for this "
                "scenario. Agreement does not independently validate the model."
            )

    # Screening flags describe conditions for review; they are not
    # independently validated causal contributions to the hazard score.
    st.markdown("#### 🔎 Environmental Screening Flags")
    phase5_flags = [
        {
            "Factor": "Oxygen concentration",
            "Observed value": f"{o2:.1f}%",
            "Screening note": (
                "Elevated oxygen setting — check material-specific flammability data"
                if o2 >= 26 else "Below the project's 26% review trigger"
            ),
            "Review priority": "Review" if o2 >= 26 else "Routine"
        },
        {
            "Factor": "Airflow configuration",
            "Observed value": f"{airflow:.1f} cm/s, {flow_dir}",
            "Screening note": (
                "Concurrent flow selected — compare with relevant airflow-direction evidence"
                if flow_dir == "Concurrent" else "Opposed flow selected — do not assume flame extinction"
            ),
            "Review priority": "Review" if flow_dir == "Concurrent" else "Check evidence"
        },
        {
            "Factor": "Gravity regime",
            "Observed value": f"{gravity_g:.3f} g",
            "Screening note": (
                "Reduced-gravity regime — buoyancy-driven Earth assumptions may not transfer directly"
                if gravity_g < 0.5 else "Higher-gravity benchmark regime"
            ),
            "Review priority": "Review" if gravity_g < 0.5 else "Reference"
        },
        {
            "Factor": "Cabin pressure",
            "Observed value": f"{pressure:.1f} kPa",
            "Screening note": "Interpret jointly with oxygen fraction and the validated model domain",
            "Review priority": "Check model domain"
        }
    ]
    phase5_flags_df = pd.DataFrame(phase5_flags)
    st.dataframe(
        phase5_flags_df,
        width="stretch",
        hide_index=True
    )

    st.markdown("#### 🧭 Suggested Engineering Review Actions")
    phase5_actions = []
    if o2 >= 26:
        phase5_actions.append(
            "Review the selected atmosphere against material-specific flammability and oxygen-compatibility test evidence."
        )
    if flow_dir == "Concurrent":
        phase5_actions.append(
            "Compare the scenario with relevant concurrent-flow experiments, including the limitations of any single reference test."
        )
    if gravity_g < 0.5:
        phase5_actions.append(
            "Use microgravity or partial-gravity evidence where available; do not directly transfer normal-gravity assumptions."
        )
    if phase5_model_gap >= 20:
        phase5_actions.append(
            "Investigate the ML/physics score disagreement before relying on the hybrid estimate."
        )
    if phase5_score >= 62:
        phase5_actions.append(
            "Escalate the case for a formal, material-specific fire-safety engineering review before mission use."
        )
    if not phase5_actions:
        phase5_actions.append(
            "Retain routine material screening, verify input assumptions and document the scenario before further review."
        )

    for phase5_i, phase5_action in enumerate(phase5_actions, start=1):
        st.markdown(f"**{phase5_i}.** {phase5_action}")

    st.info(
        "Decision-support limitation: These review actions are rule-based prompts "
        "derived from the displayed inputs and project score bands. They are not "
        "NASA procedures, certified mitigation instructions, or flight-readiness approval. "
        "The hybrid index is a project model output, not a NASA-standard pass/fail result."
    )

    # Dataset analogue: nearest row in the available PSI table, using only
    # the four numeric environment fields shared by this app. This is a
    # similarity aid, not proof that the experiment is physically equivalent.
    st.markdown("#### 🧪 Closest Available PSI Dataset Record")
    phase5_match_cols = [
        "Oxygen_Percent", "Pressure_kPa", "Airflow_cm_s", "Gravity_g"
    ]
    phase5_match_required = phase5_match_cols + ["PSI_ID", "Investigation"]
    phase5_match_missing = [
        c for c in phase5_match_required if c not in base_df.columns
    ]

    phase5_match_row = None
    if phase5_match_missing:
        st.caption(
            "Dataset comparison is unavailable because required columns are missing: "
            + ", ".join(phase5_match_missing)
        )
    else:
        phase5_match_df = base_df.copy()
        for phase5_col in phase5_match_cols:
            phase5_match_df[phase5_col] = pd.to_numeric(
                phase5_match_df[phase5_col], errors="coerce"
            )
        phase5_match_df = phase5_match_df.dropna(
            subset=phase5_match_cols
        ).copy()

        if phase5_match_df.empty:
            st.caption("No complete numeric PSI records are available for comparison.")
        else:
            phase5_target = np.array([o2, pressure, airflow, gravity_g], dtype=float)
            phase5_ranges = (
                phase5_match_df[phase5_match_cols].max()
                - phase5_match_df[phase5_match_cols].min()
            ).replace(0, 1).to_numpy(dtype=float)
            phase5_matrix = phase5_match_df[phase5_match_cols].to_numpy(dtype=float)
            phase5_distances = np.sqrt(
                np.sum(((phase5_matrix - phase5_target) / phase5_ranges) ** 2, axis=1)
            )
            phase5_best_index = int(np.argmin(phase5_distances))
            phase5_match_row = phase5_match_df.iloc[phase5_best_index]
            phase5_match_distance = float(phase5_distances[phase5_best_index])

            phase5_match_details = {
                "PSI ID": str(phase5_match_row["PSI_ID"]),
                "Investigation": str(phase5_match_row["Investigation"]),
                "Oxygen": f"{phase5_match_row['Oxygen_Percent']:.2f}%",
                "Pressure": f"{phase5_match_row['Pressure_kPa']:.2f} kPa",
                "Airflow": f"{phase5_match_row['Airflow_cm_s']:.2f} cm/s",
                "Gravity": f"{phase5_match_row['Gravity_g']:.3f} g",
                "Normalized distance (lower = closer)": f"{phase5_match_distance:.3f}"
            }
            st.dataframe(
                pd.DataFrame(
                    list(phase5_match_details.items()),
                    columns=["Field", "Dataset record"]
                ),
                width="stretch",
                hide_index=True
            )
            st.caption(
                "The closest record is selected by normalized distance across oxygen, "
                "pressure, airflow and gravity. It is a dataset-neighbour reference, "
                "not a validated analogue or a prediction of the current mission outcome."
            )

    # Build a compact, downloadable mission brief from values already shown.
    phase5_report_lines = [
        "EIGENMIND | MISSION INTELLIGENCE BRIEF",
        "=" * 48,
        f"Mission preset: {preset}",
        f"Screening band: {phase5_level}",
        f"Hybrid Fire Hazard Index: {phase5_score:.2f}/100",
        f"ML surrogate score: {float(ml_score):.2f}/100",
        f"Physics scaling score: {float(physics_score):.2f}/100",
        f"ML/physics difference: {phase5_model_gap:.2f} points",
        f"Uncertainty indicator: {conf_text}",
        "",
        "ENVIRONMENT",
        f"Oxygen: {o2:.2f}%",
        f"Pressure: {pressure:.2f} kPa",
        f"Airflow: {airflow:.2f} cm/s",
        f"Flow direction: {flow_dir}",
        f"Gravity: {gravity_g:.3f} g",
        f"Partial oxygen pressure: {(o2 / 100.0) * pressure:.2f} kPa",
        "",
        "ASSESSMENT",
        phase5_summary,
        "",
        "ENGINEERING REVIEW PROMPTS",
        *[f"{i}. {action}" for i, action in enumerate(phase5_actions, start=1)],
        "",
        "LIMITATIONS",
        "Project decision-support output only; not NASA certification, a NASA-standard pass/fail result,",
        "validated CFD, calibrated sensor telemetry, or flight-readiness approval.",
        "Verify all source data, model assumptions and material-specific evidence independently."
    ]
    if phase5_match_row is not None:
        phase5_report_lines.extend([
            "",
            "CLOSEST DATASET RECORD (NOT A VALIDATED PHYSICAL ANALOGUE)",
            f"PSI ID: {phase5_match_row['PSI_ID']}",
            f"Investigation: {phase5_match_row['Investigation']}"
        ])

    st.download_button(
        label="⬇️ Download Mission Intelligence Brief (TXT)",
        data="\n".join(phase5_report_lines).encode("utf-8"),
        file_name="EigenMind_Mission_Intelligence_Brief.txt",
        mime="text/plain",
        key="eigenmind_phase5_download_brief"
    )

    # ============================================================
    # PHASE 6: PROFESSIONAL PDF MISSION REPORT GENERATOR
    # Creates a downloadable, reproducible project assessment report.
    # ============================================================

    st.markdown("---")
    st.subheader("📄 Phase 6 | Professional Mission Report Generator")
    st.caption(
        "EXPORTABLE TECHNICAL REPORT | Mission inputs, model outputs, "
        "screening flags, PSI dataset evidence, references and limitations"
    )

    phase6_include_dataset_refs = st.checkbox(
        "Include available NASA PSI dataset DOI references (up to 12)",
        value=True,
        key="eigenmind_phase6_include_refs"
    )

    try:
        from io import BytesIO
        from xml.sax.saxutils import escape
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            KeepTogether
        )
        from reportlab.graphics.shapes import Drawing
        from reportlab.graphics.charts.barcharts import VerticalBarChart

        phase6_buffer = BytesIO()
        phase6_doc = SimpleDocTemplate(
            phase6_buffer,
            pagesize=A4,
            rightMargin=17 * mm,
            leftMargin=17 * mm,
            topMargin=19 * mm,
            bottomMargin=18 * mm,
            title="EigenMind Mission Fire-Safety Screening Report",
            author="EigenMind Project Dashboard"
        )

        phase6_styles = getSampleStyleSheet()
        phase6_styles.add(ParagraphStyle(
            name="EigenTitle", parent=phase6_styles["Title"],
            fontName="Helvetica-Bold", fontSize=19, leading=23,
            textColor=colors.HexColor("#17324D"), alignment=TA_CENTER,
            spaceAfter=5 * mm
        ))
        phase6_styles.add(ParagraphStyle(
            name="EigenSubtitle", parent=phase6_styles["Normal"],
            fontSize=8.5, leading=12, textColor=colors.HexColor("#526779"),
            alignment=TA_CENTER, spaceAfter=6 * mm
        ))
        phase6_styles.add(ParagraphStyle(
            name="EigenSection", parent=phase6_styles["Heading2"],
            fontSize=12, leading=15, textColor=colors.HexColor("#145477"),
            spaceBefore=4 * mm, spaceAfter=2 * mm
        ))
        phase6_styles.add(ParagraphStyle(
            name="EigenBody", parent=phase6_styles["BodyText"],
            fontSize=8.5, leading=12, spaceAfter=1.5 * mm
        ))
        phase6_styles.add(ParagraphStyle(
            name="EigenSmall", parent=phase6_styles["BodyText"],
            fontSize=7, leading=9, textColor=colors.HexColor("#526779")
        ))

        phase6_story = []
        phase6_story.append(Paragraph(
            "EIGENMIND | MISSION FIRE-SAFETY SCREENING REPORT",
            phase6_styles["EigenTitle"]
        ))
        phase6_story.append(Paragraph(
            "Integrated project-model assessment and NASA PSI dataset context",
            phase6_styles["EigenSubtitle"]
        ))
        phase6_story.append(Paragraph("1. Executive Summary", phase6_styles["EigenSection"]))
        phase6_story.append(Paragraph(
            f"Mission preset: <b>{escape(str(preset))}</b>. "
            f"Project screening band: <b>{escape(str(phase5_level))}</b>. "
            f"Hybrid Fire Hazard Index: <b>{phase5_score:.2f}/100</b>. "
            f"{escape(str(phase5_summary))}",
            phase6_styles["EigenBody"]
        ))

        phase6_story.append(Paragraph("2. Mission Environment and Model Outputs", phase6_styles["EigenSection"]))
        phase6_metric_rows = [
            ["Parameter", "Reported value", "Parameter", "Reported value"],
            ["Oxygen concentration", f"{o2:.2f} %", "Cabin pressure", f"{pressure:.2f} kPa"],
            ["Airflow speed", f"{airflow:.2f} cm/s", "Flow direction", str(flow_dir)],
            ["Gravity regime", f"{gravity_g:.4f} g", "Partial oxygen pressure", f"{(o2 / 100.0) * pressure:.2f} kPa"],
            ["Hybrid index", f"{phase5_score:.2f} / 100", "Screening band", str(phase5_level)],
            ["ML score", f"{float(ml_score):.2f} / 100", "Physics score", f"{float(physics_score):.2f} / 100"],
            ["ML-physics difference", f"{phase5_model_gap:.2f} points", "Uncertainty indicator", str(conf_text)]
        ]
        phase6_metric_table = Table(phase6_metric_rows, colWidths=[39*mm, 43*mm, 39*mm, 43*mm], repeatRows=1)
        phase6_metric_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F1F5F8")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C8D3")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
        ]))
        phase6_story.append(phase6_metric_table)
        phase6_story.append(Spacer(1, 3 * mm))

        phase6_story.append(Paragraph("3. Model Score Comparison", phase6_styles["EigenSection"]))
        phase6_chart = Drawing(165*mm, 58*mm)
        phase6_barchart = VerticalBarChart()
        phase6_barchart.x = 13*mm
        phase6_barchart.y = 8*mm
        phase6_barchart.width = 135*mm
        phase6_barchart.height = 43*mm
        phase6_barchart.data = [[
            float(np.clip(phase5_score, 0, 100)),
            float(np.clip(float(ml_score), 0, 100)),
            float(np.clip(float(physics_score), 0, 100))
        ]]
        phase6_barchart.categoryAxis.categoryNames = ["Hybrid", "ML", "Physics"]
        phase6_barchart.categoryAxis.labels.fontSize = 8
        phase6_barchart.valueAxis.valueMin = 0
        phase6_barchart.valueAxis.valueMax = 100
        phase6_barchart.valueAxis.valueStep = 20
        phase6_barchart.valueAxis.labels.fontSize = 7
        phase6_barchart.bars[0].fillColor = colors.HexColor("#2386B8")
        phase6_barchart.barWidth = 18
        phase6_barchart.groupSpacing = 18
        phase6_chart.add(phase6_barchart)
        phase6_story.append(phase6_chart)
        phase6_story.append(Paragraph(
            "Note: These are the dashboard's model outputs on a 0-100 scale; "
            "the chart does not represent measured fire-test outcomes.",
            phase6_styles["EigenSmall"]
        ))

        phase6_story.append(Paragraph("4. Environmental Screening Flags", phase6_styles["EigenSection"]))
        phase6_flag_rows = [["Factor", "Observed value", "Screening note", "Priority"]]
        for phase6_flag in phase5_flags:
            phase6_flag_rows.append([
                str(phase6_flag["Factor"]),
                str(phase6_flag["Observed value"]),
                Paragraph(escape(str(phase6_flag["Screening note"])), phase6_styles["EigenSmall"]),
                str(phase6_flag["Review priority"])
            ])
        phase6_flag_table = Table(phase6_flag_rows, colWidths=[32*mm, 31*mm, 91*mm, 20*mm], repeatRows=1)
        phase6_flag_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#145477")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B9C8D3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F8")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4)
        ]))
        phase6_story.append(phase6_flag_table)

        phase6_story.append(Paragraph("5. Suggested Engineering Review Actions", phase6_styles["EigenSection"]))
        for phase6_num, phase6_action in enumerate(phase5_actions, start=1):
            phase6_story.append(Paragraph(
                f"{phase6_num}. {escape(str(phase6_action))}",
                phase6_styles["EigenBody"]
            ))

        phase6_story.append(Paragraph("6. NASA PSI Dataset Context", phase6_styles["EigenSection"]))
        if phase5_match_row is not None:
            phase6_match_lines = [
                f"Closest numerical dataset neighbour: PSI ID {escape(str(phase5_match_row['PSI_ID']))}",
                f"Investigation: {escape(str(phase5_match_row['Investigation']))}",
                f"Oxygen: {float(phase5_match_row['Oxygen_Percent']):.2f}%; "
                f"pressure: {float(phase5_match_row['Pressure_kPa']):.2f} kPa; "
                f"airflow: {float(phase5_match_row['Airflow_cm_s']):.2f} cm/s; "
                f"gravity: {float(phase5_match_row['Gravity_g']):.4f} g"
            ]
            if "Hardware" in phase5_match_row.index:
                phase6_match_lines.append(f"Hardware: {escape(str(phase5_match_row['Hardware']))}")
            if "Material" in phase5_match_row.index:
                phase6_match_lines.append(f"Material: {escape(str(phase5_match_row['Material']))}")
            if "Key_Finding" in phase5_match_row.index:
                phase6_match_lines.append(f"Dataset key finding: {escape(str(phase5_match_row['Key_Finding']))}")
            for phase6_line in phase6_match_lines:
                phase6_story.append(Paragraph(phase6_line, phase6_styles["EigenBody"]))
            if "DOI" in phase5_match_row.index and pd.notna(phase5_match_row["DOI"]):
                phase6_doi = str(phase5_match_row["DOI"]).strip()
                if phase6_doi and phase6_doi.lower() != "nan":
                    phase6_doi_url = phase6_doi if phase6_doi.startswith("http") else "https://doi.org/" + phase6_doi
                    phase6_story.append(Paragraph(
                        "Matched record DOI: "
                        f'<link href="{escape(phase6_doi_url)}" color="#1476A5">{escape(phase6_doi)}</link>',
                        phase6_styles["EigenBody"]
                    ))
        else:
            phase6_story.append(Paragraph(
                "No nearest PSI dataset record was available from the current dataset fields.",
                phase6_styles["EigenBody"]
            ))

        if phase6_include_dataset_refs and "DOI" in base_df.columns:
            phase6_refs = base_df[[c for c in ["PSI_ID", "Investigation", "DOI"] if c in base_df.columns]].copy()
            phase6_refs["DOI"] = phase6_refs["DOI"].fillna("").astype(str).str.strip()
            phase6_refs = phase6_refs[
                (phase6_refs["DOI"] != "") & (phase6_refs["DOI"].str.lower() != "nan")
            ].drop_duplicates(subset=["DOI"]).head(12)
            if not phase6_refs.empty:
                phase6_story.append(Paragraph(
                    "Available DOI entries from the loaded dataset (up to 12 unique records):",
                    phase6_styles["EigenBody"]
                ))
                for _, phase6_ref in phase6_refs.iterrows():
                    phase6_doi_value = str(phase6_ref["DOI"])
                    phase6_doi_href = phase6_doi_value if phase6_doi_value.startswith("http") else "https://doi.org/" + phase6_doi_value
                    phase6_ref_label = " | ".join([
                        str(phase6_ref.get("PSI_ID", "")),
                        str(phase6_ref.get("Investigation", ""))
                    ])
                    phase6_story.append(Paragraph(
                        f'{escape(phase6_ref_label)}: '
                        f'<link href="{escape(phase6_doi_href)}" color="#1476A5">{escape(phase6_doi_value)}</link>',
                        phase6_styles["EigenSmall"]
                    ))

        phase6_story.append(Paragraph("7. Scope, Assumptions and Limitations", phase6_styles["EigenSection"]))
        phase6_limitations = [
            "This report documents a project-level screening calculation and its current inputs; it is not a NASA certification, NASA-standard pass/fail determination, flight-readiness decision or operational procedure.",
            "The hybrid index, ML score, physics score, score bands and review prompts are outputs or rules implemented in this project. They require independent validation against suitable, documented experimental evidence before scientific or operational use.",
            "A nearest-neighbour PSI record is a numerical dataset comparison across oxygen, pressure, airflow and gravity. It is not proof of physical equivalence, causation or a prediction of the current scenario.",
            "The Phase 1 flame visualizer, Phase 2 chamber digital twin, Phase 3 investigation explorer and Phase 4 duration playback contain conceptual or dataset-driven visualizations. They should not be interpreted as calibrated sensor telemetry, validated CFD or continuous time-resolved experimental reconstruction.",
            "Material flammability and fire response depend on material-specific test evidence and the applicable environment. Oxygen concentration alone cannot establish compliance or safety."
        ]
        for phase6_limit in phase6_limitations:
            phase6_story.append(Paragraph(
    escape(phase6_limit),
    phase6_styles["EigenBody"],
    bulletText="•"
))

        phase6_story.append(Spacer(1, 3*mm))
        phase6_story.append(Paragraph(
            "Report generated by the EigenMind dashboard. Confirm source metadata, "
            "dataset provenance, model version and assumptions before external submission.",
            phase6_styles["EigenSmall"]
        ))

        def phase6_page_footer(canvas, doc):
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#C4D0D9"))
            canvas.setLineWidth(0.5)
            canvas.line(17*mm, 13*mm, A4[0] - 17*mm, 13*mm)
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(colors.HexColor("#526779"))
            canvas.drawString(17*mm, 8*mm, "EIGENMIND | PROJECT SCREENING REPORT")
            canvas.drawRightString(A4[0] - 17*mm, 8*mm, f"Page {doc.page}")
            canvas.restoreState()

        phase6_doc.build(
            phase6_story,
            onFirstPage=phase6_page_footer,
            onLaterPages=phase6_page_footer
        )
        phase6_pdf_bytes = phase6_buffer.getvalue()
        st.success("PDF report is ready. Review the report contents before sharing it.")
        st.download_button(
            label="⬇️ Download Professional Mission Report (PDF)",
            data=phase6_pdf_bytes,
            file_name="EigenMind_Professional_Mission_Report.pdf",
            mime="application/pdf",
            key="eigenmind_phase6_download_pdf"
        )

    except ImportError:
        st.error(
            "PDF export needs ReportLab. Install it in your active environment "
            "with: pip install reportlab, then restart Streamlit."
        )
    except Exception as phase6_error:
        st.error(f"The PDF report could not be generated: {phase6_error}")
        st.caption(
            "The dashboard remains available. Share the full terminal traceback "
            "if this report-generation issue continues."
        )

    # ============================================================
    # END OF PHASE 6
    # ============================================================

    # ============================================================
    # END OF PHASE 5
    # ============================================================


    # ============================================================
    # PHASE 7: SCIENTIFIC VALIDATION & NASA PSI DATASET AUDIT
    # Data-quality checks and transparent validation readiness.
    # This is an audit, not proof of model scientific validity.
    # ============================================================

    st.markdown("---")
    st.subheader("🔬 Phase 7 | Scientific Validation & NASA PSI Dataset Audit")
    st.caption(
        "DATA QUALITY + MODEL TRANSPARENCY | Audits the records loaded into "
        "EigenMind and identifies evidence still needed for independent validation"
    )

    phase7_df = base_df.copy()
    phase7_rows = len(phase7_df)
    phase7_cols = len(phase7_df.columns)
    phase7_missing_total = int(phase7_df.isna().sum().sum())
    phase7_duplicate_rows = int(phase7_df.duplicated().sum())
    phase7_missing_pct = (
        100.0 * phase7_missing_total / (phase7_rows * phase7_cols)
        if phase7_rows and phase7_cols else 0.0
    )

    p7a, p7b, p7c, p7d = st.columns(4)
    p7a.metric("Dataset records", f"{phase7_rows:,}")
    p7b.metric("Dataset fields", f"{phase7_cols:,}")
    p7c.metric("Missing cells", f"{phase7_missing_total:,}", f"{phase7_missing_pct:.2f}%")
    p7d.metric("Duplicate rows", f"{phase7_duplicate_rows:,}")

    st.markdown("#### 1. Dataset completeness")
    phase7_missing = phase7_df.isna().sum().rename("Missing values").to_frame()
    phase7_missing["Missing (%)"] = (
        phase7_missing["Missing values"] / max(phase7_rows, 1) * 100
    ).round(2)
    phase7_missing["Data type"] = phase7_df.dtypes.astype(str)
    phase7_missing = phase7_missing.sort_values(
        ["Missing values", "Missing (%)"], ascending=False
    )
    st.dataframe(phase7_missing, width="stretch")

    phase7_quality_flags = []
    if phase7_rows == 0:
        phase7_quality_flags.append(("error", "The NASA PSI dataset is empty."))
    if phase7_duplicate_rows:
        phase7_quality_flags.append(("warning", f"{phase7_duplicate_rows} fully duplicated rows were detected."))
    if phase7_missing_total:
        phase7_quality_flags.append(("warning", f"{phase7_missing_total} missing cells were detected. Review the field-level table before analysis."))
    if not phase7_missing_total and not phase7_duplicate_rows and phase7_rows:
        phase7_quality_flags.append(("success", "No missing cells or fully duplicated rows were detected in the loaded table."))

    phase7_expected = [
        "PSI_ID", "Investigation", "Hardware", "Material", "Fuel_State",
        "Oxygen_Percent", "Pressure_kPa", "Airflow_cm_s", "Flow_Direction",
        "Gravity_g", "Burn_Time_s", "DOI", "Key_Finding"
    ]
    phase7_absent = [c for c in phase7_expected if c not in phase7_df.columns]
    if phase7_absent:
        phase7_quality_flags.append(("warning", "Expected PSI fields absent from this dataset: " + ", ".join(phase7_absent)))

    for phase7_level, phase7_message in phase7_quality_flags:
        if phase7_level == "error":
            st.error(phase7_message)
        elif phase7_level == "warning":
            st.warning(phase7_message)
        else:
            st.success(phase7_message)

    st.markdown("#### 2. Numerical range screening")
    phase7_numeric_candidates = [
        "Oxygen_Percent", "Pressure_kPa", "Airflow_cm_s", "Gravity_g", "Burn_Time_s"
    ]
    phase7_range_rows = []
    phase7_range_rules = {
        "Oxygen_Percent": (0, 100, "Expected percentage: 0–100"),
        "Pressure_kPa": (0, None, "Expected positive pressure"),
        "Airflow_cm_s": (0, None, "Expected non-negative airflow speed"),
        "Gravity_g": (0, None, "Magnitude in g; verify source convention"),
        "Burn_Time_s": (0, None, "Expected non-negative duration")
    }
    for phase7_col in phase7_numeric_candidates:
        if phase7_col not in phase7_df.columns:
            continue
        phase7_values = pd.to_numeric(phase7_df[phase7_col], errors="coerce")
        phase7_non_numeric = int(phase7_df[phase7_col].notna().sum() - phase7_values.notna().sum())
        phase7_invalid = 0
        phase7_low, phase7_high, phase7_rule = phase7_range_rules[phase7_col]
        if phase7_values.notna().any():
            phase7_invalid = int((phase7_values < phase7_low).sum())
            if phase7_high is not None:
                phase7_invalid += int((phase7_values > phase7_high).sum())
        phase7_range_rows.append({
            "Field": phase7_col,
            "Valid numeric values": int(phase7_values.notna().sum()),
            "Non-numeric values": phase7_non_numeric,
            "Min": phase7_values.min() if phase7_values.notna().any() else np.nan,
            "Max": phase7_values.max() if phase7_values.notna().any() else np.nan,
            "Rule-screen flags": phase7_invalid,
            "Screening rule": phase7_rule
        })
    if phase7_range_rows:
        phase7_range_df = pd.DataFrame(phase7_range_rows)
        st.dataframe(phase7_range_df, width="stretch")
        if int(phase7_range_df["Rule-screen flags"].sum()) > 0:
            st.warning("Some numerical values fall outside simple screening rules. Verify units, sign conventions and source metadata; do not automatically delete these records.")
        else:
            st.success("No values violated the simple numerical screening rules for the available fields. This does not establish physical validity.")
    else:
        phase7_range_df = pd.DataFrame()
        st.info("No expected numerical fields were available for range screening.")

    st.markdown("#### 3. NASA PSI reference coverage")
    if "DOI" in phase7_df.columns:
        phase7_doi = phase7_df["DOI"].fillna("").astype(str).str.strip()
        phase7_doi_present = phase7_doi.ne("") & phase7_doi.str.lower().ne("nan")
        phase7_doi_count = int(phase7_doi_present.sum())
        phase7_unique_doi = int(phase7_doi[phase7_doi_present].nunique())
        p7d1, p7d2, p7d3 = st.columns(3)
        p7d1.metric("Records with DOI text", f"{phase7_doi_count:,}")
        p7d2.metric("Unique DOI strings", f"{phase7_unique_doi:,}")
        p7d3.metric("DOI coverage", f"{(100*phase7_doi_count/max(phase7_rows,1)):.1f}%")
        st.caption("DOI coverage counts non-empty DOI fields; it does not verify that a DOI resolves or that it supports the associated record.")
    else:
        phase7_doi_count = 0
        phase7_unique_doi = 0
        st.warning("The dataset does not contain a DOI column, so reference coverage cannot be assessed.")

    st.markdown("#### 4. Investigation and material overview")
    phase7_summary_cols = [c for c in ["Investigation", "Hardware", "Material", "Fuel_State", "Flow_Direction"] if c in phase7_df.columns]
    if phase7_summary_cols:
        phase7_summary_field = st.selectbox(
            "Choose a category to summarize",
            phase7_summary_cols,
            key="eigenmind_phase7_summary_field"
        )
        phase7_category_summary = (
            phase7_df[phase7_summary_field].fillna("Not specified").astype(str)
            .value_counts(dropna=False).head(20).rename_axis(phase7_summary_field)
            .reset_index(name="Record count")
        )
        st.dataframe(phase7_category_summary, width="stretch")
    else:
        phase7_category_summary = pd.DataFrame()
        st.info("No categorical investigation fields were available for this summary.")

    st.markdown("#### 5. Model transparency and validation readiness")
    phase7_model_checks = []
    phase7_model_checks.append({
        "Validation item": "Model artifact loaded",
        "Current evidence": "Yes — dashboard startup completed with a loaded model object",
        "Status": "Loaded; performance not established"
    })
    phase7_model_checks.append({
        "Validation item": "Independent held-out test set",
        "Current evidence": "Not demonstrated by this dashboard audit",
        "Status": "Required before reporting generalization performance"
    })
    phase7_model_checks.append({
        "Validation item": "Ground-truth target and units",
        "Current evidence": "Must be traced to the original dataset and model-training pipeline",
        "Status": "Verify documentation"
    })
    phase7_model_checks.append({
        "Validation item": "External experimental validation",
        "Current evidence": "Not established by dataset completeness checks",
        "Status": "Required for scientific claims"
    })
    phase7_model_checks.append({
        "Validation item": "NASA standard compliance",
        "Current evidence": "Not evaluated by the project score or this audit",
        "Status": "Requires applicable standard, test protocol and qualified review"
    })
    st.dataframe(pd.DataFrame(phase7_model_checks), width="stretch", hide_index=True)

    st.info(
        "Interpretation boundary: a clean dataset audit does not prove that the "
        "dataset is representative, that the model is accurate, or that the "
        "hybrid score predicts real fire behaviour. Report test-set metrics only "
        "when a documented independent evaluation has actually been performed."
    )

    st.markdown("#### 6. Download the audit evidence")
    from io import StringIO
    phase7_audit_buffer = StringIO()
    phase7_df.to_csv(phase7_audit_buffer, index=False)
    st.download_button(
        label="⬇️ Download audited NASA PSI dataset (CSV)",
        data=phase7_audit_buffer.getvalue(),
        file_name="EigenMind_NASA_PSI_Dataset_Audit.csv",
        mime="text/csv",
        key="eigenmind_phase7_download_dataset"
    )
    phase7_report_lines = [
        "EIGENMIND | SCIENTIFIC VALIDATION & DATASET AUDIT",
        f"Dataset records: {phase7_rows}",
        f"Dataset fields: {phase7_cols}",
        f"Missing cells: {phase7_missing_total}",
        f"Duplicate rows: {phase7_duplicate_rows}",
        f"Records with DOI text: {phase7_doi_count}",
        "",
        "IMPORTANT LIMITATIONS:",
        "This report summarizes data-quality checks only.",
        "It does not establish model accuracy, external validity, NASA certification or operational fire safety.",
        "Numerical range rules are basic screening checks and must be reviewed against source units and metadata."
    ]
    st.download_button(
        label="⬇️ Download Phase 7 audit summary (TXT)",
        data="\n".join(phase7_report_lines),
        file_name="EigenMind_Phase7_Audit_Summary.txt",
        mime="text/plain",
        key="eigenmind_phase7_download_summary"
    )

    st.markdown("#### 7. Final scientific-readiness checklist")
    phase7_checklist = [
        "Verify every dataset field against its original NASA PSI source and units.",
        "Document dataset provenance, collection conditions and preprocessing steps.",
        "Separate training, validation and test records without leakage (for subject- or experiment-linked data, split by the appropriate group).",
        "Report MAE, RMSE and R² on a genuinely held-out test set where the target and metric are appropriate.",
        "Compare against transparent baseline models and document hyperparameter selection.",
        "Perform error analysis across oxygen, pressure, airflow, gravity and material regimes where data support it.",
        "Document uncertainty, applicability limits and all assumptions.",
        "Have domain experts review scientific claims and any safety-related interpretation."
    ]
    for phase7_i, phase7_item in enumerate(phase7_checklist):
        st.checkbox(
            phase7_item,
            value=False,
            key=f"eigenmind_phase7_check_{phase7_i}"
        )

    st.caption(
        "PHASE 7 NOTE: This module performs transparent dataset-quality and "
        "validation-readiness checks. It does not independently validate the "
        "combustion model, reproduce NASA experiments, or certify compliance "
        "with NASA-STD-6001 or any other safety standard."
    )

    # ============================================================
    # END OF PHASE 7
    # ============================================================


# ============================================================
# PHASE 8: ADVANCED SCIENTIFIC MODEL VALIDATION
# Reproducible evaluation of the existing Random Forest model
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import plotly.graph_objects as go

st.markdown("---")
st.subheader("🧪 Phase 8 | Advanced Model Validation")
st.caption(
    "RANDOM FOREST PERFORMANCE | Residual Analysis | "
    "Feature Importance | PSI Investigation Audit"
)

phase8_features = [
    "Oxygen_Percent",
    "Pressure_kPa",
    "Airflow_cm_s",
    "Is_Concurrent",
    "Gravity_g"
]

phase8_target = "Fire_Hazard_Score"

try:
    phase8_df = pd.read_csv("nasa_parametric_grid.csv")

    phase8_missing = [
        c for c in phase8_features + [phase8_target]
        if c not in phase8_df.columns
    ]

    if phase8_missing:
        st.error(
            "Missing required model columns: "
            + ", ".join(phase8_missing)
        )

    elif len(phase8_df) < 10:
        st.warning("Insufficient records for model evaluation.")

    else:
        phase8_clean = phase8_df.dropna(
            subset=phase8_features + [phase8_target]
        ).copy()

        X = phase8_clean[phase8_features]
        y = phase8_clean[phase8_target]

        # Reproduce the original model's train/test split.
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Evaluate the already-loaded model.
        phase8_predictions = model.predict(X_test)

        phase8_mae = mean_absolute_error(
            y_test, phase8_predictions
        )

        phase8_mse = mean_squared_error(
            y_test, phase8_predictions
        )

        phase8_rmse = np.sqrt(phase8_mse)

        phase8_r2 = r2_score(
            y_test, phase8_predictions
        )

        st.markdown("### 1. Regression Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("MAE", f"{phase8_mae:.4f}")
        c2.metric("MSE", f"{phase8_mse:.4f}")
        c3.metric("RMSE", f"{phase8_rmse:.4f}")
        c4.metric("R²", f"{phase8_r2:.4f}")

        st.caption(
            "These metrics measure approximation of the "
            "synthetically generated physics-score target, "
            "not experimentally measured fire behaviour."
        )

        # ----------------------------------------------------
        # 2. Actual vs predicted
        # ----------------------------------------------------

        st.markdown("### 2. Actual vs Predicted Values")

        phase8_plot = go.Figure()

        phase8_plot.add_trace(
            go.Scatter(
                x=y_test,
                y=phase8_predictions,
                mode="markers",
                name="Test predictions"
            )
        )

        phase8_min = float(
            min(y_test.min(), phase8_predictions.min())
        )

        phase8_max = float(
            max(y_test.max(), phase8_predictions.max())
        )

        phase8_plot.add_trace(
            go.Scatter(
                x=[phase8_min, phase8_max],
                y=[phase8_min, phase8_max],
                mode="lines",
                name="Ideal prediction"
            )
        )

        phase8_plot.update_layout(
            xaxis_title="Physics-generated target score",
            yaxis_title="Random Forest predicted score",
            height=450
        )

        st.plotly_chart(
            phase8_plot,
            width="stretch"
        )

        # ----------------------------------------------------
        # 3. Residual analysis
        # ----------------------------------------------------

        st.markdown("### 3. Residual Error Analysis")

        phase8_residuals = (
            y_test.to_numpy() - phase8_predictions
        )

        phase8_residual_plot = go.Figure()

        phase8_residual_plot.add_trace(
            go.Scatter(
                x=phase8_predictions,
                y=phase8_residuals,
                mode="markers",
                name="Residuals"
            )
        )

        phase8_residual_plot.add_hline(
            y=0,
            line_dash="dash"
        )

        phase8_residual_plot.update_layout(
            xaxis_title="Predicted score",
            yaxis_title="Residual (actual - predicted)",
            height=400
        )

        st.plotly_chart(
            phase8_residual_plot,
            width="stretch"
        )

        st.metric(
            "Mean residual",
            f"{np.mean(phase8_residuals):.4f}"
        )

        # ----------------------------------------------------
        # 4. Random Forest feature importance
        # ----------------------------------------------------

        st.markdown("### 4. Model Feature Importance")

        if hasattr(model, "feature_importances_"):

            phase8_importance = pd.DataFrame({
                "Feature": phase8_features,
                "Importance": model.feature_importances_
            }).sort_values(
                "Importance",
                ascending=True
            )

            phase8_importance_plot = go.Figure(
                go.Bar(
                    x=phase8_importance["Importance"],
                    y=phase8_importance["Feature"],
                    orientation="h"
                )
            )

            phase8_importance_plot.update_layout(
                xaxis_title="Relative feature importance",
                yaxis_title="Input feature",
                height=400
            )

            st.plotly_chart(
                phase8_importance_plot,
                width="stretch"
            )

            st.dataframe(
                phase8_importance.sort_values(
                    "Importance",
                    ascending=False
                ),
                width="stretch",
                hide_index=True
            )

        else:
            st.info(
                "Feature importance is unavailable for this model."
            )

        # ----------------------------------------------------
        # 5. Investigation-wise evaluation
        # ----------------------------------------------------

        st.markdown("### 5. PSI Investigation-wise Evaluation")

        phase8_results = phase8_clean.loc[
            X_test.index,
            ["PSI_ID"]
        ].copy()

        phase8_results["Actual_Score"] = y_test
        phase8_results["Predicted_Score"] = phase8_predictions
        phase8_results["Absolute_Error"] = np.abs(
            phase8_residuals
        )

        phase8_group_summary = (
            phase8_results.groupby("PSI_ID")
            .agg(
                Test_Records=("Actual_Score", "count"),
                MAE=("Absolute_Error", "mean"),
                Mean_Actual=("Actual_Score", "mean"),
                Mean_Predicted=("Predicted_Score", "mean")
            )
            .reset_index()
        )

        st.dataframe(
            phase8_group_summary,
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------------------
        # 6. Download test predictions
        # ----------------------------------------------------

        st.markdown("### 6. Export Validation Results")

        phase8_csv = phase8_results.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Test Prediction Results",
            data=phase8_csv,
            file_name="EigenMind_Phase8_Test_Results.csv",
            mime="text/csv",
            key="eigenmind_phase8_test_download"
        )

        # ----------------------------------------------------
        # 7. Scientific interpretation
        # ----------------------------------------------------

        st.markdown("### 7. Scientific Validation Status")

        st.warning(
            """
            IMPORTANT SCIENTIFIC LIMITATION

            The training target Fire_Hazard_Score is generated
            using the project's own physics-based equation.

            The Random Forest therefore approximates that
            equation using synthetic samples.

            The reported test metrics are not independent
            experimental validation of spacecraft fire behaviour.

            Further validation requires independently sourced
            experimental measurements, documented target values,
            and appropriate investigation-level data splitting.
            """
        )

        st.info(
            "Phase 8 evaluates the existing trained model. "
            "It does not retrain or replace the model."
        )

except FileNotFoundError:
    st.error(
        "The nasa_parametric_grid.csv file was not found. "
        "Keep it in the same folder as app.py."
    )

except Exception as phase8_error:
    st.error(
        f"Phase 8 evaluation error: {phase8_error}"
    )

# ============================================================
# END OF PHASE 81
# ============================================================



# ---------------- TAB 2: RANK & COMPARE ----------------
with tab2:
    st.subheader("📊 NASA PSI Investigation Comparison")

    st.caption(
        "Source-linked investigation records are compared using their "
        "available experimental information. Missing source parameters "
        "are not estimated or treated as measured values."
    )

    ranked_df = base_df.copy().reset_index(drop=True)

    # Only PSI-98 SAFFIRE-I records are currently included in this
    # illustrative numerical comparison. Other investigations are
    # excluded until their parameters are verified and comparable.
    def get_comparison_score(row):
        if str(row["PSI_ID"]).strip() != "PSI-98":
            return np.nan

        direction = str(row["Flow_Direction"]).strip().lower()

        if direction == "concurrent":
            is_concurrent = 1
        elif direction == "opposed":
            is_concurrent = 0
        else:
            return np.nan

        numeric_fields = [
            "Oxygen_Percent",
            "Pressure_kPa",
            "Airflow_cm_s",
            "Gravity_g"
        ]

        values = pd.to_numeric(
            row[numeric_fields],
            errors="coerce"
        )

        if values.isna().any():
            return np.nan

        try:
            return calculate_physics_hazard_score(
                values["Oxygen_Percent"],
                values["Pressure_kPa"],
                values["Airflow_cm_s"],
                is_concurrent,
                values["Gravity_g"]
            )
        except (ValueError, TypeError):
            return np.nan

    ranked_df["Illustrative_Project_Score"] = ranked_df.apply(
        get_comparison_score,
        axis=1
    )

    ranked_df["Comparison_Status"] = np.where(
        ranked_df["Illustrative_Project_Score"].notna(),
        "Illustrative PSI-98 comparison only",
        "Not comparable - insufficient verified parameters"
    )

    display_cols = [
        "PSI_ID",
        "Investigation",
        "Hardware",
        "Material",
        "Oxygen_Percent",
        "Pressure_kPa",
        "Airflow_cm_s",
        "Flow_Direction",
        "Burn_Time_s",
        "Illustrative_Project_Score",
        "Comparison_Status"
    ]

    st.dataframe(
        ranked_df[display_cols],
        width="stretch",
        hide_index=True
    )

    st.info(
        "**Scientific limitation:** The illustrative project score is "
        "calculated using EigenMind's own mathematical formula. It is "
        "not an official NASA hazard score, experimental measurement, "
        "or validated prediction of fire behaviour. Scores are not "
        "used to rank different NASA investigations."
    )

    st.subheader("🔬 Side-by-Side Investigation Comparator")


    # ============================================================
    # TRANSPARENT NASA PSI EVIDENCE COVERAGE RANKING
    # This ranks data availability, NOT scientific importance or risk.
    # ============================================================

    st.markdown("---")
    st.subheader("?? NASA PSI Evidence Coverage Ranking")

    st.caption(
        "TRANSPARENT DATA-READINESS INDEX | Ranks catalogue records "
        "by source verification and available numerical evidence."
    )

    phase2_catalogue_path = "data/nasa_investigation_catalogue.csv"

    if os.path.exists(phase2_catalogue_path):
        try:
            phase2_catalogue = pd.read_csv(
                phase2_catalogue_path,
                dtype=str,
                keep_default_na=False
            )

            phase2_core_fields = [
                field for field in [
                    "Oxygen_Percent",
                    "Pressure_kPa",
                    "Airflow_cm_s",
                    "Gravity_g",
                    "Burn_Time_s"
                ]
                if field in base_df.columns
            ]

            phase2_ranking_rows = []

            for _, catalogue_row in phase2_catalogue.iterrows():

                phase2_psi_id = str(
                    catalogue_row["PSI_ID"]
                ).strip()

                phase2_investigation = str(
                    catalogue_row["Investigation"]
                )

                phase2_verification = str(
                    catalogue_row["Verification_Status"]
                )

                phase2_is_verified = (
                    "Verified NASA PSI record"
                    in phase2_verification
                )

                if phase2_psi_id == "Not verified":
                    phase2_matches = base_df.iloc[0:0].copy()
                else:
                    phase2_matches = base_df[
                        base_df["PSI_ID"].astype(str).apply(
                            lambda value: phase2_psi_id in [
                                item.strip()
                                for item in value.split("/")
                            ]
                        )
                    ]

                phase2_record_count = len(phase2_matches)

                if (
                    phase2_record_count > 0
                    and phase2_core_fields
                ):
                    phase2_numeric = phase2_matches[
                        phase2_core_fields
                    ].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    phase2_available_cells = int(
                        phase2_numeric.notna().sum().sum()
                    )

                    phase2_total_cells = (
                        phase2_record_count
                        * len(phase2_core_fields)
                    )

                    phase2_coverage = (
                        phase2_available_cells
                        / phase2_total_cells
                    )
                else:
                    phase2_coverage = 0.0

                phase2_verification_points = (
                    30 if phase2_is_verified else 0
                )

                phase2_record_points = (
                    30 if phase2_record_count > 0 else 0
                )

                phase2_numeric_points = (
                    40 * phase2_coverage
                )

                phase2_total_score = (
                    phase2_verification_points
                    + phase2_record_points
                    + phase2_numeric_points
                )

                if phase2_record_count > 0:
                    phase2_data_status = (
                        "Numerical master-dataset records available"
                    )
                else:
                    phase2_data_status = (
                        "No matching numerical master-dataset record"
                    )

                phase2_ranking_rows.append({
                    "PSI_ID": phase2_psi_id,
                    "Investigation": phase2_investigation,
                    "Evidence_Coverage_Index": round(
                        phase2_total_score, 2
                    ),
                    "Verified_Record": (
                        "Yes" if phase2_is_verified else "Pending"
                    ),
                    "Master_Dataset_Records": phase2_record_count,
                    "Numerical_Field_Coverage_Percent": round(
                        phase2_coverage * 100, 1
                    ),
                    "Data_Status": phase2_data_status
                })

            phase2_ranking_df = pd.DataFrame(
                phase2_ranking_rows
            )

            phase2_ranking_df = phase2_ranking_df.sort_values(
                by=[
                    "Evidence_Coverage_Index",
                    "Investigation"
                ],
                ascending=[False, True]
            ).reset_index(drop=True)

            phase2_ranking_df.insert(
                0,
                "Evidence_Rank",
                range(1, len(phase2_ranking_df) + 1)
            )

            st.dataframe(
                phase2_ranking_df,
                width="stretch",
                hide_index=True
            )

            with st.expander(
                "How is the Evidence Coverage Index calculated?"
            ):
                st.markdown(
                    """
                    - **30 points:** verified NASA PSI identifier.
                    - **30 points:** at least one matching record in the
                      numerical master dataset.
                    - **40 points:** proportion of available numerical
                      parameter cells across matched records.

                    The index measures data readiness only. It does not
                    measure fire danger, scientific importance, experimental
                    quality, or mission safety.
                    """
                )

            st.info(
                "A high evidence coverage score means that more verified "
                "and structured numerical information is available in "
                "EigenMind. It does not mean that an investigation is "
                "more scientifically important or more hazardous."
            )

        except Exception as phase2_ranking_error:
            st.error(
                "Evidence coverage ranking could not be generated: "
                + str(phase2_ranking_error)
            )
    else:
        st.warning(
            "Evidence ranking unavailable: catalogue CSV was not found."
        )


    # Give each record a unique selection label.
    record_labels = {}

    for idx, row in ranked_df.iterrows():
        label = (
            f"{row['PSI_ID']} | "
            f"{row['Investigation']} | "
            f"{row['Flow_Direction']} | "
            f"Burn time: {row['Burn_Time_s']}"
        )
        record_labels[label] = idx

    options = list(record_labels.keys())

    col_a, col_b = st.columns(2)

    def show_investigation(label, column):
        row = ranked_df.iloc[record_labels[label]]

        with column:
            st.markdown(f"**PSI Identifier:** `{row['PSI_ID']}`")
            st.markdown(f"**Investigation:** {row['Investigation']}")
            st.markdown(f"**DOI:** {row['DOI']}")
            st.markdown(f"**Hardware:** {row['Hardware']}")
            st.markdown(
                f"**Fuel / Material:** {row['Material']} "
                f"({row['Fuel_State']})"
            )

            def source_value(value, unit=""):
                if pd.isna(value):
                    return "Not specified in source"
                return f"{value}{unit}"

            st.markdown(
                "**Oxygen:** "
                + source_value(row["Oxygen_Percent"], "%")
            )
            st.markdown(
                "**Pressure:** "
                + source_value(row["Pressure_kPa"], " kPa")
            )
            st.markdown(
                "**Airflow:** "
                + source_value(row["Airflow_cm_s"], " cm/s")
            )
            st.markdown(
                f"**Flow direction:** {row['Flow_Direction']}"
            )
            st.markdown(
                "**Gravity:** "
                + source_value(row["Gravity_g"], " g")
            )

            score = row["Illustrative_Project_Score"]

            if pd.notna(score):
                st.metric(
                    "Illustrative Project Score",
                    f"{score:.2f} / 100"
                )
            else:
                st.caption(
                    "Numerical comparison unavailable: "
                    "insufficient verified source parameters."
                )

            st.markdown("**Scientific Finding**")
            st.write(row["Key_Finding"])

            st.markdown("**Potential Earth Application**")
            st.write(row["Earth_Application"])

    with col_a:
        choice_a = st.selectbox(
            "Select Investigation A",
            options,
            index=0,
            key="psi_compare_a"
        )
        show_investigation(choice_a, col_a)

    with col_b:
        choice_b = st.selectbox(
            "Select Investigation B",
            options,
            index=min(1, len(options) - 1),
            key="psi_compare_b"
        )
        show_investigation(choice_b, col_b)

# ---------------- TAB 3: SI RESEARCH ASSISTANT ----------------
with tab3:
    st.subheader("EigenMind SI Knowledge Engine (Grounded in NASA PSI Records)")

    st.caption(
        "Retrieve source-linked NASA PSI records about microgravity combustion, "
        "smoke, cool flames, fire suppression, and spacecraft fire safety."
    )

    st.info(
        "This tool uses TF-IDF text retrieval and cosine similarity over the "
        "available PSI records. Similarity is a text-matching measure, not "
        "scientific confidence, model accuracy, or an independently generated answer."
    )

    user_query = st.text_input(
        "Enter your research or mission safety question:",
        placeholder="e.g., How does airflow direction affect SAFFIRE-I?"
    )

    text_columns = [
        "PSI_ID",
        "Investigation",
        "Hardware",
        "Material",
        "Key_Finding",
        "Earth_Application"
    ]

    corpus_df = base_df[text_columns].fillna("").astype(str)

    corpus = (
        corpus_df["PSI_ID"] + " " +
        corpus_df["Investigation"] + " " +
        corpus_df["Hardware"] + " " +
        corpus_df["Material"] + " " +
        corpus_df["Key_Finding"] + " " +
        corpus_df["Earth_Application"]
    ).tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    if user_query and user_query.strip():
        query = user_query.strip()
        q_vec = vectorizer.transform([query])

        sims = cosine_similarity(
            q_vec,
            tfidf_matrix
        ).flatten()

        threshold = 0.05
        top_indices = sims.argsort()[::-1][:3]
        valid_indices = [
            int(idx) for idx in top_indices
            if sims[int(idx)] >= threshold
        ]

        if not valid_indices:
            st.warning(
                "No sufficiently similar record was found in the currently "
                "available NASA PSI dataset. Try a question about SAFFIRE, "
                "BASS, smoke and aerosols, cool flames, SLICE, or microgravity combustion."
            )
            st.caption(
                f"Retrieval threshold: {threshold:.2f} cosine similarity. "
                "This is an initial project-defined filter, not a scientifically "
                "validated relevance threshold."
            )

        else:
            st.markdown("### NASA PSI-Grounded Research Brief")
            st.caption(
                "Results are ordered by text similarity to your question. "
                "A higher score does not establish scientific applicability."
            )

            for rank, idx in enumerate(valid_indices, start=1):
                rec = base_df.iloc[idx]
                score_pct = sims[idx] * 100

                with st.expander(
                    f"#{rank} {rec['Investigation']} "
                    f"({rec['PSI_ID']}) — Text similarity: {score_pct:.1f}%",
                    expanded=True
                ):
                    st.write(
                        f"**Summary Insight:** {rec['Key_Finding']}"
                    )

                    oxygen_text = (
                        f"{rec['Oxygen_Percent']}% O2"
                        if pd.notna(rec["Oxygen_Percent"])
                        else "O2: Not specified in source"
                    )

                    pressure_text = (
                        f"{rec['Pressure_kPa']} kPa"
                        if pd.notna(rec["Pressure_kPa"])
                        else "Pressure: Not specified in source"
                    )

                    airflow_text = (
                        f"{rec['Airflow_cm_s']} cm/s"
                        if pd.notna(rec["Airflow_cm_s"])
                        else "Airflow: Not specified in source"
                    )

                    flow_direction = (
                        str(rec["Flow_Direction"])
                        if pd.notna(rec["Flow_Direction"])
                        else "Not specified in source"
                    )

                    st.write(
                        f"**Test Parameters:** {rec['Material']} | "
                        f"{oxygen_text} | {pressure_text} | "
                        f"{airflow_text} | {flow_direction}"
                    )

                    doi_value = str(rec["DOI"]).strip()
 
                    is_valid_doi = (
                        doi_value.startswith("10.")
                        and "/" in doi_value
                        and doi_value.lower() != "nan"
                    )

                    if is_valid_doi:
                        doi_url = f"https://doi.org/{doi_value}"

                        st.markdown(
                            f"**Hardware Platform:** {rec['Hardware']} | "
                            f"**Official Citation:** [{doi_value}]({doi_url})"
                        )

                    elif doi_value and doi_value.lower() != "nan":
                        st.write(
                            f"**Hardware Platform:** {rec['Hardware']} | "
                            f"**Source Identifier:** {doi_value} "
                            "(DOI not supplied)"
                        )

                    else:
                        st.write(
                            f"**Hardware Platform:** {rec['Hardware']} | "
                            "**Official Citation:** Not available"
                        )

                    st.write(
                        f"**Potential Earth Applications:**"
                        f"{rec['Earth _Applications']}"
                    )

    else:
        st.markdown("**Suggested Questions to Test:**")

        st.markdown(
            "- *How does airflow direction affect fire duration in SAFFIRE-I?*"
        )
        st.markdown(
            "- *What happens to smoke detectors and particulates in microgravity (SAME)?*"
        )
        st.markdown(
            "- *How are cool flames and high oxygen tested in ACME CFI-G and FLEX?*"
        )

# ---------------- TAB 4: CITATIONS & EARTH SPIN-OFFS ----------------
with tab4:
    st.subheader(
        "📚 Official NASA Data Citations & Open-Source Attribution"
    )

    st.markdown("""
    In compliance with **NASA Science Mission Directorate Open Source Science guidance**, EigenMind acknowledges the following NASA PSI investigations and research records.

    1. **PSI-26 — Burning and Suppression of Solids (BASS)**

       NASA Physical Sciences Informatics (PSI), Version 4. Microgravity investigation conducted aboard the International Space Station (ISS), using the Microgravity Science Glovebox (MSG). DOI: [10.60555/58h4-dy80](https://doi.org/10.60555/58h4-dy80). [NASA PSI record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-26).

    2. **PSI-62 — Concurrent Flame Spread Modelling**

       Ground-based computational modelling study using BASS and BASS-II experimental data as benchmarks. It is not the BASS flight experiment. DOI: [10.60555/e9xp-ex83](https://doi.org/10.60555/e9xp-ex83). [NASA PSI record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-62).

    3. **PSI-69 — Flame Extinguishment Experiment (FLEX)**

       NASA PSI, Version 5. Microgravity investigation aboard the ISS, studying liquid-fuel droplet combustion, flame extinction and gaseous fire suppressants. DOI: [10.60555/mbq8-0451](https://doi.org/10.60555/mbq8-0451). [NASA PSI record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69).

    4. **PSI-142 — Quantitative Studies of Cool Flame Transitions at Radiation / Stretch Extinction Using Counterflow Flames**

       NASA PSI, Version 2. Combustion Science research record related to PSI-69 (FLEX). This investigation studies cool-flame transitions using counterflow flames; it is not the FLEX flight experiment.

       DOI: [10.60555/80qf-vj95](https://doi.org/10.60555/80qf-vj95). [NASA PSI record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-142).

    5. **PSI-98 — Spacecraft Fire Safety Experiment-I (SAFFIRE-I)**

       NASA PSI, Version 4. Microgravity spacecraft fire-safety investigation conducted aboard the Cygnus spacecraft. DOI: [10.60555/0t15-1z43](https://doi.org/10.60555/0t15-1z43). [NASA PSI record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-98).

    6. **PSI-159 — ACME Cool Flames Investigation (CFI-G)**

       Advanced Combustion via Microgravity Experiments (ACME), Cool Flames Investigation — Gases. DOI: [10.60555/cyt1-kh64](https://doi.org/10.60555/cyt1-kh64). [NASA PSI record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-159).

    7. **PSI-101 and PSI-102 — SAME-R and SAME**

       Smoke and Aerosol Measurement Experiment and its reflight. [PSI-101](https://psi.nasa.gov/physci/repo/data/investigations/PSI-101) | [PSI-102](https://psi.nasa.gov/physci/repo/data/investigations/PSI-102).

    8. **PSI-106 — SLICE**

       Structure and Liftoff in Combustion Experiment. DOI: [10.60555/pm12-rg61](https://doi.org/10.60555/pm12-rg61). [NASA PSI record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-106).

    9. **Research Reference**

       *A Researcher's Guide to International Space Station Combustion Science*, NASA ISS Program Science Office.

    10. **Open-Source Software Stack**

        Python, Streamlit, Pandas, NumPy, Scikit-Learn, Plotly and Joblib. Software and dependencies are acknowledged according to their respective licenses.

    **Scientific attribution note:** Microgravity flight experiments and ground-based modelling investigations are classified separately. A modelling study that uses flight-experiment data is not itself a flight experiment.
    """)

    st.subheader(
        "🌍 Potential Earth Fire-Safety Applications"
    )

    earth_cols = list((
        "PSI_ID",
        "Investigation",
        "Earth_Application"
    ))

    st.dataframe(
        base_df.loc[:, earth_cols],
        width="stretch"
    )


# ---------------- TAB 5: PSI-26 BASS EXPERIMENTAL DATA ----------------
with tab5:
    st.subheader("🔥 PSI-26 — Burning and Suppression of Solids (BASS)")

    st.markdown("""
    
        
    **NASA Physical Sciences Informatics (PSI) — Investigation Details**

    - **Investigation ID:** PSI-26
    - **Investigation:** Burning and Suppression of Solids (BASS)
    - **Project Type:** Microgravity Investigation
    - **Flight Platform:** International Space Station (ISS)
    - **Hardware:** Microgravity Science Glovebox (MSG)
    - **Version:** 4
    - **DOI:** [10.60555/58h4-dy80](https://doi.org/10.60555/58h4-dy80)
    - **Official NASA Source:** [PSI-26 Investigation Record](https://psi.nasa.gov/physci/repo/data/investigations/PSI-26)

    **About this dataset**

    The table below contains PSI-26 experimental test-index records, including test dates, test identifiers and fuel sample materials. It is not raw, time-resolved flame imagery or sensor telemetry.

    The original source values are preserved. Records flagged by the data-quality audit should be checked against the NASA source before scientific interpretation.
    
     
    """)


    psi26_path = "data/psi_26_bass_experimental_table.csv"
    audit_path = "data/psi_26_bass_quality_audit.csv"

    if os.path.exists(psi26_path):
        psi26_df = pd.read_csv(psi26_path)

        st.success(f"PSI-26 experimental table loaded: {len(psi26_df)} records")

        st.markdown("### Experimental test records")
        st.dataframe(psi26_df, width="stretch")

        st.markdown("### Dataset overview")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Total records", len(psi26_df))
        metric2.metric("Columns", len(psi26_df.columns))
        metric3.metric(
            "Fuel/material categories",
            psi26_df["Fuel Sample Material"].nunique()
        )

        st.markdown("### Fuel sample material distribution")
        material_counts = (
            psi26_df["Fuel Sample Material"]
            .value_counts()
            .reset_index()
        )
        material_counts.columns = ["Fuel Sample Material", "Number of records"]

        st.bar_chart(
            material_counts.set_index("Fuel Sample Material")
        )

        st.markdown("### Data quality review")

        if os.path.exists(audit_path):
            audit_df = pd.read_csv(audit_path)
            st.dataframe(audit_df, width="stretch")

            flagged = audit_df[
                audit_df["Data_Quality_Flag"] != "No issue detected"
            ]

            st.warning(
                f"{len(flagged)} records require source verification. "
                "Original source values have been preserved."
            )
        else:
            st.info("Quality audit file was not found.")

        st.download_button(
            "⬇️ Download PSI-26 Experimental Dataset",
            data=psi26_df.to_csv(index=False),
            file_name="psi_26_bass_experimental_table.csv",
            mime="text/csv"
        )

        st.download_button(
            "⬇️ Download PSI-26 Quality Audit",
            data=pd.read_csv(audit_path).to_csv(index=False)
            if os.path.exists(audit_path) else "",
            file_name="psi_26_bass_quality_audit.csv",
            mime="text/csv",
            disabled=not os.path.exists(audit_path)
        )

    else:
        st.error(
            "PSI-26 dataset not found. Check the data folder and filename."
        )

