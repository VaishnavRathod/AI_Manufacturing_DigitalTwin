from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Manufacturing Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

DECISION_FILE = (
    REPORT_DIR /
    "phase12_decision_states.csv"
)

TWIN_FILE = (
    REPORT_DIR /
    "phase11_digital_twin_states.csv"
)

FINAL_RESULTS_FILE = (
    REPORT_DIR /
    "phase8_final_test_results.csv"
)

PER_TOOL_RESULTS_FILE = (
    REPORT_DIR /
    "phase8_per_tool_results.csv"
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    required_files = [
        DECISION_FILE,
        TWIN_FILE
    ]

    for file in required_files:

        if not file.exists():

            st.error(
                f"Required file not found:\n{file}"
            )

            st.stop()

    decisions = pd.read_csv(
        DECISION_FILE
    )

    twin = pd.read_csv(
        TWIN_FILE
    )

    # --------------------------------------------------------
    # Keep useful SHAP contribution columns
    # from Phase 11.
    # --------------------------------------------------------

    driver_columns = [
        column
        for column in twin.columns
        if column.startswith(
            "ConditionDriver"
        )
    ]

    merge_columns = [
        "ToolIndex",
        "NumberOfCycle"
    ] + driver_columns

    twin_subset = (
        twin[
            merge_columns
        ]
        .copy()
    )

    data = decisions.merge(
        twin_subset,
        on=[
            "ToolIndex",
            "NumberOfCycle"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Make IsAnomaly safely boolean
    # --------------------------------------------------------

    if data[
        "IsAnomaly"
    ].dtype != bool:

        data[
            "IsAnomaly"
        ] = (
            data[
                "IsAnomaly"
            ]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                }
            )
            .fillna(False)
        )

    final_results = None

    if FINAL_RESULTS_FILE.exists():

        final_results = pd.read_csv(
            FINAL_RESULTS_FILE
        )

    per_tool_results = None

    if PER_TOOL_RESULTS_FILE.exists():

        per_tool_results = pd.read_csv(
            PER_TOOL_RESULTS_FILE
        )

    return (
        data,
        final_results,
        per_tool_results
    )


(
    data,
    final_results,
    per_tool_results

) = load_data()


# ============================================================
# DISPLAY UTILITIES
# ============================================================

def state_icon(state):

    mapping = {
        "NORMAL": "🟢",
        "WATCH": "🟡",
        "HIGH_RISK": "🟠",
        "CRITICAL": "🔴"
    }

    return mapping.get(
        state,
        "⚪"
    )


def health_icon(band):

    mapping = {
        "HEALTHY": "🟢",
        "EARLY_WEAR": "🟡",
        "DEGRADED": "🟠",
        "CRITICAL": "🔴"
    }

    return mapping.get(
        band,
        "⚪"
    )


def health_percentage(value):

    return (
        max(
            0.0,
            min(
                1.0,
                float(value)
            )
        )
        *
        100
    )


def show_health_progress(
    label,
    value
):

    percentage = (
        health_percentage(
            value
        )
    )

    st.write(
        f"**{label}: {percentage:.1f}%**"
    )

    st.progress(
        float(
            percentage / 100
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏭 Digital Twin"
)

page = st.sidebar.radio(

    "Navigation",

    [
        "Twin Explorer",
        "System Overview",
        "Model Validation"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "AI-Driven CNC Manufacturing Digital Twin"
)

st.sidebar.caption(
    "Predictive tool health • anomaly detection • "
    "explainable AI • decision support"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "AI-Driven CNC Manufacturing Digital Twin"
)

st.caption(
    "Predictive Tool Health, Condition Monitoring, "
    "Anomaly Detection and Explainable Decision Support"
)


# ============================================================
# PAGE 1 — TWIN EXPLORER
# ============================================================

if page == "Twin Explorer":

    st.subheader(
        "Digital Twin Explorer"
    )

    st.write(
        "Explore the estimated state of a cutting tool "
        "at a specific machining cycle."
    )

    # --------------------------------------------------------
    # Tool selection
    # --------------------------------------------------------

    available_tools = sorted(
        data[
            "ToolIndex"
        ]
        .unique()
        .tolist()
    )

    selected_tool = st.sidebar.selectbox(
        "Tool",
        available_tools
    )

    tool_data = (
        data[
            data[
                "ToolIndex"
            ]
            ==
            selected_tool
        ]
        .sort_values(
            "NumberOfCycle"
        )
        .reset_index(
            drop=True
        )
    )

    available_cycles = (
        tool_data[
            "NumberOfCycle"
        ]
        .tolist()
    )

    selected_cycle = st.sidebar.select_slider(

        "Machining Cycle",

        options=available_cycles,

        value=available_cycles[-1]
    )

    selected = (
        tool_data[
            tool_data[
                "NumberOfCycle"
            ]
            ==
            selected_cycle
        ]
        .iloc[0]
    )

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Machine",
            selected[
                "MachineID"
            ]
        )

    with c2:

        st.metric(
            "Tool",
            int(
                selected_tool
            )
        )

    with c3:

        st.metric(
            "Cycle",
            int(
                selected_cycle
            )
        )

    with c4:

        decision = (
            selected[
                "DecisionState"
            ]
        )

        st.metric(
            "Twin State",
            f"{state_icon(decision)} {decision}"
        )

    st.divider()

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    st.subheader(
        "Health Assessment"
    )

    h1, h2, h3 = st.columns(3)

    with h1:

        lifecycle = float(
            selected[
                "LifecycleHealth"
            ]
        )

        st.metric(
            "Lifecycle Health",
            f"{lifecycle * 100:.1f}%"
        )

        show_health_progress(
            selected[
                "LifecycleBand"
            ],
            lifecycle
        )

    with h2:

        condition = float(
            selected[
                "ConditionHealth"
            ]
        )

        st.metric(
            "Condition Health",
            f"{condition * 100:.1f}%"
        )

        show_health_progress(
            selected[
                "ConditionBand"
            ],
            condition
        )

    with h3:

        gap = float(
            selected[
                "HealthGap"
            ]
        )

        st.metric(
            "Health Gap",
            f"{gap * 100:+.1f} %"
        )

        st.write(
            "**Model agreement**"
        )

        st.write(
            selected[
                "HealthGapState"
            ]
        )

    # --------------------------------------------------------
    # Anomaly state
    # --------------------------------------------------------

    st.subheader(
        "Condition Monitoring"
    )

    a1, a2, a3 = st.columns(3)

    with a1:

        anomaly_text = (
            "ABNORMAL"
            if selected[
                "IsAnomaly"
            ]
            else
            "NORMAL"
        )

        anomaly_icon = (
            "⚠️"
            if selected[
                "IsAnomaly"
            ]
            else
            "✅"
        )

        st.metric(
            "Electrical Behaviour",
            f"{anomaly_icon} {anomaly_text}"
        )

    with a2:

        st.metric(
            "Anomaly Score",
            f"{selected['AnomalyScore']:.4f}"
        )

    with a3:

        st.metric(
            "Evidence Signals",
            int(
                selected[
                    "EvidenceCount"
                ]
            )
        )

    # --------------------------------------------------------
    # Decision support
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "AI Decision Support"
    )

    state = (
        selected[
            "DecisionState"
        ]
    )

    st.markdown(
        f"### {state_icon(state)} {state}"
    )

    st.write(
        f"**Decision confidence:** "
        f"{selected['DecisionConfidence']}"
    )

    st.write(
        "**Evidence:**"
    )

    evidence_text = str(
        selected[
            "Evidence"
        ]
    )

    if evidence_text.strip():

        for item in (
            evidence_text.split(
                " | "
            )
        ):

            st.write(
                f"- {item}"
            )

    else:

        st.write(
            "- No significant warning evidence."
        )

    # --------------------------------------------------------
    # Root cause
    # --------------------------------------------------------

    st.subheader(
        "Explainable AI — Degradation Drivers"
    )

    root_causes = str(
        selected[
            "RootCauseDrivers"
        ]
    )

    for i, driver in enumerate(
        root_causes.split(
            " | "
        ),
        start=1
    ):

        if driver.strip():

            shap_column = (
                f"ConditionDriver{i}_SHAP"
            )

            if (
                shap_column
                in
                selected.index
                and
                pd.notna(
                    selected[
                        shap_column
                    ]
                )
            ):

                st.write(
                    f"**{i}. {driver}**  "
                    f"— SHAP "
                    f"{selected[shap_column]:+.4f}"
                )

            else:

                st.write(
                    f"**{i}. {driver}**"
                )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.subheader(
        "Recommended Engineering Action"
    )

    if state == "CRITICAL":

        st.error(
            selected[
                "Recommendation"
            ]
        )

    elif state == "HIGH_RISK":

        st.warning(
            selected[
                "Recommendation"
            ]
        )

    elif state == "WATCH":

        st.info(
            selected[
                "Recommendation"
            ]
        )

    else:

        st.success(
            selected[
                "Recommendation"
            ]
        )

    # ========================================================
    # TOOL TRAJECTORY
    # ========================================================

    st.divider()

    st.subheader(
        "Digital Twin Health Trajectory"
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(

            x=tool_data[
                "NumberOfCycle"
            ],

            y=(
                tool_data[
                    "LifecycleHealth"
                ]
                *
                100
            ),

            mode="lines+markers",

            name="Lifecycle Health"
        )
    )

    figure.add_trace(
        go.Scatter(

            x=tool_data[
                "NumberOfCycle"
            ],

            y=(
                tool_data[
                    "ConditionHealth"
                ]
                *
                100
            ),

            mode="lines+markers",

            name="Condition Health"
        )
    )

    figure.add_vline(
        x=selected_cycle,
        line_dash="dash"
    )

    figure.update_layout(

        xaxis_title=
            "Machining Cycle",

        yaxis_title=
            "Predicted Health (%)",

        yaxis=dict(
            range=[
                0,
                100
            ]
        ),

        hovermode=
            "x unified",

        height=500
    )

    st.plotly_chart(
        figure,
        use_container_width=True
    )

    # ========================================================
    # ANOMALY TRAJECTORY
    # ========================================================

    st.subheader(
        "Anomaly Score Across Tool Life"
    )

    anomaly_figure = go.Figure()

    anomaly_figure.add_trace(
        go.Scatter(

            x=tool_data[
                "NumberOfCycle"
            ],

            y=tool_data[
                "AnomalyScore"
            ],

            mode=
                "lines+markers",

            name=
                "Anomaly Score"
        )
    )

    anomaly_figure.add_hline(
        y=0,
        line_dash="dash"
    )

    anomaly_figure.add_vline(
        x=selected_cycle,
        line_dash="dash"
    )

    anomaly_figure.update_layout(

        xaxis_title=
            "Machining Cycle",

        yaxis_title=
            "Anomaly Score",

        hovermode=
            "x unified",

        height=400
    )

    st.plotly_chart(
        anomaly_figure,
        use_container_width=True
    )

    st.caption(
        "Higher anomaly scores indicate greater deviation "
        "from the healthier electrical-current reference population."
    )


# ============================================================
# PAGE 2 — SYSTEM OVERVIEW
# ============================================================

elif page == "System Overview":

    st.subheader(
        "Digital Twin System Overview"
    )

    total_samples = len(
        data
    )

    anomalies = int(
        data[
            "IsAnomaly"
        ].sum()
    )

    critical = int(
        (
            data[
                "DecisionState"
            ]
            ==
            "CRITICAL"
        )
        .sum()
    )

    high_risk = int(
        (
            data[
                "DecisionState"
            ]
            ==
            "HIGH_RISK"
        )
        .sum()
    )

    c1, c2, c3, c4 = st.columns(
        4
    )

    c1.metric(
        "Twin States",
        total_samples
    )

    c2.metric(
        "Anomalies",
        anomalies
    )

    c3.metric(
        "High Risk",
        high_risk
    )

    c4.metric(
        "Critical",
        critical
    )

    st.divider()

    # --------------------------------------------------------
    # Decision distribution
    # --------------------------------------------------------

    left, right = st.columns(
        2
    )

    with left:

        st.subheader(
            "Decision-State Distribution"
        )

        state_counts = (
            data[
                "DecisionState"
            ]
            .value_counts()
            .reset_index()
        )

        state_counts.columns = [
            "DecisionState",
            "Count"
        ]

        fig = go.Figure(
            go.Bar(

                x=state_counts[
                    "DecisionState"
                ],

                y=state_counts[
                    "Count"
                ]
            )
        )

        fig.update_layout(
            xaxis_title=
                "Decision State",

            yaxis_title=
                "Samples",

            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Health distributions
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Predicted Health Distribution"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(

                x=(
                    data[
                        "LifecycleHealth"
                    ]
                    *
                    100
                ),

                name=
                    "Lifecycle Health",

                opacity=0.65
            )
        )

        fig.add_trace(
            go.Histogram(

                x=(
                    data[
                        "ConditionHealth"
                    ]
                    *
                    100
                ),

                name=
                    "Condition Health",

                opacity=0.65
            )
        )

        fig.update_layout(

            barmode="overlay",

            xaxis_title=
                "Predicted Health (%)",

            yaxis_title=
                "Samples",

            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Tool-level summary
    # --------------------------------------------------------

    st.subheader(
        "Tool-Level Digital Twin Summary"
    )

    tool_summary = (
        data
        .groupby(
            "ToolIndex"
        )
        .agg(

            Samples=(
                "ToolIndex",
                "size"
            ),

            MeanLifecycleHealth=(
                "LifecycleHealth",
                "mean"
            ),

            MeanConditionHealth=(
                "ConditionHealth",
                "mean"
            ),

            AnomalyRate=(
                "IsAnomaly",
                "mean"
            ),

            CriticalStates=(
                "DecisionState",
                lambda x:
                    (
                        x
                        ==
                        "CRITICAL"
                    )
                    .sum()
            ),

            HighRiskStates=(
                "DecisionState",
                lambda x:
                    (
                        x
                        ==
                        "HIGH_RISK"
                    )
                    .sum()
            )
        )
        .reset_index()
    )

    tool_summary[
        "MeanLifecycleHealth"
    ] = (
        tool_summary[
            "MeanLifecycleHealth"
        ]
        *
        100
    )

    tool_summary[
        "MeanConditionHealth"
    ] = (
        tool_summary[
            "MeanConditionHealth"
        ]
        *
        100
    )

    tool_summary[
        "AnomalyRate"
    ] = (
        tool_summary[
            "AnomalyRate"
        ]
        *
        100
    )

    tool_summary = (
        tool_summary.rename(
            columns={
                "MeanLifecycleHealth":
                    "Mean Lifecycle Health (%)",

                "MeanConditionHealth":
                    "Mean Condition Health (%)",

                "AnomalyRate":
                    "Anomaly Rate (%)"
            }
        )
    )

    st.dataframe(
        tool_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This prototype represents one CNC machine with "
        "multiple cutting-tool trajectories from the experimental dataset."
    )


# ============================================================
# PAGE 3 — MODEL VALIDATION
# ============================================================

elif page == "Model Validation":

    st.subheader(
        "Experimental Model Validation"
    )

    st.warning(
        "ActualHealth is shown only for retrospective "
        "experimental validation. In a deployed Digital Twin, "
        "ground-truth remaining tool health would not normally "
        "be continuously available."
    )

    # --------------------------------------------------------
    # Final model results
    # --------------------------------------------------------

    if final_results is not None:

        st.subheader(
            "Locked-Test Performance"
        )

        display = (
            final_results[
                [
                    "experiment",
                    "number_features",
                    "CV_MAE",
                    "Test_MAE",
                    "CV_RMSE",
                    "Test_RMSE",
                    "CV_R2",
                    "Test_R2"
                ]
            ]
            .copy()
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # Predicted vs actual trajectory
    # --------------------------------------------------------

    st.subheader(
        "Predicted vs Actual Tool Health"
    )

    validation_tools = sorted(
        data[
            "ToolIndex"
        ]
        .unique()
        .tolist()
    )

    validation_tool = st.selectbox(
        "Validation Tool",
        validation_tools
    )

    validation_data = (
        data[
            data[
                "ToolIndex"
            ]
            ==
            validation_tool
        ]
        .sort_values(
            "NumberOfCycle"
        )
    )

    validation_fig = go.Figure()

    validation_fig.add_trace(
        go.Scatter(

            x=validation_data[
                "NumberOfCycle"
            ],

            y=(
                validation_data[
                    "ActualHealth"
                ]
                *
                100
            ),

            mode="lines+markers",

            name="Actual Health"
        )
    )

    validation_fig.add_trace(
        go.Scatter(

            x=validation_data[
                "NumberOfCycle"
            ],

            y=(
                validation_data[
                    "LifecycleHealth"
                ]
                *
                100
            ),

            mode="lines+markers",

            name="Lifecycle Model"
        )
    )

    validation_fig.add_trace(
        go.Scatter(

            x=validation_data[
                "NumberOfCycle"
            ],

            y=(
                validation_data[
                    "ConditionHealth"
                ]
                *
                100
            ),

            mode="lines+markers",

            name="Condition Model"
        )
    )

    validation_fig.update_layout(

        xaxis_title=
            "Machining Cycle",

        yaxis_title=
            "Health (%)",

        yaxis=dict(
            range=[
                0,
                100
            ]
        ),

        hovermode=
            "x unified",

        height=500
    )

    st.plotly_chart(
        validation_fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Per-tool result
    # --------------------------------------------------------

    if per_tool_results is not None:

        st.subheader(
            "Per-Tool Test Performance"
        )

        st.dataframe(
            per_tool_results,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Tool 11 contains only one sample, so its "
            "tool-level R² is undefined and should not "
            "be interpreted as a trajectory-level result."
        )

    # --------------------------------------------------------
    # Retrospective decision validation
    # --------------------------------------------------------

    st.subheader(
        "Decision Support vs Actual Health"
    )

    validation_table = (
        pd.crosstab(

            data[
                "ActualHealth"
            ]
            .apply(
                lambda x:
                    (
                        "HEALTHY"
                        if x >= 0.75
                        else
                        "EARLY_WEAR"
                        if x >= 0.50
                        else
                        "DEGRADED"
                        if x >= 0.25
                        else
                        "CRITICAL"
                    )
            ),

            data[
                "DecisionState"
            ]
        )
    )

    st.dataframe(
        validation_table,
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research / portfolio prototype — decision thresholds "
    "are transparent experimental rules and are not "
    "certified manufacturing operating limits."
)