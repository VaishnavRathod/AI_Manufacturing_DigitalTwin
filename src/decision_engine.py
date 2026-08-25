from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = PROJECT_ROOT / "reports"

TWIN_STATE_FILE = (
    REPORT_DIR /
    "phase11_digital_twin_states.csv"
)

OUTPUT_FILE = (
    REPORT_DIR /
    "phase12_decision_states.csv"
)

SUMMARY_FILE = (
    REPORT_DIR /
    "phase12_decision_summary.txt"
)

VALIDATION_FILE = (
    REPORT_DIR /
    "phase12_decision_validation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

HEALTHY = 0.75
EARLY_WEAR = 0.50
CRITICAL_HEALTH = 0.25

HEALTH_GAP_THRESHOLD = 0.15


# ============================================================
# ACTUAL HEALTH BAND
#
# Used ONLY for retrospective validation.
# It must NEVER be used by the decision logic.
# ============================================================

def actual_health_band(value):

    if value >= 0.75:
        return "HEALTHY"

    if value >= 0.50:
        return "EARLY_WEAR"

    if value >= 0.25:
        return "DEGRADED"

    return "CRITICAL"


# ============================================================
# COLLECT EVIDENCE
# ============================================================

def collect_evidence(row):

    lifecycle = float(
        row["LifecycleHealth"]
    )

    condition = float(
        row["ConditionHealth"]
    )

    anomaly = bool(
        row["IsAnomaly"]
    )

    health_gap = float(
        row["HealthGap"]
    )

    gap_state = row[
        "HealthGapState"
    ]

    evidence = []

    # --------------------------------------------------------
    # Lifecycle evidence
    # --------------------------------------------------------

    if lifecycle < CRITICAL_HEALTH:

        evidence.append(
            "Lifecycle model indicates critical remaining health"
        )

    elif lifecycle < EARLY_WEAR:

        evidence.append(
            "Lifecycle model indicates significant degradation"
        )

    elif lifecycle < HEALTHY:

        evidence.append(
            "Lifecycle model indicates early wear"
        )

    # --------------------------------------------------------
    # Physical condition evidence
    # --------------------------------------------------------

    if condition < CRITICAL_HEALTH:

        evidence.append(
            "Current-based condition model indicates critical degradation"
        )

    elif condition < EARLY_WEAR:

        evidence.append(
            "Current-based condition model indicates significant degradation"
        )

    elif condition < HEALTHY:

        evidence.append(
            "Current-based condition model indicates early wear"
        )

    # --------------------------------------------------------
    # Anomaly evidence
    # --------------------------------------------------------

    if anomaly:

        evidence.append(
            "Electrical-current behaviour is anomalous"
        )

    # --------------------------------------------------------
    # Model disagreement
    # --------------------------------------------------------

    if (
        gap_state
        ==
        "CONDITION_WORSE_THAN_LIFECYCLE"
    ):

        evidence.append(
            "Observed condition is worse than lifecycle expectation"
        )

    elif (
        gap_state
        ==
        "CONDITION_BETTER_THAN_LIFECYCLE"
    ):

        evidence.append(
            "Observed condition is healthier than lifecycle expectation"
        )

    return evidence


# ============================================================
# DETERMINE DECISION STATE
# ============================================================

def determine_state(row):

    lifecycle = float(
        row["LifecycleHealth"]
    )

    condition = float(
        row["ConditionHealth"]
    )

    anomaly = bool(
        row["IsAnomaly"]
    )

    gap_state = (
        row["HealthGapState"]
    )

    # ========================================================
    # CRITICAL
    # ========================================================

    # Either model individually indicates
    # extremely low health.

    if (
        lifecycle < CRITICAL_HEALTH
        or
        condition < CRITICAL_HEALTH
    ):

        return "CRITICAL"

    # Multiple independent degradation signals:
    #
    # both supervised models indicate
    # significant degradation
    #
    # AND
    #
    # unsupervised current behaviour
    # is also abnormal.

    if (
        lifecycle < EARLY_WEAR
        and
        condition < EARLY_WEAR
        and
        anomaly
    ):

        return "CRITICAL"

    # ========================================================
    # HIGH RISK
    # ========================================================

    # Significant degradation according
    # to either supervised model.

    if (
        lifecycle < EARLY_WEAR
        or
        condition < EARLY_WEAR
    ):

        return "HIGH_RISK"

    # Unexpected condition degradation
    # compared with lifecycle estimate.

    if (
        gap_state
        ==
        "CONDITION_WORSE_THAN_LIFECYCLE"
    ):

        return "HIGH_RISK"

    # Anomalous behaviour when neither model
    # has yet entered a degraded state.

    if anomaly:

        return "WATCH"

    # ========================================================
    # WATCH
    # ========================================================

    if (
        lifecycle < HEALTHY
        or
        condition < HEALTHY
    ):

        return "WATCH"

    # ========================================================
    # NORMAL
    # ========================================================

    return "NORMAL"


# ============================================================
# RECOMMENDATION
# ============================================================

def generate_recommendation(
    decision_state,
    row
):

    if decision_state == "NORMAL":

        return (
            "Continue normal operation and standard "
            "condition monitoring."
        )

    if decision_state == "WATCH":

        if bool(row["IsAnomaly"]):

            return (
                "Increase condition monitoring and review "
                "electrical-current trends at the next "
                "scheduled inspection."
            )

        return (
            "Continue operation with enhanced monitoring "
            "for progression of tool degradation."
        )

    if decision_state == "HIGH_RISK":

        if (
            row["HealthGapState"]
            ==
            "CONDITION_WORSE_THAN_LIFECYCLE"
        ):

            return (
                "Inspect the cutting tool and review machining "
                "conditions because measured condition appears "
                "worse than expected lifecycle health."
            )

        return (
            "Schedule tool inspection before continued "
            "high-load production and prepare a replacement "
            "tool if degradation persists."
        )

    if decision_state == "CRITICAL":

        return (
            "Hold the tool for inspection before further "
            "machining and prepare tool replacement. "
            "Review machine-current behaviour and the "
            "identified degradation drivers."
        )

    return "Review Digital Twin state."


# ============================================================
# GET ROOT CAUSES
# ============================================================

def extract_condition_drivers(
    row,
    maximum=3
):

    drivers = []

    for i in range(
        1,
        maximum + 1
    ):

        feature_column = (
            f"ConditionDriver{i}"
        )

        shap_column = (
            f"ConditionDriver{i}_SHAP"
        )

        if feature_column not in row.index:

            continue

        feature = row[
            feature_column
        ]

        if pd.isna(feature):

            continue

        shap_value = None

        if shap_column in row.index:

            shap_value = (
                row[
                    shap_column
                ]
            )

        drivers.append(
            {
                "feature":
                    feature,

                "shap":
                    shap_value
            }
        )

    return drivers


# ============================================================
# HUMAN-READABLE ROOT CAUSE
# ============================================================

def root_cause_text(row):

    drivers = extract_condition_drivers(
        row,
        maximum=3
    )

    if not drivers:

        return (
            "No dominant condition-health "
            "driver identified."
        )

    names = [
        driver["feature"]
        for driver in drivers
    ]

    return " | ".join(names)


# ============================================================
# EVIDENCE COUNT
# ============================================================

def calculate_evidence_count(
    row
):

    count = 0

    lifecycle = float(
        row["LifecycleHealth"]
    )

    condition = float(
        row["ConditionHealth"]
    )

    if lifecycle < EARLY_WEAR:
        count += 1

    if condition < EARLY_WEAR:
        count += 1

    if bool(row["IsAnomaly"]):
        count += 1

    if (
        row["HealthGapState"]
        ==
        "CONDITION_WORSE_THAN_LIFECYCLE"
    ):
        count += 1

    return count


# ============================================================
# DECISION CONFIDENCE
#
# This is NOT model probability.
#
# It is simply a description of how many
# independent warning signals agree.
# ============================================================

def decision_confidence(
    state,
    evidence_count
):

    if state == "NORMAL":

        return "HIGH"

    if evidence_count >= 3:

        return "HIGH"

    if evidence_count == 2:

        return "MEDIUM"

    return "LOW"


# ============================================================
# BUILD DECISIONS
# ============================================================

def build_decisions(states):

    output_rows = []

    for _, row in states.iterrows():

        state = determine_state(
            row
        )

        evidence = collect_evidence(
            row
        )

        evidence_count = (
            calculate_evidence_count(
                row
            )
        )

        confidence = (
            decision_confidence(
                state,
                evidence_count
            )
        )

        recommendation = (
            generate_recommendation(
                state,
                row
            )
        )

        root_cause = (
            root_cause_text(
                row
            )
        )

        output_rows.append({

            # --------------------------------------------
            # Digital Twin identity
            # --------------------------------------------

            "MachineID":
                row["MachineID"],

            "ToolIndex":
                int(row["ToolIndex"]),

            "NumberOfCycle":
                int(row["NumberOfCycle"]),

            # --------------------------------------------
            # Model states
            # --------------------------------------------

            "LifecycleHealth":
                float(
                    row["LifecycleHealth"]
                ),

            "LifecycleBand":
                row["LifecycleBand"],

            "ConditionHealth":
                float(
                    row["ConditionHealth"]
                ),

            "ConditionBand":
                row["ConditionBand"],

            "HealthGap":
                float(
                    row["HealthGap"]
                ),

            "HealthGapState":
                row["HealthGapState"],

            "AnomalyScore":
                float(
                    row["AnomalyScore"]
                ),

            "IsAnomaly":
                bool(
                    row["IsAnomaly"]
                ),

            # --------------------------------------------
            # Final decision
            # --------------------------------------------

            "DecisionState":
                state,

            "EvidenceCount":
                evidence_count,

            "DecisionConfidence":
                confidence,

            "Evidence":
                " | ".join(
                    evidence
                ),

            "RootCauseDrivers":
                root_cause,

            "Recommendation":
                recommendation,

            # --------------------------------------------
            # Ground truth
            #
            # Validation only.
            # Never part of decision logic.
            # --------------------------------------------

            "ActualHealth":
                float(
                    row["ActualHealth"]
                )
        })

    return pd.DataFrame(
        output_rows
    )


# ============================================================
# DECISION SUMMARY
# ============================================================

def print_summary(decisions):

    print("=" * 78)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 12 — DECISION SUPPORT ENGINE")
    print("=" * 78)

    print(
        "\nDecision-state distribution:"
    )

    print(
        decisions[
            "DecisionState"
        ]
        .value_counts()
    )

    print(
        "\nDecision confidence:"
    )

    print(
        decisions[
            "DecisionConfidence"
        ]
        .value_counts()
    )

    print(
        "\nMean health by decision state:"
    )

    health_summary = (
        decisions
        .groupby(
            "DecisionState"
        )
        .agg(

            Samples=(
                "DecisionState",
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
            )
        )
        .reset_index()
    )

    print(
        "\n"
        +
        health_summary
        .to_string(
            index=False
        )
    )


# ============================================================
# RETROSPECTIVE VALIDATION
#
# Again: ActualHealth is used ONLY here,
# after decisions have already been generated.
# ============================================================

def validate_against_actual(
    decisions
):

    validation = (
        decisions.copy()
    )

    validation[
        "ActualHealthBand"
    ] = (
        validation[
            "ActualHealth"
        ]
        .apply(
            actual_health_band
        )
    )

    crosstab = pd.crosstab(

        validation[
            "ActualHealthBand"
        ],

        validation[
            "DecisionState"
        ]
    )

    print("\n" + "=" * 78)
    print("RETROSPECTIVE VALIDATION")
    print("=" * 78)

    print(
        "\nActual health band vs "
        "Digital Twin decision:\n"
    )

    print(crosstab)

    crosstab.to_csv(
        VALIDATION_FILE
    )

    print(
        f"\n✓ Saved:\n"
        f"{VALIDATION_FILE}"
    )


# ============================================================
# REPRESENTATIVE DECISION
# ============================================================

def show_representative_decision(
    decisions
):

    # Prefer the most severe decision
    # with anomaly evidence and a trajectory tool.

    candidates = decisions[
        (
            decisions[
                "DecisionState"
            ]
            ==
            "CRITICAL"
        )
        &
        (
            decisions[
                "IsAnomaly"
            ]
            ==
            True
        )
        &
        (
            decisions[
                "ToolIndex"
            ]
            !=
            11
        )
    ]

    if candidates.empty:

        candidates = decisions[
            decisions[
                "DecisionState"
            ]
            ==
            "HIGH_RISK"
        ]

    if candidates.empty:

        candidates = (
            decisions.copy()
        )

    row = (
        candidates
        .sort_values(
            [
                "ActualHealth",
                "LifecycleHealth"
            ]
        )
        .iloc[0]
    )

    print("\n" + "=" * 78)
    print("REPRESENTATIVE DECISION")
    print("=" * 78)

    print(
        f"\nMachine          : "
        f"{row['MachineID']}"
    )

    print(
        f"Tool             : "
        f"{int(row['ToolIndex'])}"
    )

    print(
        f"Cycle            : "
        f"{int(row['NumberOfCycle'])}"
    )

    print(
        "\nDIGITAL TWIN HEALTH"
    )

    print(
        f"Lifecycle health : "
        f"{row['LifecycleHealth'] * 100:.1f}%"
    )

    print(
        f"Condition health : "
        f"{row['ConditionHealth'] * 100:.1f}%"
    )

    print(
        f"Anomaly          : "
        f"{row['IsAnomaly']}"
    )

    print(
        f"Anomaly score    : "
        f"{row['AnomalyScore']:.4f}"
    )

    print(
        "\nDECISION"
    )

    print(
        f"State            : "
        f"{row['DecisionState']}"
    )

    print(
        f"Evidence count   : "
        f"{row['EvidenceCount']}"
    )

    print(
        f"Confidence       : "
        f"{row['DecisionConfidence']}"
    )

    print(
        "\nEVIDENCE"
    )

    print(
        row["Evidence"]
    )

    print(
        "\nROOT-CAUSE DRIVERS"
    )

    print(
        row[
            "RootCauseDrivers"
        ]
    )

    print(
        "\nRECOMMENDED ACTION"
    )

    print(
        row["Recommendation"]
    )

    print(
        "\nExperimental ground truth:"
    )

    print(
        f"Actual health = "
        f"{row['ActualHealth'] * 100:.1f}%"
    )


# ============================================================
# SAVE
# ============================================================

def save_results(
    decisions
):

    decisions.to_csv(
        OUTPUT_FILE,
        index=False
    )

    with open(
        SUMMARY_FILE,
        "w"
    ) as file:

        file.write(
            "AI MANUFACTURING DIGITAL TWIN\n"
        )

        file.write(
            "PHASE 12 — DECISION SUPPORT ENGINE\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            "Decision states:\n"
        )

        counts = (
            decisions[
                "DecisionState"
            ]
            .value_counts()
        )

        for state, count in (
            counts.items()
        ):

            file.write(
                f"{state}: {count}\n"
            )

        file.write("\n")

        file.write(
            "IMPORTANT:\n"
        )

        file.write(
            "Decision thresholds are transparent "
            "prototype engineering rules and are "
            "not certified manufacturing limits.\n\n"
        )

        file.write(
            "ActualHealth is retained only for "
            "retrospective experimental validation "
            "and is never used as an input to "
            "the decision engine.\n"
        )

    print(
        "\nSaved:"
    )

    print(
        f"✓ {OUTPUT_FILE}"
    )

    print(
        f"✓ {SUMMARY_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not TWIN_STATE_FILE.exists():

        raise FileNotFoundError(
            f"Digital Twin states not found:\n"
            f"{TWIN_STATE_FILE}"
        )

    states = pd.read_csv(
        TWIN_STATE_FILE
    )

    decisions = build_decisions(
        states
    )

    print_summary(
        decisions
    )

    validate_against_actual(
        decisions
    )

    show_representative_decision(
        decisions
    )

    save_results(
        decisions
    )

    print("\n" + "=" * 78)
    print("PHASE 12 COMPLETE")
    print("=" * 78)

    print(
        "\nNext:"
    )

    print(
        "Streamlit manufacturing "
        "Digital Twin dashboard."
    )


if __name__ == "__main__":
    main()