from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TEST_FILE = (
    PROCESSED_DIR /
    "test.csv"
)

AGE_MODEL_FILE = (
    MODEL_DIR /
    "D_age_aware_xgboost.joblib"
)

CONDITION_MODEL_FILE = (
    MODEL_DIR /
    "B_current_based_xgboost.joblib"
)

ANOMALY_MODEL_FILE = (
    MODEL_DIR /
    "isolation_forest_current.joblib"
)

OUTPUT_FILE = (
    REPORT_DIR /
    "phase11_digital_twin_states.csv"
)

SUMMARY_FILE = (
    REPORT_DIR /
    "phase11_digital_twin_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "CycleToFailureNormalized"

HEALTHY_THRESHOLD = 0.75
EARLY_WEAR_THRESHOLD = 0.50
DEGRADED_THRESHOLD = 0.25

# Prototype discrepancy threshold.
# This is our decision-support definition,
# not a label supplied by the dataset.
HEALTH_GAP_THRESHOLD = 0.15


# ============================================================
# HEALTH BAND
# ============================================================

def health_band(health):

    if health >= HEALTHY_THRESHOLD:
        return "HEALTHY"

    if health >= EARLY_WEAR_THRESHOLD:
        return "EARLY_WEAR"

    if health >= DEGRADED_THRESHOLD:
        return "DEGRADED"

    return "CRITICAL"


# ============================================================
# HEALTH GAP INTERPRETATION
# ============================================================

def health_gap_state(
    lifecycle_health,
    condition_health
):

    gap = (
        condition_health
        -
        lifecycle_health
    )

    if gap <= -HEALTH_GAP_THRESHOLD:

        return (
            "CONDITION_WORSE_THAN_LIFECYCLE"
        )

    if gap >= HEALTH_GAP_THRESHOLD:

        return (
            "CONDITION_BETTER_THAN_LIFECYCLE"
        )

    return "CONSISTENT"


# ============================================================
# DIGITAL TWIN ENGINE
# ============================================================

class DigitalTwinEngine:

    def __init__(self):

        # ----------------------------------------------------
        # Load lifecycle model
        # ----------------------------------------------------

        age_payload = joblib.load(
            AGE_MODEL_FILE
        )

        self.age_model = (
            age_payload["model"]
        )

        self.age_features = (
            age_payload["features"]
        )

        # ----------------------------------------------------
        # Load condition model
        # ----------------------------------------------------

        condition_payload = joblib.load(
            CONDITION_MODEL_FILE
        )

        self.condition_model = (
            condition_payload["model"]
        )

        self.condition_features = (
            condition_payload["features"]
        )

        # ----------------------------------------------------
        # Load anomaly model
        # ----------------------------------------------------

        anomaly_payload = joblib.load(
            ANOMALY_MODEL_FILE
        )

        self.anomaly_model = (
            anomaly_payload["model"]
        )

        self.anomaly_features = (
            anomaly_payload["features"]
        )

        # ----------------------------------------------------
        # SHAP explainers
        # ----------------------------------------------------

        self.age_explainer = (
            shap.TreeExplainer(
                self.age_model
            )
        )

        self.condition_explainer = (
            shap.TreeExplainer(
                self.condition_model
            )
        )


    # ========================================================
    # SAFE DISPLAY HEALTH
    # ========================================================

    @staticmethod
    def display_health(
        raw_prediction
    ):

        # Metrics in Phase 8 used raw predictions.
        #
        # For Digital Twin visualization only,
        # normalized health is constrained to
        # the meaningful [0, 1] interval.

        return float(
            np.clip(
                raw_prediction,
                0.0,
                1.0
            )
        )


    # ========================================================
    # GET NEGATIVE SHAP DRIVERS
    # ========================================================

    @staticmethod
    def degradation_drivers(
        feature_names,
        feature_values,
        shap_values,
        top_n=5
    ):

        explanation = pd.DataFrame({

            "Feature":
                feature_names,

            "FeatureValue":
                feature_values,

            "SHAPValue":
                shap_values
        })

        # Negative SHAP means the variable
        # pushes predicted health DOWN.

        explanation = explanation[
            explanation[
                "SHAPValue"
            ] < 0
        ].copy()

        explanation[
            "AbsoluteSHAP"
        ] = np.abs(
            explanation[
                "SHAPValue"
            ]
        )

        explanation = (
            explanation
            .sort_values(
                "AbsoluteSHAP",
                ascending=False
            )
            .head(top_n)
        )

        drivers = []

        for _, row in (
            explanation.iterrows()
        ):

            drivers.append({

                "feature":
                    row["Feature"],

                "value":
                    float(
                        row[
                            "FeatureValue"
                        ]
                    ),

                "shap":
                    float(
                        row[
                            "SHAPValue"
                        ]
                    )
            })

        return drivers


    # ========================================================
    # EVALUATE ONE MANUFACTURING STATE
    # ========================================================

    def evaluate(
        self,
        row
    ):

        # ----------------------------------------------------
        # Lifecycle health
        # ----------------------------------------------------

        X_age = pd.DataFrame(
            [
                row[
                    self.age_features
                ].values
            ],
            columns=
                self.age_features
        )

        raw_lifecycle = float(
            self.age_model.predict(
                X_age
            )[0]
        )

        lifecycle_health = (
            self.display_health(
                raw_lifecycle
            )
        )

        # ----------------------------------------------------
        # Condition health
        # ----------------------------------------------------

        X_condition = pd.DataFrame(
            [
                row[
                    self.condition_features
                ].values
            ],
            columns=
                self.condition_features
        )

        raw_condition = float(
            self.condition_model.predict(
                X_condition
            )[0]
        )

        condition_health = (
            self.display_health(
                raw_condition
            )
        )

        # ----------------------------------------------------
        # Anomaly detection
        # ----------------------------------------------------

        X_anomaly = pd.DataFrame(
            [
                row[
                    self.anomaly_features
                ].values
            ],
            columns=
                self.anomaly_features
        )

        normality_score = float(
            self.anomaly_model
            .decision_function(
                X_anomaly
            )[0]
        )

        anomaly_score = (
            -normality_score
        )

        anomaly_label = int(
            self.anomaly_model.predict(
                X_anomaly
            )[0]
        )

        is_anomaly = (
            anomaly_label == -1
        )

        # ----------------------------------------------------
        # SHAP — lifecycle
        # ----------------------------------------------------

        age_shap = (
            self.age_explainer
            .shap_values(
                X_age
            )
        )

        if isinstance(
            age_shap,
            list
        ):
            age_shap = age_shap[0]

        age_shap = np.asarray(
            age_shap
        )[0]

        age_drivers = (
            self.degradation_drivers(

                feature_names=
                    self.age_features,

                feature_values=
                    X_age.iloc[0].values,

                shap_values=
                    age_shap,

                top_n=5
            )
        )

        # ----------------------------------------------------
        # SHAP — condition
        # ----------------------------------------------------

        condition_shap = (
            self.condition_explainer
            .shap_values(
                X_condition
            )
        )

        if isinstance(
            condition_shap,
            list
        ):
            condition_shap = (
                condition_shap[0]
            )

        condition_shap = (
            np.asarray(
                condition_shap
            )[0]
        )

        condition_drivers = (
            self.degradation_drivers(

                feature_names=
                    self.condition_features,

                feature_values=
                    X_condition
                    .iloc[0]
                    .values,

                shap_values=
                    condition_shap,

                top_n=5
            )
        )

        # ----------------------------------------------------
        # Health comparison
        # ----------------------------------------------------

        health_gap = (
            condition_health
            -
            lifecycle_health
        )

        gap_state = (
            health_gap_state(
                lifecycle_health,
                condition_health
            )
        )

        # ----------------------------------------------------
        # Construct Digital Twin state
        # ----------------------------------------------------

        twin_state = {

            "MachineID":
                "CNC-01",

            "ToolIndex":
                int(
                    row["ToolIndex"]
                ),

            "NumberOfCycle":
                int(
                    row["NumberOfCycle"]
                ),

            # Process context

            "MillingToolType":
                int(
                    row[
                        "MillingToolType"
                    ]
                ),

            "ADOC":
                float(
                    row["ADOC"]
                ),

            "RDOC":
                float(
                    row["RDOC"]
                ),

            "HardnessMean":
                float(
                    row[
                        "HardnessMean"
                    ]
                ),

            "ToolHolderLength":
                float(
                    row[
                        "ToolHolderLength"
                    ]
                ),

            # Lifecycle health

            "LifecycleHealthRaw":
                raw_lifecycle,

            "LifecycleHealth":
                lifecycle_health,

            "LifecycleBand":
                health_band(
                    lifecycle_health
                ),

            # Condition health

            "ConditionHealthRaw":
                raw_condition,

            "ConditionHealth":
                condition_health,

            "ConditionBand":
                health_band(
                    condition_health
                ),

            # Difference between models

            "HealthGap":
                health_gap,

            "HealthGapState":
                gap_state,

            # Unsupervised condition monitoring

            "AnomalyScore":
                anomaly_score,

            "IsAnomaly":
                bool(
                    is_anomaly
                ),

            # Ground truth is available only
            # because this is an experimental
            # validation dataset.

            "ActualHealth":
                float(
                    row[TARGET]
                ),

            # Explainability

            "LifecycleDrivers":
                age_drivers,

            "ConditionDrivers":
                condition_drivers
        }

        return twin_state


# ============================================================
# FLATTEN STATE FOR CSV
# ============================================================

def flatten_state(
    state
):

    output = {

        key: value

        for key, value
        in state.items()

        if key not in [
            "LifecycleDrivers",
            "ConditionDrivers"
        ]
    }

    # --------------------------------------------------------
    # Lifecycle SHAP drivers
    # --------------------------------------------------------

    for i, driver in enumerate(
        state[
            "LifecycleDrivers"
        ],
        start=1
    ):

        output[
            f"LifecycleDriver{i}"
        ] = driver[
            "feature"
        ]

        output[
            f"LifecycleDriver{i}_SHAP"
        ] = driver[
            "shap"
        ]

    # --------------------------------------------------------
    # Condition SHAP drivers
    # --------------------------------------------------------

    for i, driver in enumerate(
        state[
            "ConditionDrivers"
        ],
        start=1
    ):

        output[
            f"ConditionDriver{i}"
        ] = driver[
            "feature"
        ]

        output[
            f"ConditionDriver{i}_SHAP"
        ] = driver[
            "shap"
        ]

    return output


# ============================================================
# BUILD TEST-TWIN STATES
# ============================================================

def build_twin_states():

    print("=" * 78)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 11 — DIGITAL TWIN STATE ENGINE")
    print("=" * 78)

    test = pd.read_csv(
        TEST_FILE
    )

    print(
        f"\nTest samples: {len(test)}"
    )

    print(
        f"Test tools: "
        f"{sorted(test['ToolIndex'].unique())}"
    )

    engine = (
        DigitalTwinEngine()
    )

    twin_states = []

    for _, row in (
        test.iterrows()
    ):

        state = (
            engine.evaluate(
                row
            )
        )

        twin_states.append(
            flatten_state(
                state
            )
        )

    states_df = pd.DataFrame(
        twin_states
    )

    states_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✓ Digital Twin states saved:\n"
        f"{OUTPUT_FILE}"
    )

    return states_df


# ============================================================
# SUMMARY
# ============================================================

def summarize_states(
    states
):

    print("\n" + "=" * 78)
    print("DIGITAL TWIN STATE SUMMARY")
    print("=" * 78)

    print(
        "\nLifecycle health bands:"
    )

    print(
        states[
            "LifecycleBand"
        ]
        .value_counts()
    )

    print(
        "\nCondition health bands:"
    )

    print(
        states[
            "ConditionBand"
        ]
        .value_counts()
    )

    print(
        "\nHealth-gap states:"
    )

    print(
        states[
            "HealthGapState"
        ]
        .value_counts()
    )

    print(
        "\nAnomaly states:"
    )

    print(
        states[
            "IsAnomaly"
        ]
        .value_counts()
    )

    print(
        "\nMean absolute health gap:"
    )

    print(
        f"{states['HealthGap'].abs().mean():.4f}"
    )

    with open(
        SUMMARY_FILE,
        "w"
    ) as file:

        file.write(
            "AI MANUFACTURING DIGITAL TWIN\n"
        )

        file.write(
            "PHASE 11 DIGITAL TWIN STATE ENGINE\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            f"Samples: {len(states)}\n\n"
        )

        file.write(
            "The Digital Twin combines:\n"
        )

        file.write(
            "1. Age-aware lifecycle health\n"
        )

        file.write(
            "2. Current/context condition health\n"
        )

        file.write(
            "3. Isolation Forest anomaly score\n"
        )

        file.write(
            "4. SHAP degradation drivers\n\n"
        )

        file.write(
            "Health-gap interpretation:\n"
        )

        file.write(
            "Condition health substantially below "
            "lifecycle health may indicate "
            "unexpected/accelerated degradation.\n"
        )

        file.write(
            "Condition health substantially above "
            "lifecycle health indicates the observed "
            "condition looks healthier than its "
            "lifecycle estimate.\n"
        )

    print(
        f"\n✓ Summary saved:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# SHOW REPRESENTATIVE STATE
# ============================================================

def show_representative_state(
    states
):

    # Prefer a genuine anomalous / low-health
    # observation from a tool with a trajectory.

    candidates = states[
        (
            states["IsAnomaly"]
            ==
            True
        )
        &
        (
            states["ActualHealth"]
            <
            0.25
        )
        &
        (
            states["ToolIndex"]
            !=
            11
        )
    ]

    if candidates.empty:

        candidates = (
            states.copy()
        )

    row = (
        candidates
        .sort_values(
            "ActualHealth"
        )
        .iloc[0]
    )

    print("\n" + "=" * 78)
    print("REPRESENTATIVE DIGITAL TWIN STATE")
    print("=" * 78)

    print(
        f"\nMachine        : "
        f"{row['MachineID']}"
    )

    print(
        f"Tool           : "
        f"{int(row['ToolIndex'])}"
    )

    print(
        f"Cycle          : "
        f"{int(row['NumberOfCycle'])}"
    )

    print(
        "\nHEALTH"
    )

    print(
        f"Lifecycle      : "
        f"{row['LifecycleHealth'] * 100:.1f}% "
        f"({row['LifecycleBand']})"
    )

    print(
        f"Condition      : "
        f"{row['ConditionHealth'] * 100:.1f}% "
        f"({row['ConditionBand']})"
    )

    print(
        f"Actual         : "
        f"{row['ActualHealth'] * 100:.1f}%"
    )

    print(
        "\nCONDITION MONITORING"
    )

    print(
        f"Anomaly        : "
        f"{row['IsAnomaly']}"
    )

    print(
        f"Anomaly score  : "
        f"{row['AnomalyScore']:.4f}"
    )

    print(
        "\nMODEL AGREEMENT"
    )

    print(
        f"Health gap     : "
        f"{row['HealthGap']:+.4f}"
    )

    print(
        f"Gap state      : "
        f"{row['HealthGapState']}"
    )

    print(
        "\nTOP CONDITION DRIVERS"
    )

    for i in range(
        1,
        4
    ):

        feature_column = (
            f"ConditionDriver{i}"
        )

        shap_column = (
            f"ConditionDriver{i}_SHAP"
        )

        if (
            feature_column
            in
            row.index
        ):

            print(
                f"{i}. "
                f"{row[feature_column]} "
                f"(SHAP "
                f"{row[shap_column]:+.4f})"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    states = (
        build_twin_states()
    )

    summarize_states(
        states
    )

    show_representative_state(
        states
    )

    print("\n" + "=" * 78)
    print("PHASE 11 COMPLETE")
    print("=" * 78)

    print(
        "\nNext:"
    )

    print(
        "Decision-support engine and "
        "maintenance recommendations."
    )


if __name__ == "__main__":
    main()