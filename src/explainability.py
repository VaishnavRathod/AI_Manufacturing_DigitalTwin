from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

TEST_FILE = (
    PROCESSED_DIR /
    "test.csv"
)

ANOMALY_FILE = (
    REPORT_DIR /
    "phase9_anomaly_results.csv"
)

AGE_MODEL_FILE = (
    MODEL_DIR /
    "D_age_aware_xgboost.joblib"
)

CURRENT_MODEL_FILE = (
    MODEL_DIR /
    "B_current_based_xgboost.joblib"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "CycleToFailureNormalized"

PROCESS_CONTEXT = [
    "MillingToolType",
    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength"
]


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

def load_resources():

    print("=" * 78)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 10 — SHAP EXPLAINABILITY")
    print("=" * 78)

    test = pd.read_csv(
        TEST_FILE
    )

    age_payload = joblib.load(
        AGE_MODEL_FILE
    )

    current_payload = joblib.load(
        CURRENT_MODEL_FILE
    )

    models = {

        "D_age_aware":
            age_payload,

        "B_current_based":
            current_payload
    }

    print(
        f"\nTest samples: {len(test)}"
    )

    for name, payload in models.items():

        print(
            f"\n{name}"
        )

        print(
            f"Features: "
            f"{len(payload['features'])}"
        )

    return test, models


# ============================================================
# FEATURE FAMILY
# ============================================================

def get_feature_family(feature):

    if feature.startswith(
        "Current -"
    ):

        return "Electrical Current"

    if feature.startswith(
        "Accelerometer -"
    ):

        return "Vibration"

    if feature == "NumberOfCycle":

        return "Cycle Age"

    if feature in PROCESS_CONTEXT:

        return "Process Context"

    return "Other"


# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

def calculate_shap(
    test,
    model_payload,
    experiment_name
):

    model = (
        model_payload[
            "model"
        ]
    )

    features = (
        model_payload[
            "features"
        ]
    )

    X = test[
        features
    ].copy()

    print("\n" + "=" * 78)
    print(
        f"SHAP ANALYSIS — {experiment_name}"
    )
    print("=" * 78)

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X
        )
    )

    # Compatibility with versions of SHAP
    # that may return lists.
    if isinstance(
        shap_values,
        list
    ):

        shap_values = (
            shap_values[0]
        )

    shap_values = np.asarray(
        shap_values
    )

    print(
        f"\nSHAP matrix shape: "
        f"{shap_values.shape}"
    )

    return (
        X,
        shap_values,
        explainer
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def global_importance(
    X,
    shap_values,
    experiment_name
):

    mean_absolute_shap = (
        np.abs(
            shap_values
        )
        .mean(axis=0)
    )

    importance = pd.DataFrame({

        "Feature":
            X.columns,

        "MeanAbsoluteSHAP":
            mean_absolute_shap
    })

    importance[
        "FeatureFamily"
    ] = (
        importance[
            "Feature"
        ]
        .apply(
            get_feature_family
        )
    )

    importance = (
        importance
        .sort_values(
            "MeanAbsoluteSHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\nTop 20 global SHAP features:\n"
    )

    print(
        importance
        .head(20)
        .to_string(
            index=False
        )
    )

    output = (
        REPORT_DIR /
        f"phase10_{experiment_name}_global_shap.csv"
    )

    importance.to_csv(
        output,
        index=False
    )

    print(
        f"\n✓ Saved:\n{output}"
    )

    return importance


# ============================================================
# GLOBAL IMPORTANCE PLOT
# ============================================================

def plot_global_importance(
    importance,
    experiment_name,
    n_features=20
):

    top = (
        importance
        .head(n_features)
        .sort_values(
            "MeanAbsoluteSHAP"
        )
    )

    plt.figure(
        figsize=(11, 8)
    )

    plt.barh(
        top["Feature"],
        top["MeanAbsoluteSHAP"]
    )

    plt.xlabel(
        "Mean |SHAP value|"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{experiment_name}\n"
        "Global Feature Importance"
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR /
        f"phase10_{experiment_name}_global_shap.png"
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ {output}"
    )


# ============================================================
# FEATURE FAMILY IMPORTANCE
# ============================================================

def feature_family_importance(
    importance,
    experiment_name
):

    family = (
        importance
        .groupby(
            "FeatureFamily"
        )
        .agg(

            TotalMeanAbsoluteSHAP=(
                "MeanAbsoluteSHAP",
                "sum"
            ),

            MeanFeatureImportance=(
                "MeanAbsoluteSHAP",
                "mean"
            ),

            NumberFeatures=(
                "Feature",
                "size"
            )
        )
        .reset_index()
    )

    total = (
        family[
            "TotalMeanAbsoluteSHAP"
        ]
        .sum()
    )

    family[
        "ImportanceShare"
    ] = (
        family[
            "TotalMeanAbsoluteSHAP"
        ]
        /
        total
    )

    family = (
        family
        .sort_values(
            "TotalMeanAbsoluteSHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\nFeature-family contribution:\n"
    )

    print(
        family.to_string(
            index=False
        )
    )

    output = (
        REPORT_DIR /
        f"phase10_{experiment_name}_family_importance.csv"
    )

    family.to_csv(
        output,
        index=False
    )

    print(
        f"\n✓ Saved:\n{output}"
    )

    return family


# ============================================================
# FIND MOST ANOMALOUS TEST SAMPLE
# ============================================================

def find_most_anomalous_sample(
    test
):

    if not ANOMALY_FILE.exists():

        print(
            "\nPhase 9 anomaly file "
            "not found."
        )

        return None

    anomaly = pd.read_csv(
        ANOMALY_FILE
    )

    anomaly = anomaly[
        anomaly["Dataset"]
        ==
        "TEST"
    ].copy()

    # Tool 11 contains only one sample.
    # We retain it in evaluation, but avoid
    # using it as the representative example.
    tool_counts = (
        anomaly[
            "ToolIndex"
        ]
        .value_counts()
    )

    valid_tools = (
        tool_counts[
            tool_counts >= 5
        ]
        .index
    )

    anomaly = anomaly[
        anomaly[
            "ToolIndex"
        ]
        .isin(
            valid_tools
        )
    ]

    if anomaly.empty:

        return None

    row = (
        anomaly
        .sort_values(
            "AnomalyScore",
            ascending=False
        )
        .iloc[0]
    )

    tool_id = row[
        "ToolIndex"
    ]

    cycle = row[
        "NumberOfCycle"
    ]

    matches = test[
        (
            test["ToolIndex"]
            ==
            tool_id
        )
        &
        (
            test["NumberOfCycle"]
            ==
            cycle
        )
    ]

    if matches.empty:

        print(
            "\nCould not match "
            "anomaly sample to test data."
        )

        return None

    selected_index = (
        matches.index[0]
    )

    print("\n" + "=" * 78)
    print("REPRESENTATIVE ABNORMAL SAMPLE")
    print("=" * 78)

    print(
        f"\nToolIndex      : "
        f"{tool_id}"
    )

    print(
        f"NumberOfCycle  : "
        f"{cycle}"
    )

    print(
        f"Actual Health  : "
        f"{row['ActualHealth']:.4f}"
    )

    print(
        f"Anomaly Score  : "
        f"{row['AnomalyScore']:.4f}"
    )

    return selected_index


# ============================================================
# LOCAL EXPLANATION
# ============================================================

def explain_sample(
    test,
    X,
    shap_values,
    model_payload,
    sample_index,
    experiment_name,
    top_n=12
):

    if sample_index is None:

        return None

    features = (
        model_payload[
            "features"
        ]
    )

    model = (
        model_payload[
            "model"
        ]
    )

    # X retains the same index as test
    row_position = (
        X.index.get_loc(
            sample_index
        )
    )

    values = (
        shap_values[
            row_position
        ]
    )

    feature_values = (
        X.loc[
            sample_index
        ]
    )

    prediction = float(
        model.predict(
            X.loc[
                [sample_index]
            ]
        )[0]
    )

    actual = float(
        test.loc[
            sample_index,
            TARGET
        ]
    )

    local = pd.DataFrame({

        "Feature":
            features,

        "FeatureValue":
            feature_values.values,

        "SHAPValue":
            values,

        "AbsoluteSHAP":
            np.abs(
                values
            )
    })

    local[
        "FeatureFamily"
    ] = (
        local[
            "Feature"
        ]
        .apply(
            get_feature_family
        )
    )

    local = (
        local
        .sort_values(
            "AbsoluteSHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    local[
        "Direction"
    ] = np.where(

        local[
            "SHAPValue"
        ]
        < 0,

        "Pushes health DOWN",

        "Pushes health UP"
    )

    local[
        "Experiment"
    ] = experiment_name

    local[
        "ToolIndex"
    ] = test.loc[
        sample_index,
        "ToolIndex"
    ]

    local[
        "NumberOfCycle"
    ] = test.loc[
        sample_index,
        "NumberOfCycle"
    ]

    local[
        "ActualHealth"
    ] = actual

    local[
        "PredictedHealth"
    ] = prediction

    print("\n" + "=" * 78)

    print(
        f"LOCAL EXPLANATION — "
        f"{experiment_name}"
    )

    print("=" * 78)

    print(
        f"\nActual health    : "
        f"{actual:.4f}"
    )

    print(
        f"Predicted health : "
        f"{prediction:.4f}"
    )

    print(
        "\nTop contributors:\n"
    )

    print(
        local[
            [
                "Feature",
                "FeatureValue",
                "SHAPValue",
                "Direction"
            ]
        ]
        .head(top_n)
        .to_string(
            index=False
        )
    )

    output = (
        REPORT_DIR /
        f"phase10_{experiment_name}_local_explanation.csv"
    )

    local.to_csv(
        output,
        index=False
    )

    # --------------------------------------------------------
    # Plot top local contributions
    # --------------------------------------------------------

    plot_data = (
        local
        .head(top_n)
        .sort_values(
            "AbsoluteSHAP"
        )
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        plot_data[
            "Feature"
        ],
        plot_data[
            "SHAPValue"
        ]
    )

    plt.axvline(
        x=0,
        linewidth=1
    )

    plt.xlabel(
        "SHAP Contribution to Predicted Health"
    )

    plt.title(
        f"{experiment_name}\n"
        f"Tool {int(test.loc[sample_index, 'ToolIndex'])}, "
        f"Cycle {int(test.loc[sample_index, 'NumberOfCycle'])}"
    )

    plt.tight_layout()

    figure_file = (
        FIGURE_DIR /
        f"phase10_{experiment_name}_local_explanation.png"
    )

    plt.savefig(
        figure_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n✓ {output}"
    )

    print(
        f"✓ {figure_file}"
    )

    return local


# ============================================================
# SAVE COMBINED SUMMARY
# ============================================================

def save_combined_summary(
    model_results
):

    output = (
        REPORT_DIR /
        "phase10_shap_summary.txt"
    )

    with open(
        output,
        "w"
    ) as file:

        file.write(
            "AI MANUFACTURING DIGITAL TWIN\n"
        )

        file.write(
            "PHASE 10 — SHAP EXPLAINABILITY\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            "SHAP interpretation:\n"
        )

        file.write(
            "Positive SHAP values push "
            "predicted normalized tool health upward.\n"
        )

        file.write(
            "Negative SHAP values push "
            "predicted normalized tool health downward.\n\n"
        )

        for (
            experiment,
            importance

        ) in model_results.items():

            file.write(
                f"{experiment}\n"
            )

            file.write(
                "-" * 50
                +
                "\n"
            )

            for _, row in (
                importance
                .head(10)
                .iterrows()
            ):

                file.write(

                    f"{row['Feature']} | "
                    f"{row['MeanAbsoluteSHAP']:.6f}\n"
                )

            file.write("\n")

    print(
        f"\n✓ Summary saved:\n"
        f"{output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    test, models = (
        load_resources()
    )

    representative_index = (
        find_most_anomalous_sample(
            test
        )
    )

    model_results = {}

    for (
        experiment,
        payload

    ) in models.items():

        (
            X,
            shap_values,
            explainer

        ) = calculate_shap(

            test=test,

            model_payload=
                payload,

            experiment_name=
                experiment
        )

        importance = (
            global_importance(

                X,
                shap_values,
                experiment
            )
        )

        plot_global_importance(
            importance,
            experiment
        )

        feature_family_importance(
            importance,
            experiment
        )

        explain_sample(

            test=test,

            X=X,

            shap_values=
                shap_values,

            model_payload=
                payload,

            sample_index=
                representative_index,

            experiment_name=
                experiment
        )

        model_results[
            experiment
        ] = importance

    save_combined_summary(
        model_results
    )

    print("\n" + "=" * 78)
    print("PHASE 10 COMPLETE")
    print("=" * 78)

    print(
        "\nNext:"
    )

    print(
        "Digital Twin state engine "
        "and decision-support logic."
    )


if __name__ == "__main__":
    main()