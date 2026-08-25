from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

FEATURE_FILE = (
    REPORT_DIR /
    "phase5_feature_sets.json"
)

OUTPUT_FILE = (
    REPORT_DIR /
    "phase9_anomaly_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR /
    "phase9_anomaly_summary.txt"
)

MODEL_FILE = (
    MODEL_DIR /
    "isolation_forest_current.joblib"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "CycleToFailureNormalized"

NORMAL_HEALTH_THRESHOLD = 0.70

CONTAMINATION = 0.10

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 78)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 9 — ANOMALY DETECTION")
    print("=" * 78)

    train = pd.read_csv(
        TRAIN_FILE
    )

    test = pd.read_csv(
        TEST_FILE
    )

    with open(
        FEATURE_FILE,
        "r"
    ) as file:

        manifest = json.load(file)

    current_features = (
        manifest[
            "feature_groups"
        ][
            "current"
        ]
    )

    print(
        f"\nTraining rows : {len(train)}"
    )

    print(
        f"Testing rows  : {len(test)}"
    )

    print(
        f"Current sensor features : "
        f"{len(current_features)}"
    )

    return (
        train,
        test,
        current_features
    )


# ============================================================
# SELECT NORMAL REFERENCE DATA
# ============================================================

def select_normal_data(
    train
):

    normal = train[
        train[TARGET]
        >=
        NORMAL_HEALTH_THRESHOLD
    ].copy()

    print("\n" + "=" * 78)
    print("NORMAL REFERENCE POPULATION")
    print("=" * 78)

    print(
        f"\nThreshold:"
        f"\n{TARGET} >= "
        f"{NORMAL_HEALTH_THRESHOLD}"
    )

    print(
        f"\nNormal training samples : "
        f"{len(normal)}"
    )

    print(
        f"Normal training tools   : "
        f"{normal['ToolIndex'].nunique()}"
    )

    if len(normal) < 50:

        raise ValueError(
            "Too few normal samples "
            "for anomaly detection."
        )

    return normal


# ============================================================
# BUILD MODEL
# ============================================================

def build_anomaly_model():

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),

            (
                "isolation_forest",
                IsolationForest(

                    n_estimators=400,

                    contamination=
                        CONTAMINATION,

                    random_state=
                        RANDOM_STATE,

                    n_jobs=-1
                )
            )
        ]
    )

    return model


# ============================================================
# FIT MODEL
# ============================================================

def train_model(
    normal,
    features
):

    print("\n" + "=" * 78)
    print("TRAINING ANOMALY DETECTOR")
    print("=" * 78)

    X_normal = (
        normal[
            features
        ]
    )

    model = build_anomaly_model()

    model.fit(
        X_normal
    )

    joblib.dump(
        {
            "model":
                model,

            "features":
                features,

            "normal_health_threshold":
                NORMAL_HEALTH_THRESHOLD
        },
        MODEL_FILE
    )

    print(
        f"\n✓ Model saved:\n"
        f"{MODEL_FILE}"
    )

    return model


# ============================================================
# SCORE DATA
# ============================================================

def score_dataset(
    model,
    dataframe,
    features,
    dataset_name
):

    X = dataframe[
        features
    ]

    # sklearn:
    # higher decision_function = more normal
    #
    # We invert it so:
    # higher AnomalyScore = more abnormal
    normality_score = (
        model.decision_function(
            X
        )
    )

    anomaly_score = (
        -normality_score
    )

    prediction = (
        model.predict(
            X
        )
    )

    output = pd.DataFrame({

        "Dataset":
            dataset_name,

        "ToolIndex":
            dataframe[
                "ToolIndex"
            ].values,

        "NumberOfCycle":
            dataframe[
                "NumberOfCycle"
            ].values,

        "ActualHealth":
            dataframe[
                TARGET
            ].values,

        "AnomalyScore":
            anomaly_score,

        "IsolationForestLabel":
            prediction,

        "IsAnomaly":
            prediction == -1
    })

    return output


# ============================================================
# ANALYSE HEALTH BANDS
# ============================================================

def assign_health_band(
    health
):

    if health >= 0.75:

        return "HEALTHY"

    elif health >= 0.50:

        return "EARLY_WEAR"

    elif health >= 0.25:

        return "DEGRADED"

    return "CRITICAL"


def analyse_results(
    results
):

    results = results.copy()

    results[
        "HealthBand"
    ] = (
        results[
            "ActualHealth"
        ]
        .apply(
            assign_health_band
        )
    )

    print("\n" + "=" * 78)
    print("ANOMALY RATE BY TOOL HEALTH")
    print("=" * 78)

    band_summary = (
        results
        .groupby(
            "HealthBand"
        )
        .agg(

            Samples=(
                "IsAnomaly",
                "size"
            ),

            Anomalies=(
                "IsAnomaly",
                "sum"
            ),

            AnomalyRate=(
                "IsAnomaly",
                "mean"
            ),

            MeanAnomalyScore=(
                "AnomalyScore",
                "mean"
            )
        )
        .reset_index()
    )

    preferred_order = [

        "HEALTHY",
        "EARLY_WEAR",
        "DEGRADED",
        "CRITICAL"
    ]

    band_summary[
        "HealthBand"
    ] = pd.Categorical(

        band_summary[
            "HealthBand"
        ],

        categories=
            preferred_order,

        ordered=True
    )

    band_summary = (
        band_summary
        .sort_values(
            "HealthBand"
        )
    )

    print(
        "\n"
        +
        band_summary
        .to_string(
            index=False
        )
    )

    return (
        results,
        band_summary
    )


# ============================================================
# PER-TOOL SUMMARY
# ============================================================

def per_tool_summary(
    test_results
):

    summary = (
        test_results
        .groupby(
            "ToolIndex"
        )
        .agg(

            Samples=(
                "IsAnomaly",
                "size"
            ),

            AnomalyRate=(
                "IsAnomaly",
                "mean"
            ),

            MeanAnomalyScore=(
                "AnomalyScore",
                "mean"
            ),

            MeanHealth=(
                "ActualHealth",
                "mean"
            )
        )
        .reset_index()
    )

    print("\n" + "=" * 78)
    print("TEST TOOL ANOMALY SUMMARY")
    print("=" * 78)

    print(
        "\n"
        +
        summary.to_string(
            index=False
        )
    )

    return summary


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    combined_results,
    band_summary,
    tool_summary
):

    combined_results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    band_file = (
        REPORT_DIR /
        "phase9_anomaly_health_bands.csv"
    )

    tool_file = (
        REPORT_DIR /
        "phase9_anomaly_per_tool.csv"
    )

    band_summary.to_csv(
        band_file,
        index=False
    )

    tool_summary.to_csv(
        tool_file,
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
            "PHASE 9 ANOMALY DETECTION\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            "Method: Isolation Forest\n"
        )

        file.write(
            "Sensor family: "
            "Electrical current\n"
        )

        file.write(
            f"Normal reference threshold: "
            f"{NORMAL_HEALTH_THRESHOLD}\n"
        )

        file.write(
            f"Contamination: "
            f"{CONTAMINATION}\n\n"
        )

        file.write(
            "Important interpretation:\n"
        )

        file.write(
            "Anomaly detection is not "
            "a failure classifier.\n"
        )

        file.write(
            "It measures deviation from "
            "healthier reference behavior.\n"
        )

    print(
        "\nSaved:"
    )

    print(
        f"✓ {OUTPUT_FILE}"
    )

    print(
        f"✓ {band_file}"
    )

    print(
        f"✓ {tool_file}"
    )

    print(
        f"✓ {SUMMARY_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        train,
        test,
        features

    ) = load_data()

    normal = select_normal_data(
        train
    )

    model = train_model(
        normal,
        features
    )

    train_results = score_dataset(

        model,
        train,
        features,
        "TRAIN"
    )

    test_results = score_dataset(

        model,
        test,
        features,
        "TEST"
    )

    combined = pd.concat(

        [
            train_results,
            test_results
        ],

        ignore_index=True
    )

    (
        combined,
        health_band_summary

    ) = analyse_results(
        combined
    )

    test_only = combined[
        combined["Dataset"]
        ==
        "TEST"
    ]

    tool_summary = (
        per_tool_summary(
            test_only
        )
    )

    save_results(

        combined_results=
            combined,

        band_summary=
            health_band_summary,

        tool_summary=
            tool_summary
    )

    print("\n" + "=" * 78)
    print("PHASE 9 COMPLETE")
    print("=" * 78)

    print(
        "\nNext:"
    )

    print(
        "SHAP explainability for "
        "the supervised XGBoost models."
    )


if __name__ == "__main__":
    main()