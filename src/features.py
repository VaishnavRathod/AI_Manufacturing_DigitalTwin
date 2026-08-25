from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

PHASE3_MANIFEST = (
    REPORT_DIR /
    "phase3_feature_manifest.json"
)

OUTPUT_MANIFEST = (
    REPORT_DIR /
    "phase5_feature_sets.json"
)


TARGET = "CycleToFailureNormalized"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 75)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 5 — FEATURE ENGINEERING")
    print("=" * 75)

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    with open(
        PHASE3_MANIFEST,
        "r"
    ) as file:

        manifest = json.load(file)

    print(
        f"\nTraining shape: {train.shape}"
    )

    print(
        f"Testing shape : {test.shape}"
    )

    return train, test, manifest


# ============================================================
# BUILD FEATURE GROUPS
# ============================================================

def build_feature_groups(manifest):

    sensor_features = (
        manifest["sensor_features"]
    )

    current_features = [
        feature
        for feature in sensor_features
        if feature.startswith("Current -")
    ]

    vibration_features = [
        feature
        for feature in sensor_features
        if feature.startswith("Accelerometer -")
    ]

    process_context = [
        "MillingToolType",
        "ADOC",
        "RDOC",
        "HardnessMean",
        "ToolHolderLength"
    ]

    cycle_feature = [
        "NumberOfCycle"
    ]

    print("\n" + "=" * 75)
    print("FEATURE GROUPS")
    print("=" * 75)

    print(
        f"\nAll sensor features : "
        f"{len(sensor_features)}"
    )

    print(
        f"Current features    : "
        f"{len(current_features)}"
    )

    print(
        f"Vibration features  : "
        f"{len(vibration_features)}"
    )

    print(
        f"Process context     : "
        f"{len(process_context)}"
    )

    return {
        "all_sensors":
            sensor_features,

        "current":
            current_features,

        "vibration":
            vibration_features,

        "process_context":
            process_context,

        "cycle":
            cycle_feature
    }


# ============================================================
# CREATE EXPERIMENTS
# ============================================================

def create_experiments(groups):

    experiments = {}

    # --------------------------------------------------------
    # EXPERIMENT A
    #
    # Main condition-based Digital Twin model
    # --------------------------------------------------------

    experiments[
        "A_condition_based"
    ] = (
        groups["all_sensors"]
        +
        groups["process_context"]
    )

    # --------------------------------------------------------
    # EXPERIMENT B
    #
    # Electrical-current monitoring only
    # --------------------------------------------------------

    experiments[
        "B_current_based"
    ] = (
        groups["current"]
        +
        groups["process_context"]
    )

    # --------------------------------------------------------
    # EXPERIMENT C
    #
    # Vibration monitoring only
    # --------------------------------------------------------

    experiments[
        "C_vibration_based"
    ] = (
        groups["vibration"]
        +
        groups["process_context"]
    )

    # --------------------------------------------------------
    # EXPERIMENT D
    #
    # Condition + explicit tool age
    # --------------------------------------------------------

    experiments[
        "D_age_aware"
    ] = (
        groups["all_sensors"]
        +
        groups["process_context"]
        +
        groups["cycle"]
    )

    return experiments


# ============================================================
# VALIDATE EXPERIMENTS
# ============================================================

def validate_experiments(
    train,
    test,
    experiments
):

    print("\n" + "=" * 75)
    print("EXPERIMENT VALIDATION")
    print("=" * 75)

    forbidden = {
        TARGET,
        "CycleToFailure",
        "ToolIndex",
        "SampleIndex",
        "FileName"
    }

    for name, features in experiments.items():

        print(f"\n{name}")

        # --------------------------------------------
        # Duplicate features
        # --------------------------------------------

        if len(features) != len(set(features)):

            raise ValueError(
                f"{name} contains duplicate features."
            )

        # --------------------------------------------
        # Leakage
        # --------------------------------------------

        overlap = (
            set(features)
            .intersection(forbidden)
        )

        if overlap:

            raise ValueError(
                f"{name} contains forbidden "
                f"features: {overlap}"
            )

        # --------------------------------------------
        # Column existence
        # --------------------------------------------

        missing_train = [
            column
            for column in features
            if column not in train.columns
        ]

        missing_test = [
            column
            for column in features
            if column not in test.columns
        ]

        if missing_train:

            raise ValueError(
                f"Missing training features: "
                f"{missing_train}"
            )

        if missing_test:

            raise ValueError(
                f"Missing testing features: "
                f"{missing_test}"
            )

        # --------------------------------------------
        # Numeric validation
        # --------------------------------------------

        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(
                train[column]
            )
        ]

        if non_numeric:

            raise ValueError(
                f"{name} contains "
                f"non-numeric features: "
                f"{non_numeric}"
            )

        print(
            f"Features: {len(features)}"
        )

        print(
            "✓ No leakage"
        )

        print(
            "✓ All features available"
        )

        print(
            "✓ All model inputs numeric"
        )


# ============================================================
# CHECK CONSTANT / LOW-VARIANCE FEATURES
# ============================================================

def feature_variance_audit(
    train,
    experiments
):

    print("\n" + "=" * 75)
    print("FEATURE VARIANCE AUDIT")
    print("=" * 75)

    all_features = sorted(
        set(
            feature
            for features
            in experiments.values()
            for feature
            in features
        )
    )

    rows = []

    for feature in all_features:

        series = train[feature]

        variance = (
            series.var()
            if pd.api.types.is_numeric_dtype(series)
            else np.nan
        )

        unique_count = (
            series.nunique()
        )

        rows.append(
            {
                "feature":
                    feature,

                "unique_values":
                    unique_count,

                "variance":
                    variance,

                "constant":
                    unique_count <= 1
            }
        )

    audit = pd.DataFrame(rows)

    constant = audit[
        audit["constant"]
    ]

    print(
        f"\nFeatures audited : "
        f"{len(audit)}"
    )

    print(
        f"Constant features: "
        f"{len(constant)}"
    )

    output = (
        REPORT_DIR /
        "phase5_feature_variance_audit.csv"
    )

    audit.to_csv(
        output,
        index=False
    )

    print(
        f"\n✓ Saved:\n{output}"
    )

    return audit


# ============================================================
# HIGH CORRELATION AUDIT
# ============================================================

def high_correlation_audit(
    train,
    groups,
    threshold=0.98
):

    print("\n" + "=" * 75)
    print("FEATURE REDUNDANCY AUDIT")
    print("=" * 75)

    sensor_features = (
        groups["all_sensors"]
    )

    correlation_matrix = (
        train[sensor_features]
        .corr()
        .abs()
    )

    pairs = []

    columns = (
        correlation_matrix.columns
    )

    for i in range(
        len(columns)
    ):

        for j in range(
            i + 1,
            len(columns)
        ):

            correlation = (
                correlation_matrix
                .iloc[i, j]
            )

            if correlation >= threshold:

                pairs.append(
                    {
                        "feature_1":
                            columns[i],

                        "feature_2":
                            columns[j],

                        "absolute_correlation":
                            correlation
                    }
                )

    pairs_df = pd.DataFrame(
        pairs
    )

    if not pairs_df.empty:

        pairs_df = (
            pairs_df
            .sort_values(
                "absolute_correlation",
                ascending=False
            )
            .reset_index(drop=True)
        )

    print(
        f"\nSensor pairs with "
        f"|correlation| >= {threshold}: "
        f"{len(pairs_df)}"
    )

    if not pairs_df.empty:

        print(
            "\nTop redundant pairs:\n"
        )

        print(
            pairs_df
            .head(20)
            .to_string(index=False)
        )

    output = (
        REPORT_DIR /
        "phase5_highly_correlated_features.csv"
    )

    pairs_df.to_csv(
        output,
        index=False
    )

    print(
        f"\n✓ Saved:\n{output}"
    )

    return pairs_df


# ============================================================
# SAVE FEATURE MANIFEST
# ============================================================

def save_manifest(
    groups,
    experiments
):

    output = {
        "target":
            TARGET,

        "feature_groups":
            groups,

        "experiments":
            experiments,

        "experiment_counts": {
            name: len(features)
            for name, features
            in experiments.items()
        },

        "notes": {

            "A_condition_based":
                (
                    "Primary Digital Twin model. "
                    "Uses machine-condition signals "
                    "and process context without "
                    "explicit cycle age."
                ),

            "B_current_based":
                (
                    "Tests electrical-current "
                    "monitoring as a reduced "
                    "sensor configuration."
                ),

            "C_vibration_based":
                (
                    "Tests accelerometer-based "
                    "condition monitoring."
                ),

            "D_age_aware":
                (
                    "Adds NumberOfCycle. "
                    "Evaluated separately because "
                    "cycle count is strongly related "
                    "to normalized remaining life."
                )
        }
    }

    with open(
        OUTPUT_MANIFEST,
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print(
        f"\n✓ Feature manifest saved:\n"
        f"{OUTPUT_MANIFEST}"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(experiments):

    print("\n" + "=" * 75)
    print("PHASE 5 EXPERIMENT DESIGN")
    print("=" * 75)

    for name, features in experiments.items():

        print(
            f"\n{name:25s}: "
            f"{len(features)} features"
        )

    print(
        "\nTarget:"
        f"\n{TARGET}"
    )

    print(
        "\nImportant:"
        "\nNo scaling has been performed."
    )

    print(
        "\nScaling/preprocessing will be "
        "fitted only on training folds "
        "inside the ML pipeline."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    train, test, manifest = (
        load_data()
    )

    groups = build_feature_groups(
        manifest
    )

    experiments = create_experiments(
        groups
    )

    validate_experiments(
        train,
        test,
        experiments
    )

    feature_variance_audit(
        train,
        experiments
    )

    high_correlation_audit(
        train,
        groups,
        threshold=0.98
    )

    save_manifest(
        groups,
        experiments
    )

    print_summary(
        experiments
    )

    print("\n" + "=" * 75)
    print("PHASE 5 COMPLETE")
    print("=" * 75)

    print(
        "\nNext: Group-aware baseline "
        "machine-learning models."
    )


if __name__ == "__main__":
    main()