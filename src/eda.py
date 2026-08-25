from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
CLEAN_FILE = PROCESSED_DIR / "milling_clean.csv"

MANIFEST_FILE = REPORT_DIR / "phase3_feature_manifest.json"

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TARGET = "CycleToFailureNormalized"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 75)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 4 — EXPLORATORY DATA ANALYSIS")
    print("=" * 75)

    required_files = [
        TRAIN_FILE,
        TEST_FILE,
        CLEAN_FILE,
        MANIFEST_FILE
    ]

    for file in required_files:

        if not file.exists():

            raise FileNotFoundError(
                f"Required file missing:\n{file}"
            )

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    full = pd.read_csv(CLEAN_FILE)

    with open(MANIFEST_FILE, "r") as file:
        manifest = json.load(file)

    print("\nDataset loaded successfully.")

    print(
        f"\nFull  : {full.shape}"
    )

    print(
        f"Train : {train.shape}"
    )

    print(
        f"Test  : {test.shape}"
    )

    return full, train, test, manifest


# ============================================================
# SPLIT AUDIT
# ============================================================

def split_audit(train, test):

    print("\n" + "=" * 75)
    print("1. TRAIN / TEST AUDIT")
    print("=" * 75)

    train_tools = sorted(
        train["ToolIndex"]
        .unique()
        .tolist()
    )

    test_tools = sorted(
        test["ToolIndex"]
        .unique()
        .tolist()
    )

    overlap = set(train_tools).intersection(
        set(test_tools)
    )

    print(
        f"\nTraining rows  : {len(train)}"
    )

    print(
        f"Testing rows   : {len(test)}"
    )

    print(
        f"Training tools : {train_tools}"
    )

    print(
        f"Testing tools  : {test_tools}"
    )

    print(
        f"Tool overlap   : {overlap}"
    )

    if overlap:

        raise ValueError(
            "Train/test tool leakage detected."
        )

    print(
        "\n✓ Final test tools remain completely unseen."
    )


# ============================================================
# PER-TOOL SUMMARY
# ============================================================

def create_tool_summary(full):

    print("\n" + "=" * 75)
    print("2. TOOL-LEVEL SUMMARY")
    print("=" * 75)

    summary = (
        full
        .groupby("ToolIndex")
        .agg(
            number_of_samples=("ToolIndex", "size"),

            min_cycle=(
                "NumberOfCycle",
                "min"
            ),

            max_cycle=(
                "NumberOfCycle",
                "max"
            ),

            start_health=(
                TARGET,
                "max"
            ),

            end_health=(
                TARGET,
                "min"
            ),

            mean_health=(
                TARGET,
                "mean"
            ),

            MillingToolType=(
                "MillingToolType",
                "first"
            ),

            ADOC=(
                "ADOC",
                "first"
            ),

            RDOC=(
                "RDOC",
                "first"
            ),

            HardnessMean=(
                "HardnessMean",
                "first"
            ),

            ToolHolderLength=(
                "ToolHolderLength",
                "first"
            )
        )
        .reset_index()
        .sort_values("ToolIndex")
    )

    print(summary.to_string(index=False))

    output = (
        REPORT_DIR /
        "phase4_tool_summary.csv"
    )

    summary.to_csv(
        output,
        index=False
    )

    print(
        f"\n✓ Saved:\n{output}"
    )

    return summary


# ============================================================
# CYCLES PER TOOL
# ============================================================

def plot_cycles_per_tool(tool_summary):

    print("\nCreating cycles-per-tool plot...")

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        tool_summary["ToolIndex"].astype(str),
        tool_summary["number_of_samples"]
    )

    plt.xlabel(
        "Tool Index"
    )

    plt.ylabel(
        "Number of Milling Cycles"
    )

    plt.title(
        "Number of Recorded Cycles per Cutting Tool"
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR /
        "cycles_per_tool.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"✓ {output}"
    )


# ============================================================
# HEALTH TRAJECTORY
# ============================================================

def plot_tool_health_trajectory(full):

    print("\nCreating tool-health trajectories...")

    plt.figure(
        figsize=(11, 7)
    )

    for tool_id, tool_df in full.groupby(
        "ToolIndex"
    ):

        tool_df = (
            tool_df
            .sort_values("NumberOfCycle")
        )

        plt.plot(
            tool_df["NumberOfCycle"],
            tool_df[TARGET],
            label=f"Tool {tool_id}",
            alpha=0.75
        )

    plt.xlabel(
        "Number of Cycle"
    )

    plt.ylabel(
        "Normalized Remaining Tool Life"
    )

    plt.title(
        "Tool Degradation Trajectories"
    )

    plt.legend(
        fontsize=7,
        ncol=2
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR /
        "tool_health_trajectories.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"✓ {output}"
    )


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def plot_target_distribution(
    train,
    test
):

    print("\nCreating target-distribution plot...")

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        train[TARGET],
        bins=25,
        alpha=0.65,
        label="Training"
    )

    plt.hist(
        test[TARGET],
        bins=25,
        alpha=0.65,
        label="Testing"
    )

    plt.xlabel(
        "CycleToFailureNormalized"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Target Distribution — Train vs Test"
    )

    plt.legend()

    plt.tight_layout()

    output = (
        FIGURE_DIR /
        "target_distribution_train_test.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"✓ {output}"
    )


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_sensor_correlations(
    train,
    manifest
):

    print("\n" + "=" * 75)
    print("3. SENSOR / TARGET CORRELATION")
    print("=" * 75)

    sensor_columns = manifest[
        "sensor_features"
    ]

    correlations = []

    for column in sensor_columns:

        corr = train[column].corr(
            train[TARGET]
        )

        correlations.append(
            {
                "feature": column,
                "correlation": corr,
                "absolute_correlation":
                    abs(corr)
            }
        )

    corr_df = pd.DataFrame(
        correlations
    )

    corr_df = (
        corr_df
        .sort_values(
            "absolute_correlation",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        "\nTop 20 sensor features "
        "associated with tool health:\n"
    )

    print(
        corr_df.head(20).to_string(
            index=False
        )
    )

    output = (
        REPORT_DIR /
        "phase4_sensor_target_correlations.csv"
    )

    corr_df.to_csv(
        output,
        index=False
    )

    print(
        f"\n✓ Saved:\n{output}"
    )

    return corr_df


# ============================================================
# TOP CORRELATIONS PLOT
# ============================================================

def plot_top_correlations(corr_df):

    top = (
        corr_df
        .head(15)
        .sort_values(
            "absolute_correlation"
        )
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        top["feature"],
        top["absolute_correlation"]
    )

    plt.xlabel(
        "Absolute Pearson Correlation"
    )

    plt.title(
        "Top Sensor Features Associated with Tool Health"
    )

    plt.tight_layout()

    output = (
        FIGURE_DIR /
        "top_sensor_correlations.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"✓ {output}"
    )


# ============================================================
# FEATURE TYPE ANALYSIS
# ============================================================

def classify_sensor_family(feature):

    if feature.startswith(
        "Accelerometer"
    ):

        return "Vibration"

    if feature.startswith(
        "Current"
    ):

        return "Electrical Current"

    return "Other"


def classify_statistic(feature):

    feature_lower = feature.lower()

    statistics = [
        "min",
        "max",
        "mean",
        "std",
        "skewness",
        "kurtosis"
    ]

    for statistic in statistics:

        if feature_lower.endswith(
            statistic
        ):

            return statistic

    return "unknown"


def analyse_sensor_families(
    corr_df
):

    corr_df = corr_df.copy()

    corr_df["sensor_family"] = (
        corr_df["feature"]
        .apply(
            classify_sensor_family
        )
    )

    corr_df["statistic"] = (
        corr_df["feature"]
        .apply(
            classify_statistic
        )
    )

    family_summary = (
        corr_df
        .groupby("sensor_family")
        ["absolute_correlation"]
        .agg(
            [
                "mean",
                "median",
                "max"
            ]
        )
        .reset_index()
    )

    statistic_summary = (
        corr_df
        .groupby("statistic")
        ["absolute_correlation"]
        .agg(
            [
                "mean",
                "median",
                "max"
            ]
        )
        .reset_index()
    )

    family_file = (
        REPORT_DIR /
        "phase4_sensor_family_summary.csv"
    )

    statistic_file = (
        REPORT_DIR /
        "phase4_statistic_summary.csv"
    )

    family_summary.to_csv(
        family_file,
        index=False
    )

    statistic_summary.to_csv(
        statistic_file,
        index=False
    )

    print(
        "\nSensor-family correlation summary:"
    )

    print(
        family_summary.to_string(
            index=False
        )
    )

    print(
        "\nStatistical-feature summary:"
    )

    print(
        statistic_summary.to_string(
            index=False
        )
    )


# ============================================================
# NUMBER OF CYCLE ANALYSIS
# ============================================================

def analyse_cycle_relationship(
    train
):

    print("\n" + "=" * 75)
    print("4. TOOL AGE / TARGET RELATIONSHIP")
    print("=" * 75)

    overall_corr = (
        train["NumberOfCycle"]
        .corr(
            train[TARGET]
        )
    )

    print(
        "\nTraining-data correlation:"
    )

    print(
        "NumberOfCycle vs "
        f"{TARGET}: "
        f"{overall_corr:.4f}"
    )

    per_tool_results = []

    for tool_id, group in train.groupby(
        "ToolIndex"
    ):

        correlation = (
            group["NumberOfCycle"]
            .corr(
                group[TARGET]
            )
        )

        per_tool_results.append(
            {
                "ToolIndex":
                    tool_id,

                "CycleHealthCorrelation":
                    correlation
            }
        )

    per_tool = pd.DataFrame(
        per_tool_results
    )

    print(
        "\nPer-tool cycle/health correlation:"
    )

    print(
        per_tool.to_string(
            index=False
        )
    )

    output = (
        REPORT_DIR /
        "phase4_cycle_health_correlation.csv"
    )

    per_tool.to_csv(
        output,
        index=False
    )


# ============================================================
# PROCESS CONTEXT
# ============================================================

def analyse_process_context(
    train,
    test,
    manifest
):

    print("\n" + "=" * 75)
    print("5. MANUFACTURING CONTEXT")
    print("=" * 75)

    context_features = manifest[
        "process_context_features"
    ]

    rows = []

    for feature in context_features:

        train_unique = sorted(
            train[feature]
            .dropna()
            .unique()
            .tolist()
        )

        test_unique = sorted(
            test[feature]
            .dropna()
            .unique()
            .tolist()
        )

        rows.append(
            {
                "feature":
                    feature,

                "train_unique_values":
                    str(train_unique),

                "test_unique_values":
                    str(test_unique),

                "test_values_seen_in_train":
                    set(test_unique)
                    .issubset(
                        set(train_unique)
                    )
            }
        )

        print(
            f"\n{feature}"
        )

        print(
            f"TRAIN: {train_unique}"
        )

        print(
            f"TEST : {test_unique}"
        )

    context_df = pd.DataFrame(
        rows
    )

    output = (
        REPORT_DIR /
        "phase4_process_context_audit.csv"
    )

    context_df.to_csv(
        output,
        index=False
    )

    print(
        f"\n✓ Saved:\n{output}"
    )


# ============================================================
# TOP FEATURES OVER TOOL LIFE
# ============================================================

def plot_top_features_by_tool(
    full,
    corr_df,
    number_of_features=3
):

    top_features = (
        corr_df
        .head(number_of_features)
        ["feature"]
        .tolist()
    )

    for feature in top_features:

        plt.figure(
            figsize=(11, 7)
        )

        for tool_id, tool_df in full.groupby(
            "ToolIndex"
        ):

            tool_df = (
                tool_df
                .sort_values(
                    "NumberOfCycle"
                )
            )

            plt.plot(
                tool_df[
                    "NumberOfCycle"
                ],
                tool_df[feature],
                label=f"Tool {tool_id}",
                alpha=0.7
            )

        plt.xlabel(
            "Number of Cycle"
        )

        plt.ylabel(
            feature
        )

        plt.title(
            f"{feature}\n"
            "Evolution Across Tool Life"
        )

        plt.legend(
            fontsize=6,
            ncol=2
        )

        plt.grid(
            alpha=0.2
        )

        plt.tight_layout()

        safe_name = (
            feature
            .replace(" ", "_")
            .replace("/", "_")
            .replace("+", "plus")
            .replace("-", "minus")
        )

        output = (
            FIGURE_DIR /
            f"feature_trend_{safe_name}.png"
        )

        plt.savefig(
            output,
            dpi=200
        )

        plt.close()

        print(
            f"✓ {output}"
        )


# ============================================================
# EDA SUMMARY
# ============================================================

def save_eda_summary(
    train,
    test,
    corr_df
):

    output = (
        REPORT_DIR /
        "phase4_eda_summary.txt"
    )

    with open(
        output,
        "w"
    ) as file:

        file.write(
            "AI MANUFACTURING DIGITAL TWIN\n"
        )

        file.write(
            "PHASE 4 EDA SUMMARY\n"
        )

        file.write(
            "=" * 65
            + "\n\n"
        )

        file.write(
            f"Training rows: {len(train)}\n"
        )

        file.write(
            f"Testing rows: {len(test)}\n\n"
        )

        file.write(
            "Training tools:\n"
        )

        file.write(
            str(
                sorted(
                    train[
                        "ToolIndex"
                    ].unique()
                )
            )
        )

        file.write("\n\nTesting tools:\n")

        file.write(
            str(
                sorted(
                    test[
                        "ToolIndex"
                    ].unique()
                )
            )
        )

        file.write(
            "\n\nTop sensor features "
            "by absolute target correlation:\n"
        )

        for _, row in (
            corr_df
            .head(15)
            .iterrows()
        ):

            file.write(
                f"{row['feature']} : "
                f"{row['correlation']:.4f}\n"
            )

    print(
        f"\n✓ EDA summary saved:\n{output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        full,
        train,
        test,
        manifest

    ) = load_data()

    split_audit(
        train,
        test
    )

    tool_summary = create_tool_summary(
        full
    )

    plot_cycles_per_tool(
        tool_summary
    )

    plot_tool_health_trajectory(
        full
    )

    plot_target_distribution(
        train,
        test
    )

    corr_df = (
        calculate_sensor_correlations(
            train,
            manifest
        )
    )

    plot_top_correlations(
        corr_df
    )

    analyse_sensor_families(
        corr_df
    )

    analyse_cycle_relationship(
        train
    )

    analyse_process_context(
        train,
        test,
        manifest
    )

    plot_top_features_by_tool(
        full,
        corr_df,
        number_of_features=3
    )

    save_eda_summary(
        train,
        test,
        corr_df
    )

    print("\n" + "=" * 75)
    print("PHASE 4 EDA COMPLETE")
    print("=" * 75)

    print(
        "\nNext phase:"
        "\nFeature analysis and baseline "
        "tool-health prediction."
    )


if __name__ == "__main__":
    main()