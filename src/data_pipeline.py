from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_FILE = RAW_DATA_DIR / "FeatureAndMetadata_Milling.csv"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "CycleToFailureNormalized"

LEAKAGE_COLUMNS = [
    "CycleToFailure"
]

IDENTIFIER_COLUMNS = [
    "FileName",
    "SampleIndex",
    "ToolIndex"
]

PROCESS_CONTEXT_COLUMNS = [
    "MillingToolType",
    "ADOC",
    "RDOC",
    "HardnessMean",
    "ToolHolderLength"
]

CYCLE_FEATURE = "NumberOfCycle"


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_dataset():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}"
        )

    print("=" * 75)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 3 — DATA PREPARATION PIPELINE")
    print("=" * 75)

    print(f"\nLoading dataset:\n{DATA_FILE}")

    raw_df = pd.read_csv(
        DATA_FILE,
        sep=None,
        engine="python",
        dtype=str
    )

    print(
        f"\nRaw shape: "
        f"{raw_df.shape[0]} rows × "
        f"{raw_df.shape[1]} columns"
    )

    return raw_df


# ============================================================
# FIX EMBEDDED HEADER
# ============================================================

def fix_embedded_header(raw_df):

    generic_headers = all(
        str(column).startswith("Column")
        for column in raw_df.columns
    )

    first_row = (
        raw_df.iloc[0]
        .astype(str)
        .str.strip()
    )

    first_row_values = first_row.tolist()

    expected_header_values = {
        "CycleToFailure",
        "CycleToFailureNormalized",
        "TollIndex",
        "ToolIndex",
        "MillingToolType"
    }

    embedded_header_detected = bool(
        expected_header_values.intersection(
            set(first_row_values)
        )
    )

    if generic_headers and embedded_header_detected:

        print("\nEmbedded dataset header detected.")

        df = raw_df.iloc[1:].copy()

        df.columns = first_row_values

        df = df.reset_index(drop=True)

        print("✓ First row promoted to dataframe header.")

    else:

        print(
            "\nNo embedded header correction required."
        )

        df = raw_df.copy()

    return df


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_column_names(df):

    df = df.copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # Dataset contains "TollIndex".
    # Internally we standardize this to "ToolIndex".
    if (
        "TollIndex" in df.columns
        and
        "ToolIndex" not in df.columns
    ):

        df = df.rename(
            columns={
                "TollIndex": "ToolIndex"
            }
        )

        print(
            "\n✓ Renamed "
            "'TollIndex' → 'ToolIndex'"
        )

    return df


# ============================================================
# CONVERT NUMERIC DATA
# ============================================================

def convert_numeric_columns(df):

    df = df.copy()

    converted_columns = []

    for column in df.columns:

        cleaned = (
            df[column]
            .astype(str)
            .str.strip()
        )

        # Support decimal comma if present
        numeric_candidate = (
            cleaned
            .str.replace(
                ",",
                ".",
                regex=False
            )
        )

        numeric = pd.to_numeric(
            numeric_candidate,
            errors="coerce"
        )

        # Convert only when every value is numeric.
        if numeric.notna().all():

            df[column] = numeric

            converted_columns.append(column)

        else:

            df[column] = cleaned

    print(
        f"\n✓ Converted "
        f"{len(converted_columns)} "
        f"columns to numeric datatype."
    )

    return df


# ============================================================
# DETECT SENSOR FEATURES
# ============================================================

def detect_sensor_features(df):

    sensor_columns = [

        column

        for column in df.columns

        if (
            column.startswith("Accelerometer -")
            or
            column.startswith("Current -")
        )
    ]

    return sensor_columns


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    print("\n" + "=" * 75)
    print("DATASET VALIDATION")
    print("=" * 75)

    errors = []

    # --------------------------------------------------------
    # Dataset dimensions
    # --------------------------------------------------------

    if df.shape[0] != 968:

        errors.append(
            f"Expected 968 rows, found {df.shape[0]}"
        )

    if df.shape[1] != 131:

        errors.append(
            f"Expected 131 columns, found {df.shape[1]}"
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [

        TARGET_COLUMN,
        CYCLE_FEATURE,
        "ToolIndex",

        *LEAKAGE_COLUMNS,

        *PROCESS_CONTEXT_COLUMNS
    ]

    for column in required_columns:

        if column not in df.columns:

            errors.append(
                f"Missing required column: {column}"
            )

    # --------------------------------------------------------
    # Sensor features
    # --------------------------------------------------------

    sensor_columns = detect_sensor_features(df)

    if len(sensor_columns) != 120:

        errors.append(
            f"Expected 120 sensor features, "
            f"found {len(sensor_columns)}"
        )

    # --------------------------------------------------------
    # Missing data
    # --------------------------------------------------------

    missing_values = int(
        df.isna().sum().sum()
    )

    if missing_values > 0:

        errors.append(
            f"Dataset contains "
            f"{missing_values} missing values"
        )

    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if duplicate_rows > 0:

        errors.append(
            f"Dataset contains "
            f"{duplicate_rows} duplicate rows"
        )

    # --------------------------------------------------------
    # Target validation
    # --------------------------------------------------------

    if TARGET_COLUMN in df.columns:

        target = df[TARGET_COLUMN]

        if not pd.api.types.is_numeric_dtype(target):

            errors.append(
                "Target column is not numeric"
            )

        else:

            if target.min() < 0 or target.max() > 1:

                errors.append(
                    "Target values are outside [0, 1]"
                )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        f"\nRows             : {df.shape[0]}"
    )

    print(
        f"Columns          : {df.shape[1]}"
    )

    print(
        f"Sensor features  : {len(sensor_columns)}"
    )

    print(
        f"Missing values   : {missing_values}"
    )

    print(
        f"Duplicate rows   : {duplicate_rows}"
    )

    if "ToolIndex" in df.columns:

        print(
            f"Unique tools     : "
            f"{df['ToolIndex'].nunique()}"
        )

    if errors:

        print("\nVALIDATION FAILED")

        for error in errors:

            print(f"  ✗ {error}")

        raise ValueError(
            "Dataset validation failed."
        )

    print("\n✓ Dataset validation passed.")

    return sensor_columns


# ============================================================
# CREATE FEATURE MANIFEST
# ============================================================

def create_feature_manifest(
    sensor_columns
):

    # --------------------------------------------------------
    # Experiment A
    #
    # Pure sensor + manufacturing context.
    #
    # Does NOT include NumberOfCycle.
    # --------------------------------------------------------

    feature_set_a = (
        sensor_columns
        +
        PROCESS_CONTEXT_COLUMNS
    )

    # --------------------------------------------------------
    # Experiment B
    #
    # Sensor + manufacturing context
    # + tool cycle age.
    # --------------------------------------------------------

    feature_set_b = (
        sensor_columns
        +
        PROCESS_CONTEXT_COLUMNS
        +
        [CYCLE_FEATURE]
    )

    manifest = {

        "target_column":
            TARGET_COLUMN,

        "sensor_features":
            sensor_columns,

        "process_context_features":
            PROCESS_CONTEXT_COLUMNS,

        "cycle_feature":
            CYCLE_FEATURE,

        "identifier_columns":
            IDENTIFIER_COLUMNS,

        "leakage_columns":
            LEAKAGE_COLUMNS,

        "feature_set_A_sensor_context":
            feature_set_a,

        "feature_set_B_sensor_context_cycle":
            feature_set_b,

        "counts": {

            "sensor_features":
                len(sensor_columns),

            "process_context_features":
                len(PROCESS_CONTEXT_COLUMNS),

            "feature_set_A":
                len(feature_set_a),

            "feature_set_B":
                len(feature_set_b)
        }
    }

    return manifest


# ============================================================
# GROUP-AWARE TRAIN TEST SPLIT
# ============================================================

def create_group_split(df):

    print("\n" + "=" * 75)
    print("GROUP-AWARE TRAIN / TEST SPLIT")
    print("=" * 75)

    groups = df["ToolIndex"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            df,
            groups=groups
        )
    )

    train_df = (
        df.iloc[train_indices]
        .copy()
    )

    test_df = (
        df.iloc[test_indices]
        .copy()
    )

    # Sort only for readability
    sort_columns = [
        "ToolIndex",
        "NumberOfCycle"
    ]

    train_df = (
        train_df
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    test_df = (
        test_df
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    train_tools = sorted(
        train_df["ToolIndex"]
        .unique()
        .tolist()
    )

    test_tools = sorted(
        test_df["ToolIndex"]
        .unique()
        .tolist()
    )

    overlap = set(train_tools).intersection(
        set(test_tools)
    )

    print(
        f"\nTraining rows : {len(train_df)}"
    )

    print(
        f"Testing rows  : {len(test_df)}"
    )

    print(
        f"\nTraining tools ({len(train_tools)}):"
    )

    print(train_tools)

    print(
        f"\nTesting tools ({len(test_tools)}):"
    )

    print(test_tools)

    if overlap:

        raise ValueError(
            f"Tool leakage detected: {overlap}"
        )

    print(
        "\n✓ No cutting tool appears in "
        "both train and test datasets."
    )

    return (
        train_df,
        test_df,
        train_tools,
        test_tools
    )


# ============================================================
# CHECK FEATURE LEAKAGE
# ============================================================

def validate_feature_manifest(
    manifest
):

    print("\n" + "=" * 75)
    print("FEATURE LEAKAGE CHECK")
    print("=" * 75)

    target = manifest["target_column"]

    leakage = set(
        manifest["leakage_columns"]
    )

    for feature_set_name in [

        "feature_set_A_sensor_context",

        "feature_set_B_sensor_context_cycle"
    ]:

        features = set(
            manifest[feature_set_name]
        )

        if target in features:

            raise ValueError(
                f"{target} appears in "
                f"{feature_set_name}"
            )

        leakage_overlap = (
            features.intersection(leakage)
        )

        if leakage_overlap:

            raise ValueError(
                f"Leakage columns found in "
                f"{feature_set_name}: "
                f"{leakage_overlap}"
            )

        print(
            f"✓ {feature_set_name}: "
            f"no target leakage detected."
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    df,
    train_df,
    test_df,
    manifest,
    train_tools,
    test_tools
):

    print("\n" + "=" * 75)
    print("SAVING PHASE 3 OUTPUTS")
    print("=" * 75)

    clean_file = (
        PROCESSED_DATA_DIR /
        "milling_clean.csv"
    )

    train_file = (
        PROCESSED_DATA_DIR /
        "train.csv"
    )

    test_file = (
        PROCESSED_DATA_DIR /
        "test.csv"
    )

    manifest_file = (
        REPORT_DIR /
        "phase3_feature_manifest.json"
    )

    split_report_file = (
        REPORT_DIR /
        "phase3_split_summary.txt"
    )

    df.to_csv(
        clean_file,
        index=False
    )

    train_df.to_csv(
        train_file,
        index=False
    )

    test_df.to_csv(
        test_file,
        index=False
    )

    with open(
        manifest_file,
        "w"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    with open(
        split_report_file,
        "w"
    ) as file:

        file.write(
            "AI MANUFACTURING DIGITAL TWIN\n"
        )

        file.write(
            "PHASE 3 SPLIT SUMMARY\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(
            f"Full dataset rows: "
            f"{len(df)}\n"
        )

        file.write(
            f"Training rows: "
            f"{len(train_df)}\n"
        )

        file.write(
            f"Testing rows: "
            f"{len(test_df)}\n\n"
        )

        file.write(
            f"Training tools "
            f"({len(train_tools)}):\n"
        )

        file.write(
            f"{train_tools}\n\n"
        )

        file.write(
            f"Testing tools "
            f"({len(test_tools)}):\n"
        )

        file.write(
            f"{test_tools}\n\n"
        )

        file.write(
            "Split strategy:\n"
        )

        file.write(
            "GroupShuffleSplit by ToolIndex\n"
        )

        file.write(
            f"test_size={TEST_SIZE}\n"
        )

        file.write(
            f"random_state={RANDOM_STATE}\n\n"
        )

        file.write(
            "Important:\n"
        )

        file.write(
            "No tool appears in both "
            "training and testing data.\n"
        )

        file.write(
            "CycleToFailure is excluded "
            "from model feature sets "
            "because it causes target leakage.\n"
        )

    print(f"\n✓ {clean_file}")
    print(f"✓ {train_file}")
    print(f"✓ {test_file}")
    print(f"✓ {manifest_file}")
    print(f"✓ {split_report_file}")


# ============================================================
# SUMMARY
# ============================================================

def print_feature_summary(
    manifest
):

    print("\n" + "=" * 75)
    print("FEATURE ARCHITECTURE")
    print("=" * 75)

    counts = manifest["counts"]

    print(
        f"\nSensor features          : "
        f"{counts['sensor_features']}"
    )

    print(
        f"Process context features : "
        f"{counts['process_context_features']}"
    )

    print(
        f"\nExperiment A features    : "
        f"{counts['feature_set_A']}"
    )

    print(
        "  Sensors + manufacturing context"
    )

    print(
        f"\nExperiment B features    : "
        f"{counts['feature_set_B']}"
    )

    print(
        "  Sensors + manufacturing context "
        "+ NumberOfCycle"
    )

    print(
        f"\nTarget:"
        f"\n  {TARGET_COLUMN}"
    )

    print(
        "\nExcluded from prediction:"
    )

    for column in (
        IDENTIFIER_COLUMNS
        +
        LEAKAGE_COLUMNS
    ):

        print(f"  - {column}")


# ============================================================
# MAIN
# ============================================================

def main():

    raw_df = load_raw_dataset()

    df = fix_embedded_header(
        raw_df
    )

    df = standardize_column_names(
        df
    )

    df = convert_numeric_columns(
        df
    )

    sensor_columns = validate_dataset(
        df
    )

    manifest = create_feature_manifest(
        sensor_columns
    )

    validate_feature_manifest(
        manifest
    )

    print_feature_summary(
        manifest
    )

    (
        train_df,
        test_df,
        train_tools,
        test_tools

    ) = create_group_split(df)

    save_outputs(
        df=df,
        train_df=train_df,
        test_df=test_df,
        manifest=manifest,
        train_tools=train_tools,
        test_tools=test_tools
    )

    print("\n" + "=" * 75)
    print("PHASE 3 DATA PREPARATION COMPLETE")
    print("=" * 75)

    print(
        "\nNext:"
        "\nExploratory Data Analysis "
        "and tool-degradation visualization."
    )


if __name__ == "__main__":
    main()