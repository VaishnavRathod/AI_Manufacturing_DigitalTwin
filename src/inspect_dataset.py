from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_FILE = RAW_DATA_DIR / "FeatureAndMetadata_Milling.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    if not DATA_FILE.exists():

        print("\nERROR: Dataset not found.")
        print(f"Expected location:\n{DATA_FILE}")

        raise FileNotFoundError(DATA_FILE)

    print("=" * 70)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 2 — DATASET INSPECTION")
    print("=" * 70)

    print(f"\nDataset:\n{DATA_FILE}")

    file_size_mb = DATA_FILE.stat().st_size / (1024 * 1024)

    print(f"\nFile size: {file_size_mb:.2f} MB")

    # --------------------------------------------------------
    # STEP 1: Load raw CSV
    # --------------------------------------------------------

    raw_df = pd.read_csv(
        DATA_FILE,
        sep=None,
        engine="python",
        dtype=str
    )

    print("\nRaw CSV shape:")
    print(f"Rows    : {raw_df.shape[0]}")
    print(f"Columns : {raw_df.shape[1]}")

    # --------------------------------------------------------
    # STEP 2: Detect generic headers
    # --------------------------------------------------------

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

    header_keywords = [
        "TollIndex",
        "CycleToFailure",
        "CycleToFailureNormalized",
        "MillingToolType"
    ]

    real_header_detected = any(
        keyword in first_row_values
        for keyword in header_keywords
    )

    # --------------------------------------------------------
    # STEP 3: Promote first row to column names
    # --------------------------------------------------------

    if generic_headers and real_header_detected:

        print("\nGeneric Column1...Column131 headers detected.")

        print(
            "✓ First data row appears to contain "
            "the actual feature names."
        )

        df = raw_df.iloc[1:].copy()

        df.columns = first_row_values

        df = df.reset_index(drop=True)

        print(
            "✓ First row promoted to dataframe header."
        )

    else:

        print(
            "\nNo embedded header row automatically detected."
        )

        print("\nFirst row preview:")

        print(first_row_values[:15])

        df = raw_df.copy()

    # --------------------------------------------------------
    # STEP 4: Clean column names
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # STEP 5: Convert numeric columns
    # --------------------------------------------------------

    print("\nConverting numeric columns...")

    converted_count = 0

    for column in df.columns:

        original = df[column].copy()

        # Remove whitespace
        cleaned = (
            original
            .astype(str)
            .str.strip()
        )

        # Support European decimal comma if present
        cleaned_decimal = cleaned.str.replace(
            ",",
            ".",
            regex=False
        )

        numeric = pd.to_numeric(
            cleaned_decimal,
            errors="coerce"
        )

        valid_ratio = numeric.notna().mean()

        # Convert when nearly the entire column is numeric
        if valid_ratio >= 0.95:

            df[column] = numeric

            converted_count += 1

        else:

            df[column] = cleaned

    print(
        f"✓ Converted {converted_count} columns "
        "to numeric datatype."
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\nCorrected dataset shape:")

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def basic_information(df):

    print("\n" + "=" * 70)
    print("1. DATASET DIMENSIONS")
    print("=" * 70)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nExpected from publication:")
    print("Rows    : 968")
    print("Columns : 131")

    if df.shape == (968, 131):
        print("\n✓ Dataset dimensions match the published processed dataset.")
    else:
        print("\nWARNING: Dimensions differ from published 968 × 131 dataset.")


# ============================================================
# COLUMN INFORMATION
# ============================================================

def column_information(df):

    print("\n" + "=" * 70)
    print("2. COLUMN INFORMATION")
    print("=" * 70)

    print("\nFIRST 20 COLUMNS:")

    for i, column in enumerate(df.columns[:20], start=1):
        print(f"{i:3d}. {column}")

    print("\nLAST 20 COLUMNS:")

    start = max(0, len(df.columns) - 20)

    for i, column in enumerate(df.columns[start:], start=start + 1):
        print(f"{i:3d}. {column}")


# ============================================================
# DATA TYPES
# ============================================================

def datatype_analysis(df):

    print("\n" + "=" * 70)
    print("3. DATA TYPES")
    print("=" * 70)

    print(df.dtypes.value_counts())

    numeric_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns

    print(f"\nNumeric columns     : {len(numeric_columns)}")
    print(f"Non-numeric columns : {len(categorical_columns)}")

    if len(categorical_columns) > 0:

        print("\nNon-numeric columns:")

        for column in categorical_columns:
            print(f"  - {column}")


# ============================================================
# MISSING VALUES
# ============================================================

def missing_value_analysis(df):

    print("\n" + "=" * 70)
    print("4. MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()

    total_missing = missing.sum()

    print(f"\nTotal missing values: {total_missing}")

    missing_columns = missing[missing > 0]

    if len(missing_columns) == 0:

        print("✓ No missing values detected.")

    else:

        print("\nColumns containing missing values:")

        for column, count in missing_columns.items():

            percentage = 100 * count / len(df)

            print(
                f"{column:40s} "
                f"{count:5d} "
                f"({percentage:.2f}%)"
            )


# ============================================================
# DUPLICATES
# ============================================================

def duplicate_analysis(df):

    print("\n" + "=" * 70)
    print("5. DUPLICATE ROWS")
    print("=" * 70)

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate rows: {duplicates}")

    if duplicates == 0:
        print("✓ No duplicate rows detected.")


# ============================================================
# CONSTANT FEATURES
# ============================================================

def constant_feature_analysis(df):

    print("\n" + "=" * 70)
    print("6. CONSTANT FEATURES")
    print("=" * 70)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    print(f"\nNumber of constant columns: {len(constant_columns)}")

    if constant_columns:

        for column in constant_columns:
            print(f"  - {column}")

    else:
        print("✓ No constant columns detected.")


# ============================================================
# METADATA ANALYSIS
# ============================================================

def metadata_analysis(df):

    print("\n" + "=" * 70)
    print("7. MANUFACTURING METADATA")
    print("=" * 70)

    possible_metadata = [

        "ExperimentIndex",
        "SampleIndex",
        "TollIndex",
        "MillingToolType",
        "ADOC",
        "RDOC",
        "HardnessMean",
        "CycleToFailure",
        "CycleToFailureNormalized",
        "ToolHolderLength",
        "ToolRotation",
        "FeedRate",
        "ToolDiameter",

    ]

    detected = []

    for column in possible_metadata:

        if column in df.columns:

            detected.append(column)

            print(
                f"{column:30s} "
                f"unique={df[column].nunique()}"
            )

    if not detected:
        print("Expected metadata column names were not detected.")

    return detected


# ============================================================
# TOOL INFORMATION
# ============================================================

def tool_analysis(df):

    print("\n" + "=" * 70)
    print("8. CUTTING TOOL INFORMATION")
    print("=" * 70)

    if "TollIndex" in df.columns:

        print(
            f"\nNumber of unique tools: "
            f"{df['TollIndex'].nunique()}"
        )

        print("\nCycles per tool:")

        counts = (
            df.groupby("TollIndex")
            .size()
            .sort_index()
        )

        print(counts)

    else:
        print("\nTollIndex column not found.")


# ============================================================
# TARGET ANALYSIS
# ============================================================

def target_analysis(df):

    print("\n" + "=" * 70)
    print("9. TARGET VARIABLE")
    print("=" * 70)

    target = "CycleToFailureNormalized"

    if target not in df.columns:

        print(
            f"\nWARNING: Expected target '{target}' "
            "was not found."
        )

        return

    y = df[target]

    print(f"\nTarget: {target}")

    print("\nStatistics:")

    print(y.describe())

    print("\nExpected interpretation:")

    print("1.0 → tool is at/near beginning of life")
    print("0.0 → tool has reached failure cycle")


# ============================================================
# DATA LEAKAGE WARNING
# ============================================================

def leakage_analysis(df):

    print("\n" + "=" * 70)
    print("10. POSSIBLE DATA LEAKAGE")
    print("=" * 70)

    leakage_candidates = [

        "CycleToFailure",
        "CycleToFailureNormalized"

    ]

    print(
        "\nThe following variables must NOT both be used "
        "as predictors when predicting tool remaining life:"
    )

    for column in leakage_candidates:

        if column in df.columns:
            print(f"  ⚠ {column}")

    print(
        "\nCycleToFailure directly contains future failure "
        "information and would create target leakage."
    )


# ============================================================
# CREATE DATA DICTIONARY
# ============================================================

def create_data_dictionary(df):

    rows = []

    for column in df.columns:

        series = df[column]

        row = {

            "column": column,

            "dtype": str(series.dtype),

            "missing_count": int(
                series.isnull().sum()
            ),

            "missing_percent": round(
                100 * series.isnull().mean(),
                4
            ),

            "unique_values": int(
                series.nunique(dropna=True)
            ),

            "is_constant": (
                series.nunique(dropna=False) <= 1
            ),

        }

        if pd.api.types.is_numeric_dtype(series):

            row["min"] = series.min()

            row["max"] = series.max()

            row["mean"] = series.mean()

            row["std"] = series.std()

        rows.append(row)

    dictionary = pd.DataFrame(rows)

    output_file = (
        REPORT_DIR /
        "phase2_data_dictionary.csv"
    )

    dictionary.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("11. DATA DICTIONARY")
    print("=" * 70)

    print(
        f"\nSaved to:\n{output_file}"
    )


# ============================================================
# SAVE SUMMARY REPORT
# ============================================================

def save_summary(df):

    output_file = (
        REPORT_DIR /
        "phase2_dataset_summary.txt"
    )

    with open(output_file, "w") as f:

        f.write(
            "AI MANUFACTURING DIGITAL TWIN\n"
        )

        f.write(
            "PHASE 2 DATASET SUMMARY\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Rows: {df.shape[0]}\n"
        )

        f.write(
            f"Columns: {df.shape[1]}\n"
        )

        f.write(
            f"Missing values: "
            f"{df.isnull().sum().sum()}\n"
        )

        f.write(
            f"Duplicate rows: "
            f"{df.duplicated().sum()}\n"
        )

        if "TollIndex" in df.columns:

            f.write(
                f"Unique tools: "
                f"{df['TollIndex'].nunique()}\n"
            )

        f.write("\nColumns:\n")

        for column in df.columns:

            f.write(
                f"- {column}\n"
            )

    print(
        f"\nSummary report saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    basic_information(df)

    column_information(df)

    datatype_analysis(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    constant_feature_analysis(df)

    metadata_analysis(df)

    tool_analysis(df)

    target_analysis(df)

    leakage_analysis(df)

    create_data_dictionary(df)

    save_summary(df)

    print("\n" + "=" * 70)
    print("PHASE 2 DATASET INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()