from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_clean_dataset_exists():

    file = PROCESSED_DIR / "milling_clean.csv"

    assert file.exists()


def test_train_test_files_exist():

    assert (
        PROCESSED_DIR /
        "train.csv"
    ).exists()

    assert (
        PROCESSED_DIR /
        "test.csv"
    ).exists()


def test_dataset_shape():

    df = pd.read_csv(
        PROCESSED_DIR /
        "milling_clean.csv"
    )

    assert df.shape == (
        968,
        131
    )


def test_no_missing_values():

    df = pd.read_csv(
        PROCESSED_DIR /
        "milling_clean.csv"
    )

    assert (
        df.isna()
        .sum()
        .sum()
        ==
        0
    )


def test_no_tool_leakage():

    train = pd.read_csv(
        PROCESSED_DIR /
        "train.csv"
    )

    test = pd.read_csv(
        PROCESSED_DIR /
        "test.csv"
    )

    train_tools = set(
        train["ToolIndex"]
    )

    test_tools = set(
        test["ToolIndex"]
    )

    assert train_tools.isdisjoint(
        test_tools
    )


def test_target_range():

    df = pd.read_csv(
        PROCESSED_DIR /
        "milling_clean.csv"
    )

    target = (
        df[
            "CycleToFailureNormalized"
        ]
    )

    assert target.min() >= 0
    assert target.max() <= 1