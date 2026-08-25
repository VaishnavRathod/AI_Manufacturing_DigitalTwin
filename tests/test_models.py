from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"


def test_age_model_exists():

    file = (
        MODEL_DIR /
        "D_age_aware_xgboost.joblib"
    )

    assert file.exists()


def test_condition_model_exists():

    file = (
        MODEL_DIR /
        "B_current_based_xgboost.joblib"
    )

    assert file.exists()


def test_anomaly_model_exists():

    file = (
        MODEL_DIR /
        "isolation_forest_current.joblib"
    )

    assert file.exists()


def test_age_model_payload():

    payload = joblib.load(
        MODEL_DIR /
        "D_age_aware_xgboost.joblib"
    )

    assert "model" in payload
    assert "features" in payload
    assert "target" in payload

    assert (
        len(payload["features"])
        ==
        126
    )


def test_condition_model_payload():

    payload = joblib.load(
        MODEL_DIR /
        "B_current_based_xgboost.joblib"
    )

    assert (
        len(payload["features"])
        ==
        77
    )