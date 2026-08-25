import pandas as pd

from src.decision_engine import (
    determine_state
)


def make_row(
    lifecycle,
    condition,
    anomaly=False,
    gap_state="CONSISTENT"
):

    return pd.Series({

        "LifecycleHealth":
            lifecycle,

        "ConditionHealth":
            condition,

        "IsAnomaly":
            anomaly,

        "HealthGap":
            condition - lifecycle,

        "HealthGapState":
            gap_state
    })


def test_normal_state():

    row = make_row(
        lifecycle=0.90,
        condition=0.85,
        anomaly=False
    )

    assert (
        determine_state(row)
        ==
        "NORMAL"
    )


def test_watch_state():

    row = make_row(
        lifecycle=0.70,
        condition=0.68,
        anomaly=False
    )

    assert (
        determine_state(row)
        ==
        "WATCH"
    )


def test_high_risk_state():

    row = make_row(
        lifecycle=0.40,
        condition=0.65,
        anomaly=False
    )

    assert (
        determine_state(row)
        ==
        "HIGH_RISK"
    )


def test_critical_low_health():

    row = make_row(
        lifecycle=0.20,
        condition=0.40,
        anomaly=False
    )

    assert (
        determine_state(row)
        ==
        "CRITICAL"
    )


def test_critical_sensor_fusion():

    row = make_row(
        lifecycle=0.37,
        condition=0.47,
        anomaly=True
    )

    assert (
        determine_state(row)
        ==
        "CRITICAL"
    )


def test_condition_worse_than_lifecycle():

    row = make_row(
        lifecycle=0.80,
        condition=0.60,
        anomaly=False,
        gap_state=
            "CONDITION_WORSE_THAN_LIFECYCLE"
    )

    assert (
        determine_state(row)
        ==
        "HIGH_RISK"
    )