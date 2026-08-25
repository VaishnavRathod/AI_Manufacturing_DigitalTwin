from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = (
    PROCESSED_DIR /
    "train.csv"
)

TEST_FILE = (
    PROCESSED_DIR /
    "test.csv"
)

FEATURE_FILE = (
    REPORT_DIR /
    "phase5_feature_sets.json"
)

PARAMETER_FILE = (
    REPORT_DIR /
    "phase7_xgboost_best_params.json"
)

CV_RESULTS_FILE = (
    REPORT_DIR /
    "phase7_xgboost_summary.csv"
)

FINAL_RESULTS_FILE = (
    REPORT_DIR /
    "phase8_final_test_results.csv"
)

PER_TOOL_RESULTS_FILE = (
    REPORT_DIR /
    "phase8_per_tool_results.csv"
)

PREDICTIONS_FILE = (
    REPORT_DIR /
    "phase8_test_predictions.csv"
)

SUMMARY_FILE = (
    REPORT_DIR /
    "phase8_final_evaluation_summary.txt"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "CycleToFailureNormalized"

GROUP_COLUMN = "ToolIndex"

RANDOM_STATE = 42


# ------------------------------------------------------------
# IMPORTANT
#
# These candidates were selected BEFORE
# examining the locked test-set performance.
# ------------------------------------------------------------

FINAL_CANDIDATES = [

    "D_age_aware",

    "B_current_based"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 78)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 8 — FINAL LOCKED TEST EVALUATION")
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

        feature_manifest = json.load(
            file
        )

    with open(
        PARAMETER_FILE,
        "r"
    ) as file:

        best_parameters = json.load(
            file
        )

    cv_results = pd.read_csv(
        CV_RESULTS_FILE
    )

    print(
        f"\nTraining rows : {len(train)}"
    )

    print(
        f"Testing rows  : {len(test)}"
    )

    print(
        "\nTraining tools:"
    )

    print(
        sorted(
            train[
                GROUP_COLUMN
            ]
            .unique()
            .tolist()
        )
    )

    print(
        "\nLOCKED TEST TOOLS:"
    )

    print(
        sorted(
            test[
                GROUP_COLUMN
            ]
            .unique()
            .tolist()
        )
    )

    return (
        train,
        test,
        feature_manifest,
        best_parameters,
        cv_results
    )


# ============================================================
# VERIFY TEST ISOLATION
# ============================================================

def verify_test_isolation(
    train,
    test
):

    train_tools = set(
        train[
            GROUP_COLUMN
        ].unique()
    )

    test_tools = set(
        test[
            GROUP_COLUMN
        ].unique()
    )

    overlap = (
        train_tools
        .intersection(
            test_tools
        )
    )

    if overlap:

        raise ValueError(
            f"Tool leakage detected: "
            f"{overlap}"
        )

    print(
        "\n✓ Train/test tool isolation verified."
    )

    print(
        "✓ Final test contains completely "
        "unseen cutting tools."
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    predictions
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    }


# ============================================================
# CREATE MODEL
# ============================================================

def build_model(
    parameters
):

    return XGBRegressor(

        objective=
            "reg:squarederror",

        eval_metric=
            "rmse",

        tree_method=
            "hist",

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        **parameters
    )


# ============================================================
# EVALUATE ONE FINAL MODEL
# ============================================================

def evaluate_candidate(
    candidate,
    features,
    parameters,
    train,
    test
):

    print("\n" + "=" * 78)

    print(
        f"FINAL CANDIDATE: {candidate}"
    )

    print("=" * 78)

    print(
        f"\nNumber of features: "
        f"{len(features)}"
    )

    X_train = (
        train[
            features
        ].copy()
    )

    y_train = (
        train[
            TARGET
        ].copy()
    )

    X_test = (
        test[
            features
        ].copy()
    )

    y_test = (
        test[
            TARGET
        ].copy()
    )

    # --------------------------------------------------------
    # Train once using all 11 development tools
    # --------------------------------------------------------

    model = build_model(
        parameters
    )

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Final locked-test prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # IMPORTANT:
    #
    # Metrics use RAW model predictions.
    # We do not clip to [0, 1] because that
    # could artificially improve test scores.
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    below_zero = int(
        np.sum(
            predictions < 0
        )
    )

    above_one = int(
        np.sum(
            predictions > 1
        )
    )

    print(
        "\nFINAL TEST PERFORMANCE"
    )

    print(
        f"MAE  : "
        f"{metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : "
        f"{metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : "
        f"{metrics['R2']:.4f}"
    )

    print(
        "\nPrediction range:"
    )

    print(
        f"Min prediction : "
        f"{predictions.min():.4f}"
    )

    print(
        f"Max prediction : "
        f"{predictions.max():.4f}"
    )

    print(
        f"Predictions < 0 : "
        f"{below_zero}"
    )

    print(
        f"Predictions > 1 : "
        f"{above_one}"
    )

    # --------------------------------------------------------
    # Save trained model
    # --------------------------------------------------------

    model_file = (
        MODEL_DIR /
        f"{candidate}_xgboost.joblib"
    )

    joblib.dump(
        {
            "model":
                model,

            "features":
                features,

            "target":
                TARGET,

            "experiment":
                candidate
        },
        model_file
    )

    print(
        f"\n✓ Model saved:\n"
        f"{model_file}"
    )

    # --------------------------------------------------------
    # Prediction dataframe
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({

        "experiment":
            candidate,

        "ToolIndex":
            test[
                GROUP_COLUMN
            ].values,

        "NumberOfCycle":
            test[
                "NumberOfCycle"
            ].values,

        "ActualHealth":
            y_test.values,

        "PredictedHealth":
            predictions,

        "AbsoluteError":
            np.abs(
                y_test.values
                -
                predictions
            )
    })

    return (
        metrics,
        predictions_df,
        model
    )


# ============================================================
# PER-TOOL PERFORMANCE
# ============================================================

def calculate_per_tool_results(
    prediction_df
):

    rows = []

    experiment = (
        prediction_df[
            "experiment"
        ].iloc[0]
    )

    for tool_id, group in (
        prediction_df.groupby(
            "ToolIndex"
        )
    ):

        metrics = calculate_metrics(

            group[
                "ActualHealth"
            ],

            group[
                "PredictedHealth"
            ]
        )

        rows.append({

            "experiment":
                experiment,

            "ToolIndex":
                tool_id,

            "Samples":
                len(group),

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"]
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# RUN FINAL EVALUATION
# ============================================================

def run_final_evaluation(
    train,
    test,
    feature_manifest,
    best_parameters,
    cv_results
):

    experiments = (
        feature_manifest[
            "experiments"
        ]
    )

    all_results = []

    all_predictions = []

    all_per_tool = []

    for candidate in (
        FINAL_CANDIDATES
    ):

        if candidate not in experiments:

            raise ValueError(
                f"Missing feature set: "
                f"{candidate}"
            )

        if candidate not in best_parameters:

            raise ValueError(
                f"Missing parameters: "
                f"{candidate}"
            )

        features = (
            experiments[
                candidate
            ]
        )

        parameters = (
            best_parameters[
                candidate
            ]
        )

        (
            test_metrics,
            predictions,
            model

        ) = evaluate_candidate(

            candidate=
                candidate,

            features=
                features,

            parameters=
                parameters,

            train=
                train,

            test=
                test
        )

        # ----------------------------------------------------
        # Retrieve Phase 7 CV result
        # ----------------------------------------------------

        cv_row = (
            cv_results[
                cv_results[
                    "experiment"
                ]
                ==
                candidate
            ]
        )

        if cv_row.empty:

            cv_mae = np.nan
            cv_rmse = np.nan
            cv_r2 = np.nan

        else:

            cv_row = (
                cv_row.iloc[0]
            )

            cv_mae = (
                cv_row["CV_MAE"]
            )

            cv_rmse = (
                cv_row["CV_RMSE"]
            )

            cv_r2 = (
                cv_row["CV_R2"]
            )

        all_results.append({

            "experiment":
                candidate,

            "number_features":
                len(features),

            "CV_MAE":
                cv_mae,

            "CV_RMSE":
                cv_rmse,

            "CV_R2":
                cv_r2,

            "Test_MAE":
                test_metrics["MAE"],

            "Test_RMSE":
                test_metrics["RMSE"],

            "Test_R2":
                test_metrics["R2"],

            "RMSE_Generalization_Gap":
                (
                    test_metrics[
                        "RMSE"
                    ]
                    -
                    cv_rmse
                ),

            "R2_Generalization_Gap":
                (
                    test_metrics[
                        "R2"
                    ]
                    -
                    cv_r2
                )
        })

        all_predictions.append(
            predictions
        )

        per_tool = (
            calculate_per_tool_results(
                predictions
            )
        )

        all_per_tool.append(
            per_tool
        )

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    per_tool_df = pd.concat(
        all_per_tool,
        ignore_index=True
    )

    return (
        results_df,
        predictions_df,
        per_tool_df
    )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_final_results(
    results_df,
    per_tool_df
):

    print("\n" + "=" * 78)
    print("FINAL GENERALIZATION RESULTS")
    print("=" * 78)

    display_columns = [

        "experiment",
        "number_features",
        "CV_MAE",
        "Test_MAE",
        "CV_RMSE",
        "Test_RMSE",
        "CV_R2",
        "Test_R2"
    ]

    display = (
        results_df[
            display_columns
        ]
        .copy()
    )

    numeric_columns = [

        "CV_MAE",
        "Test_MAE",
        "CV_RMSE",
        "Test_RMSE",
        "CV_R2",
        "Test_R2"
    ]

    display[
        numeric_columns
    ] = (
        display[
            numeric_columns
        ]
        .round(4)
    )

    print(
        "\n"
        +
        display.to_string(
            index=False
        )
    )

    print(
        "\n\nPER-TOOL TEST PERFORMANCE"
    )

    per_tool_display = (
        per_tool_df.copy()
    )

    for column in [
        "MAE",
        "RMSE",
        "R2"
    ]:

        per_tool_display[
            column
        ] = (
            per_tool_display[
                column
            ]
            .round(4)
        )

    print(
        "\n"
        +
        per_tool_display
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    predictions_df,
    per_tool_df
):

    results_df.to_csv(
        FINAL_RESULTS_FILE,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    per_tool_df.to_csv(
        PER_TOOL_RESULTS_FILE,
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
            "PHASE 8 FINAL TEST EVALUATION\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            "Final candidates were selected "
            "before locked-test evaluation.\n\n"
        )

        file.write(
            "Locked test tools:\n"
        )

        file.write(
            "2, 11, 102\n\n"
        )

        for _, row in (
            results_df.iterrows()
        ):

            file.write(
                f"{row['experiment']}\n"
            )

            file.write(
                f"Features: "
                f"{row['number_features']}\n"
            )

            file.write(
                f"CV MAE: "
                f"{row['CV_MAE']:.6f}\n"
            )

            file.write(
                f"Test MAE: "
                f"{row['Test_MAE']:.6f}\n"
            )

            file.write(
                f"CV RMSE: "
                f"{row['CV_RMSE']:.6f}\n"
            )

            file.write(
                f"Test RMSE: "
                f"{row['Test_RMSE']:.6f}\n"
            )

            file.write(
                f"CV R2: "
                f"{row['CV_R2']:.6f}\n"
            )

            file.write(
                f"Test R2: "
                f"{row['Test_R2']:.6f}\n\n"
            )

    print(
        "\nSaved:"
    )

    print(
        f"✓ {FINAL_RESULTS_FILE}"
    )

    print(
        f"✓ {PER_TOOL_RESULTS_FILE}"
    )

    print(
        f"✓ {PREDICTIONS_FILE}"
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
        feature_manifest,
        best_parameters,
        cv_results

    ) = load_data()

    verify_test_isolation(
        train,
        test
    )

    (
        results_df,
        predictions_df,
        per_tool_df

    ) = run_final_evaluation(

        train=train,

        test=test,

        feature_manifest=
            feature_manifest,

        best_parameters=
            best_parameters,

        cv_results=
            cv_results
    )

    print_final_results(
        results_df,
        per_tool_df
    )

    save_results(
        results_df,
        predictions_df,
        per_tool_df
    )

    print("\n" + "=" * 78)
    print("PHASE 8 COMPLETE")
    print("=" * 78)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The locked final test set has now "
        "been evaluated."
    )

    print(
        "Do not tune model hyperparameters "
        "based on these test results."
    )


if __name__ == "__main__":
    main()