from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = PROCESSED_DIR / "train.csv"

FEATURE_MANIFEST_FILE = (
    REPORT_DIR /
    "phase5_feature_sets.json"
)

RESULTS_FILE = (
    REPORT_DIR /
    "phase6_baseline_cv_results.csv"
)

FOLD_RESULTS_FILE = (
    REPORT_DIR /
    "phase6_baseline_fold_results.csv"
)

SUMMARY_FILE = (
    REPORT_DIR /
    "phase6_baseline_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "CycleToFailureNormalized"
GROUP_COLUMN = "ToolIndex"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 78)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 6 — BASELINE TOOL HEALTH MODELS")
    print("=" * 78)

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not FEATURE_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Feature manifest not found:\n"
            f"{FEATURE_MANIFEST_FILE}"
        )

    train = pd.read_csv(
        TRAIN_FILE
    )

    with open(
        FEATURE_MANIFEST_FILE,
        "r"
    ) as file:

        manifest = json.load(file)

    print(
        f"\nTraining rows    : {len(train)}"
    )

    print(
        f"Training columns : {train.shape[1]}"
    )

    print(
        f"Training tools   : "
        f"{train[GROUP_COLUMN].nunique()}"
    )

    print(
        "\nTools:"
    )

    print(
        sorted(
            train[GROUP_COLUMN]
            .unique()
            .tolist()
        )
    )

    return train, manifest


# ============================================================
# DEFINE MODELS
# ============================================================

def build_models():

    models = {

        # ----------------------------------------------------
        # Naive benchmark
        # ----------------------------------------------------

        "DummyMean": DummyRegressor(
            strategy="mean"
        ),

        # ----------------------------------------------------
        # Linear baseline
        #
        # Scaling is fitted independently
        # inside each training fold.
        # ----------------------------------------------------

        "Ridge": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "model",
                    Ridge(
                        alpha=1.0
                    )
                )
            ]
        ),

        # ----------------------------------------------------
        # Non-linear baseline
        # ----------------------------------------------------

        "RandomForest": (
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        ),
    }

    return models


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# GROUP-AWARE CROSS VALIDATION
# ============================================================

def evaluate_model_cv(
    train,
    features,
    model,
    experiment_name,
    model_name
):

    X = train[features].copy()

    y = train[TARGET].copy()

    groups = (
        train[GROUP_COLUMN]
        .copy()
    )

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_results = []

    all_y_true = []
    all_y_pred = []

    print(
        f"\n{'-' * 78}"
    )

    print(
        f"{experiment_name} | "
        f"{model_name}"
    )

    print(
        f"Features: {len(features)}"
    )

    start_time = time.time()

    for fold_number, (
        train_indices,
        validation_indices

    ) in enumerate(
        group_kfold.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        X_train_fold = (
            X.iloc[train_indices]
        )

        X_validation_fold = (
            X.iloc[validation_indices]
        )

        y_train_fold = (
            y.iloc[train_indices]
        )

        y_validation_fold = (
            y.iloc[validation_indices]
        )

        train_groups = (
            groups.iloc[train_indices]
        )

        validation_groups = (
            groups.iloc[validation_indices]
        )

        training_tools = sorted(
            train_groups
            .unique()
            .tolist()
        )

        validation_tools = sorted(
            validation_groups
            .unique()
            .tolist()
        )

        # ----------------------------------------------------
        # Verify no tool leakage
        # ----------------------------------------------------

        overlap = (
            set(training_tools)
            .intersection(
                set(validation_tools)
            )
        )

        if overlap:

            raise ValueError(
                f"Tool leakage in fold "
                f"{fold_number}: {overlap}"
            )

        # ----------------------------------------------------
        # Clone model
        # ----------------------------------------------------

        fold_model = clone(
            model
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        fold_model.fit(
            X_train_fold,
            y_train_fold
        )

        # ----------------------------------------------------
        # Predict unseen tools
        # ----------------------------------------------------

        predictions = (
            fold_model.predict(
                X_validation_fold
            )
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            y_validation_fold,
            predictions
        )

        fold_result = {

            "experiment":
                experiment_name,

            "model":
                model_name,

            "fold":
                fold_number,

            "number_features":
                len(features),

            "training_rows":
                len(train_indices),

            "validation_rows":
                len(validation_indices),

            "training_tools":
                str(training_tools),

            "validation_tools":
                str(validation_tools),

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"]
        }

        fold_results.append(
            fold_result
        )

        all_y_true.extend(
            y_validation_fold.tolist()
        )

        all_y_pred.extend(
            predictions.tolist()
        )

        print(
            f"\nFold {fold_number}"
        )

        print(
            f"Validation tools : "
            f"{validation_tools}"
        )

        print(
            f"MAE              : "
            f"{metrics['MAE']:.4f}"
        )

        print(
            f"RMSE             : "
            f"{metrics['RMSE']:.4f}"
        )

        print(
            f"R²               : "
            f"{metrics['R2']:.4f}"
        )

    # ========================================================
    # Aggregate out-of-fold performance
    # ========================================================

    overall_metrics = (
        calculate_metrics(
            np.array(all_y_true),
            np.array(all_y_pred)
        )
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    fold_df = pd.DataFrame(
        fold_results
    )

    summary = {

        "experiment":
            experiment_name,

        "model":
            model_name,

        "number_features":
            len(features),

        "CV_MAE":
            overall_metrics["MAE"],

        "CV_RMSE":
            overall_metrics["RMSE"],

        "CV_R2":
            overall_metrics["R2"],

        "Fold_MAE_Mean":
            fold_df["MAE"].mean(),

        "Fold_MAE_Std":
            fold_df["MAE"].std(),

        "Fold_RMSE_Mean":
            fold_df["RMSE"].mean(),

        "Fold_RMSE_Std":
            fold_df["RMSE"].std(),

        "Fold_R2_Mean":
            fold_df["R2"].mean(),

        "Fold_R2_Std":
            fold_df["R2"].std(),

        "RuntimeSeconds":
            elapsed
    }

    print(
        "\nOverall out-of-fold performance"
    )

    print(
        f"MAE  : "
        f"{overall_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE : "
        f"{overall_metrics['RMSE']:.4f}"
    )

    print(
        f"R²   : "
        f"{overall_metrics['R2']:.4f}"
    )

    return summary, fold_results


# ============================================================
# RUN ALL BASELINE EXPERIMENTS
# ============================================================

def run_experiments(
    train,
    manifest
):

    experiments = (
        manifest[
            "experiments"
        ]
    )

    models = build_models()

    all_summaries = []

    all_fold_results = []

    print("\n" + "=" * 78)
    print("RUNNING GROUP-AWARE BASELINE EXPERIMENTS")
    print("=" * 78)

    for experiment_name, features in (
        experiments.items()
    ):

        for model_name, model in (
            models.items()
        ):

            (
                summary,
                fold_results

            ) = evaluate_model_cv(
                train=train,
                features=features,
                model=model,
                experiment_name=experiment_name,
                model_name=model_name
            )

            all_summaries.append(
                summary
            )

            all_fold_results.extend(
                fold_results
            )

    results_df = pd.DataFrame(
        all_summaries
    )

    fold_results_df = pd.DataFrame(
        all_fold_results
    )

    return (
        results_df,
        fold_results_df
    )


# ============================================================
# RANK MODELS
# ============================================================

def rank_models(
    results_df
):

    ranked = (
        results_df
        .sort_values(
            by=[
                "CV_RMSE",
                "CV_MAE"
            ],
            ascending=[
                True,
                True
            ]
        )
        .reset_index(
            drop=True
        )
    )

    ranked.insert(
        0,
        "Rank",
        range(
            1,
            len(ranked) + 1
        )
    )

    return ranked


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    ranked
):

    print("\n" + "=" * 78)
    print("PHASE 6 BASELINE MODEL RANKING")
    print("=" * 78)

    display_columns = [

        "Rank",
        "experiment",
        "model",
        "number_features",
        "CV_MAE",
        "CV_RMSE",
        "CV_R2",
        "Fold_RMSE_Std"
    ]

    display = (
        ranked[
            display_columns
        ]
        .copy()
    )

    numeric_columns = [

        "CV_MAE",
        "CV_RMSE",
        "CV_R2",
        "Fold_RMSE_Std"
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


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    ranked,
    fold_results_df
):

    ranked.to_csv(
        RESULTS_FILE,
        index=False
    )

    fold_results_df.to_csv(
        FOLD_RESULTS_FILE,
        index=False
    )

    best = (
        ranked.iloc[0]
    )

    # Best condition-based model specifically
    condition_models = (
        ranked[
            ranked["experiment"]
            ==
            "A_condition_based"
        ]
    )

    best_condition = (
        condition_models.iloc[0]
    )

    with open(
        SUMMARY_FILE,
        "w"
    ) as file:

        file.write(
            "AI MANUFACTURING DIGITAL TWIN\n"
        )

        file.write(
            "PHASE 6 BASELINE ML SUMMARY\n"
        )

        file.write(
            "=" * 70
            +
            "\n\n"
        )

        file.write(
            "Validation strategy:\n"
        )

        file.write(
            "5-fold GroupKFold using "
            "ToolIndex as grouping variable.\n\n"
        )

        file.write(
            "The locked external test tools "
            "were NOT used during Phase 6.\n\n"
        )

        file.write(
            "Best overall baseline:\n"
        )

        file.write(
            f"Experiment: "
            f"{best['experiment']}\n"
        )

        file.write(
            f"Model: "
            f"{best['model']}\n"
        )

        file.write(
            f"Features: "
            f"{best['number_features']}\n"
        )

        file.write(
            f"CV MAE: "
            f"{best['CV_MAE']:.6f}\n"
        )

        file.write(
            f"CV RMSE: "
            f"{best['CV_RMSE']:.6f}\n"
        )

        file.write(
            f"CV R2: "
            f"{best['CV_R2']:.6f}\n\n"
        )

        file.write(
            "Best condition-based baseline:\n"
        )

        file.write(
            f"Model: "
            f"{best_condition['model']}\n"
        )

        file.write(
            f"CV MAE: "
            f"{best_condition['CV_MAE']:.6f}\n"
        )

        file.write(
            f"CV RMSE: "
            f"{best_condition['CV_RMSE']:.6f}\n"
        )

        file.write(
            f"CV R2: "
            f"{best_condition['CV_R2']:.6f}\n"
        )

    print(
        "\nResults saved:"
    )

    print(
        f"✓ {RESULTS_FILE}"
    )

    print(
        f"✓ {FOLD_RESULTS_FILE}"
    )

    print(
        f"✓ {SUMMARY_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    train, manifest = (
        load_data()
    )

    (
        results_df,
        fold_results_df

    ) = run_experiments(
        train,
        manifest
    )

    ranked = rank_models(
        results_df
    )

    print_results(
        ranked
    )

    save_results(
        ranked,
        fold_results_df
    )

    print("\n" + "=" * 78)
    print("PHASE 6 COMPLETE")
    print("=" * 78)

    print(
        "\nImportant:"
    )

    print(
        "The locked final test tools "
        "have still NOT been evaluated."
    )

    print(
        "\nNext:"
    )

    print(
        "XGBoost + group-aware "
        "hyperparameter tuning."
    )


if __name__ == "__main__":
    main()