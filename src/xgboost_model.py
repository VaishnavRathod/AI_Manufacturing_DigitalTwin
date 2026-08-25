from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV
)

from xgboost import XGBRegressor


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

BASELINE_RESULTS_FILE = (
    REPORT_DIR /
    "phase6_baseline_cv_results.csv"
)

SEARCH_RESULTS_FILE = (
    REPORT_DIR /
    "phase7_xgboost_search_results.csv"
)

SUMMARY_RESULTS_FILE = (
    REPORT_DIR /
    "phase7_xgboost_summary.csv"
)

FOLD_RESULTS_FILE = (
    REPORT_DIR /
    "phase7_xgboost_fold_results.csv"
)

BEST_PARAMS_FILE = (
    REPORT_DIR /
    "phase7_xgboost_best_params.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "CycleToFailureNormalized"
GROUP_COLUMN = "ToolIndex"

N_SPLITS = 5

# Reduce to 10 if you want a faster first run.
N_ITER = 20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 78)
    print("AI MANUFACTURING DIGITAL TWIN")
    print("PHASE 7 — XGBOOST TOOL HEALTH MODEL")
    print("=" * 78)

    train = pd.read_csv(
        TRAIN_FILE
    )

    with open(
        FEATURE_MANIFEST_FILE,
        "r"
    ) as file:

        manifest = json.load(file)

    print(
        f"\nTraining rows  : {len(train)}"
    )

    print(
        f"Training tools : "
        f"{train[GROUP_COLUMN].nunique()}"
    )

    print(
        "\nLocked test dataset is NOT "
        "used anywhere in Phase 7."
    )

    return train, manifest


# ============================================================
# BASE XGBOOST MODEL
# ============================================================

def create_base_model():

    return XGBRegressor(

        objective="reg:squarederror",

        eval_metric="rmse",

        tree_method="hist",

        random_state=RANDOM_STATE,

        n_jobs=1
    )


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

def get_parameter_space():

    return {

        "n_estimators": [
            150,
            250,
            400,
            600
        ],

        "max_depth": [
            2,
            3,
            4,
            5
        ],

        "learning_rate": [
            0.02,
            0.05,
            0.08,
            0.10
        ],

        "subsample": [
            0.70,
            0.85,
            1.00
        ],

        "colsample_bytree": [
            0.60,
            0.80,
            1.00
        ],

        "min_child_weight": [
            1,
            3,
            5,
            8
        ],

        "gamma": [
            0.0,
            0.01,
            0.05,
            0.10
        ],

        "reg_alpha": [
            0.0,
            0.01,
            0.10,
            0.50
        ],

        "reg_lambda": [
            1.0,
            3.0,
            5.0,
            10.0
        ]
    }


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
# SEARCH ONE EXPERIMENT
# ============================================================

def tune_experiment(
    train,
    experiment_name,
    features
):

    print("\n" + "=" * 78)

    print(
        f"XGBOOST TUNING — {experiment_name}"
    )

    print("=" * 78)

    print(
        f"\nNumber of features: {len(features)}"
    )

    X = train[features].copy()

    y = train[TARGET].copy()

    groups = (
        train[GROUP_COLUMN]
        .copy()
    )

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    base_model = (
        create_base_model()
    )

    parameter_space = (
        get_parameter_space()
    )

    # --------------------------------------------------------
    # Use negative MSE for optimization.
    #
    # Minimizing MSE is equivalent to minimizing RMSE
    # for model ranking.
    # --------------------------------------------------------

    search = RandomizedSearchCV(

        estimator=base_model,

        param_distributions=
            parameter_space,

        n_iter=N_ITER,

        scoring=
            "neg_mean_squared_error",

        cv=cv,

        random_state=
            RANDOM_STATE,

        n_jobs=-1,

        verbose=0,

        refit=True,

        return_train_score=False
    )

    print(
        f"\nRandomized search: "
        f"{N_ITER} parameter combinations"
    )

    print(
        f"GroupKFold: {N_SPLITS} folds"
    )

    print(
        f"Total approximate fits: "
        f"{N_ITER * N_SPLITS}"
    )

    start_time = time.time()

    search.fit(
        X,
        y,
        groups=groups
    )

    runtime = (
        time.time()
        -
        start_time
    )

    best_model = (
        search.best_estimator_
    )

    best_params = (
        search.best_params_
    )

    best_cv_rmse = np.sqrt(
        -search.best_score_
    )

    print(
        "\nBest search CV RMSE:"
        f" {best_cv_rmse:.4f}"
    )

    print(
        "\nBest parameters:"
    )

    for key, value in (
        best_params.items()
    ):

        print(
            f"{key:20s}: {value}"
        )

    return (
        best_model,
        best_params,
        search,
        runtime
    )


# ============================================================
# TRUE OUT-OF-FOLD EVALUATION
# ============================================================

def evaluate_best_model(
    train,
    features,
    model,
    experiment_name
):

    X = train[features].copy()

    y = train[TARGET].copy()

    groups = (
        train[GROUP_COLUMN]
        .copy()
    )

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    all_true = []
    all_predictions = []

    fold_results = []

    print(
        "\nEvaluating tuned model "
        "with group-aware folds..."
    )

    for fold_number, (
        training_indices,
        validation_indices

    ) in enumerate(
        cv.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        model_fold = clone(
            model
        )

        X_train = (
            X.iloc[
                training_indices
            ]
        )

        X_validation = (
            X.iloc[
                validation_indices
            ]
        )

        y_train = (
            y.iloc[
                training_indices
            ]
        )

        y_validation = (
            y.iloc[
                validation_indices
            ]
        )

        training_tools = sorted(
            groups.iloc[
                training_indices
            ]
            .unique()
            .tolist()
        )

        validation_tools = sorted(
            groups.iloc[
                validation_indices
            ]
            .unique()
            .tolist()
        )

        overlap = (
            set(training_tools)
            .intersection(
                set(validation_tools)
            )
        )

        if overlap:

            raise ValueError(
                f"Tool leakage detected "
                f"in fold {fold_number}"
            )

        model_fold.fit(
            X_train,
            y_train
        )

        predictions = (
            model_fold.predict(
                X_validation
            )
        )

        metrics = (
            calculate_metrics(
                y_validation,
                predictions
            )
        )

        all_true.extend(
            y_validation.tolist()
        )

        all_predictions.extend(
            predictions.tolist()
        )

        fold_results.append({

            "experiment":
                experiment_name,

            "fold":
                fold_number,

            "validation_tools":
                str(validation_tools),

            "validation_rows":
                len(validation_indices),

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"],

            "R2":
                metrics["R2"]
        })

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

    overall = (
        calculate_metrics(
            np.asarray(all_true),
            np.asarray(
                all_predictions
            )
        )
    )

    fold_df = pd.DataFrame(
        fold_results
    )

    summary = {

        "experiment":
            experiment_name,

        "number_features":
            len(features),

        "CV_MAE":
            overall["MAE"],

        "CV_RMSE":
            overall["RMSE"],

        "CV_R2":
            overall["R2"],

        "Fold_MAE_Std":
            fold_df[
                "MAE"
            ].std(),

        "Fold_RMSE_Std":
            fold_df[
                "RMSE"
            ].std(),

        "Fold_R2_Std":
            fold_df[
                "R2"
            ].std()
    }

    print(
        "\nOverall tuned XGBoost "
        "out-of-fold performance:"
    )

    print(
        f"MAE  : "
        f"{overall['MAE']:.4f}"
    )

    print(
        f"RMSE : "
        f"{overall['RMSE']:.4f}"
    )

    print(
        f"R²   : "
        f"{overall['R2']:.4f}"
    )

    return summary, fold_results


# ============================================================
# RUN ALL FEATURE EXPERIMENTS
# ============================================================

def run_all_experiments(
    train,
    manifest
):

    experiments = (
        manifest[
            "experiments"
        ]
    )

    summaries = []

    fold_results = []

    best_parameter_dict = {}

    all_search_results = []

    for (
        experiment_name,
        features

    ) in experiments.items():

        (
            best_model,
            best_params,
            search,
            runtime

        ) = tune_experiment(

            train=train,

            experiment_name=
                experiment_name,

            features=features
        )

        (
            summary,
            experiment_fold_results

        ) = evaluate_best_model(

            train=train,

            features=features,

            model=best_model,

            experiment_name=
                experiment_name
        )

        summary[
            "SearchRuntimeSeconds"
        ] = runtime

        summaries.append(
            summary
        )

        fold_results.extend(
            experiment_fold_results
        )

        best_parameter_dict[
            experiment_name
        ] = best_params

        search_df = pd.DataFrame(
            search.cv_results_
        )

        search_df[
            "experiment"
        ] = experiment_name

        all_search_results.append(
            search_df
        )

    summary_df = pd.DataFrame(
        summaries
    )

    fold_df = pd.DataFrame(
        fold_results
    )

    search_results_df = pd.concat(
        all_search_results,
        ignore_index=True
    )

    return (
        summary_df,
        fold_df,
        search_results_df,
        best_parameter_dict
    )


# ============================================================
# RANK RESULTS
# ============================================================

def rank_results(
    summary_df
):

    ranked = (
        summary_df
        .sort_values(
            by=[
                "CV_RMSE",
                "CV_MAE"
            ],
            ascending=True
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
# COMPARE RANDOM FOREST
# ============================================================

def print_baseline_comparison(
    ranked
):

    print("\n" + "=" * 78)
    print("PHASE 7 XGBOOST RANKING")
    print("=" * 78)

    display_columns = [

        "Rank",
        "experiment",
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

    numeric = [

        "CV_MAE",
        "CV_RMSE",
        "CV_R2",
        "Fold_RMSE_Std"
    ]

    display[
        numeric
    ] = (
        display[
            numeric
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

    # --------------------------------------------------------
    # Optional baseline comparison
    # --------------------------------------------------------

    if BASELINE_RESULTS_FILE.exists():

        baseline = pd.read_csv(
            BASELINE_RESULTS_FILE
        )

        rf = baseline[
            baseline["model"]
            ==
            "RandomForest"
        ].copy()

        print(
            "\n\nRandom Forest comparison:"
        )

        for _, xgb_row in (
            ranked.iterrows()
        ):

            experiment = (
                xgb_row[
                    "experiment"
                ]
            )

            rf_row = rf[
                rf["experiment"]
                ==
                experiment
            ]

            if rf_row.empty:
                continue

            rf_row = (
                rf_row.iloc[0]
            )

            print(
                f"\n{experiment}"
            )

            print(
                f"RF RMSE  : "
                f"{rf_row['CV_RMSE']:.4f}"
            )

            print(
                f"XGB RMSE : "
                f"{xgb_row['CV_RMSE']:.4f}"
            )

            print(
                f"RF R²    : "
                f"{rf_row['CV_R2']:.4f}"
            )

            print(
                f"XGB R²   : "
                f"{xgb_row['CV_R2']:.4f}"
            )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    ranked,
    fold_df,
    search_df,
    best_params
):

    ranked.to_csv(
        SUMMARY_RESULTS_FILE,
        index=False
    )

    fold_df.to_csv(
        FOLD_RESULTS_FILE,
        index=False
    )

    search_df.to_csv(
        SEARCH_RESULTS_FILE,
        index=False
    )

    with open(
        BEST_PARAMS_FILE,
        "w"
    ) as file:

        json.dump(
            best_params,
            file,
            indent=4
        )

    print(
        "\nSaved:"
    )

    print(
        f"✓ {SUMMARY_RESULTS_FILE}"
    )

    print(
        f"✓ {FOLD_RESULTS_FILE}"
    )

    print(
        f"✓ {SEARCH_RESULTS_FILE}"
    )

    print(
        f"✓ {BEST_PARAMS_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    train, manifest = (
        load_data()
    )

    (
        summary_df,
        fold_df,
        search_df,
        best_params

    ) = run_all_experiments(
        train,
        manifest
    )

    ranked = rank_results(
        summary_df
    )

    print_baseline_comparison(
        ranked
    )

    save_results(
        ranked,
        fold_df,
        search_df,
        best_params
    )

    print("\n" + "=" * 78)
    print("PHASE 7 COMPLETE")
    print("=" * 78)

    print(
        "\nThe locked final test tools "
        "remain untouched."
    )

    print(
        "\nNext phase:"
    )

    print(
        "Select final candidate models "
        "and evaluate once on "
        "Tools 2, 11 and 102."
    )


if __name__ == "__main__":
    main()