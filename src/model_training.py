from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "accident_ml_dataset.csv"
)

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# FEATURES USED BY THE MODEL
# ============================================================

FEATURES = [
    "2019 Accidents",
    "2020 Accidents",
    "2021 Accidents",
    "2022 Accidents",
    "average_accidents",
    "total_accidents_2019_2022",
    "accident_growth_2019_2022",
    "accident_variation"
]


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["risk_level"]

    # XGBoost requires numerical class labels.
    # Other models will continue using High/Medium/Low.
    label_encoder = LabelEncoder()

    y_xgb = pd.Series(
        label_encoder.fit_transform(y),
        index=y.index
    )

    print("=" * 60)
    print("SMART TRANSPORTATION MODEL TRAINING")
    print("=" * 60)

    print("Dataset shape:", df.shape)
    print("Features:", len(FEATURES))

    print()
    print("Risk classes:")
    print(y.value_counts())

    print()
    print("XGBoost label mapping:")

    for class_name, class_number in zip(
        label_encoder.classes_,
        label_encoder.transform(label_encoder.classes_)
    ):
        print(
            f"{class_name} -> {class_number}"
        )

    # --------------------------------------------------------
    # INDIVIDUAL MODELS
    # --------------------------------------------------------

    models = {

        "Logistic Regression": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000
                )
            )
        ]),

        "Decision Tree": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                DecisionTreeClassifier(
                    max_depth=4,
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]),

        "Gradient Boosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]),

        "XGBoost": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ])
    }

    # --------------------------------------------------------
    # STRATIFIED CROSS VALIDATION
    # --------------------------------------------------------

    class_counts = y.value_counts()

    n_splits = max(
        2,
        min(
            5,
            int(class_counts.min())
        )
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    print()
    print("Cross-validation folds:", n_splits)

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    rows = []
    timing_rows = []

    fitted_models = {}

    rf_predictions = None

    # --------------------------------------------------------
    # TRAIN EACH INDIVIDUAL MODEL
    # --------------------------------------------------------

    for name, model in models.items():

        print()
        print("-" * 60)
        print("Training:", name)

        # ----------------------------------------------------
        # SELECT TARGET
        # ----------------------------------------------------

        if name == "XGBoost":
            current_y = y_xgb
        else:
            current_y = y

        # ----------------------------------------------------
        # CROSS-VALIDATION PREDICTION
        # ----------------------------------------------------

        prediction_start = time.perf_counter()

        predictions = cross_val_predict(
            model,
            X,
            current_y,
            cv=cv
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        # ----------------------------------------------------
        # CONVERT XGBOOST PREDICTIONS BACK TO TEXT
        # ----------------------------------------------------

        if name == "XGBoost":

            predictions_for_metrics = (
                label_encoder.inverse_transform(
                    predictions.astype(int)
                )
            )

        else:

            predictions_for_metrics = predictions

        # ----------------------------------------------------
        # PERFORMANCE METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y,
            predictions_for_metrics
        )

        precision = precision_score(
            y,
            predictions_for_metrics,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions_for_metrics,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions_for_metrics,
            average="weighted",
            zero_division=0
        )

        rows.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # ----------------------------------------------------
        # STORE RANDOM FOREST PREDICTIONS
        # ----------------------------------------------------

        if name == "Random Forest":

            rf_predictions = predictions

        # ----------------------------------------------------
        # TRAIN FINAL MODEL ON COMPLETE DATA
        # ----------------------------------------------------

        training_start = time.perf_counter()

        model.fit(
            X,
            current_y
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )

        joblib.dump(
            model,
            MODELS_DIR
            / f"{safe_name}.joblib"
        )

        fitted_models[name] = model

        # ----------------------------------------------------
        # SAVE TIMING
        # ----------------------------------------------------

        timing_rows.append({
            "Model": name,
            "Training Time (seconds)": training_time,
            "Cross-Validation Prediction Time (seconds)": (
                prediction_time
            ),
            "Total Time (seconds)": (
                training_time
                + prediction_time
            )
        })

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        print(
            f"Training Time: "
            f"{training_time:.4f} seconds"
        )

        print(
            f"CV Prediction Time: "
            f"{prediction_time:.4f} seconds"
        )

    # ========================================================
    # HYBRID ENSEMBLE
    # ========================================================

    print()
    print("=" * 60)
    print("TRAINING HYBRID ENSEMBLE")
    print("=" * 60)

    print()
    print("Base Models:")
    print("1. Random Forest")
    print("2. Gradient Boosting")
    print("3. XGBoost")
    print()
    print("Meta Model: Logistic Regression")

    # --------------------------------------------------------
    # BASE MODEL 1 - RANDOM FOREST
    # --------------------------------------------------------

    rf_base = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    # --------------------------------------------------------
    # BASE MODEL 2 - GRADIENT BOOSTING
    # --------------------------------------------------------

    gb_base = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ])

    # --------------------------------------------------------
    # BASE MODEL 3 - XGBOOST
    # --------------------------------------------------------

    xgb_base = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # --------------------------------------------------------
    # META MODEL
    # --------------------------------------------------------

    meta_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000
            )
        )
    ])

    # --------------------------------------------------------
    # STACKING MODEL
    # --------------------------------------------------------

    hybrid_model = StackingClassifier(
        estimators=[
            (
                "random_forest",
                rf_base
            ),
            (
                "gradient_boosting",
                gb_base
            ),
            (
                "xgboost",
                xgb_base
            )
        ],
        final_estimator=meta_model,
        cv=cv,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # XGBoost inside the ensemble also requires numerical
    # target labels.
    # --------------------------------------------------------

    ensemble_prediction_start = time.perf_counter()

    ensemble_predictions_encoded = cross_val_predict(
        hybrid_model,
        X,
        y_xgb,
        cv=cv
    )

    ensemble_prediction_time = (
        time.perf_counter()
        - ensemble_prediction_start
    )

    # --------------------------------------------------------
    # CONVERT ENSEMBLE PREDICTIONS BACK TO ORIGINAL LABELS
    # --------------------------------------------------------

    ensemble_predictions = (
        label_encoder.inverse_transform(
            ensemble_predictions_encoded.astype(int)
        )
    )

    # --------------------------------------------------------
    # ENSEMBLE METRICS
    # --------------------------------------------------------

    ensemble_accuracy = accuracy_score(
        y,
        ensemble_predictions
    )

    ensemble_precision = precision_score(
        y,
        ensemble_predictions,
        average="weighted",
        zero_division=0
    )

    ensemble_recall = recall_score(
        y,
        ensemble_predictions,
        average="weighted",
        zero_division=0
    )

    ensemble_f1 = f1_score(
        y,
        ensemble_predictions,
        average="weighted",
        zero_division=0
    )

    rows.append({
        "Model": "Hybrid Ensemble",
        "Accuracy": ensemble_accuracy,
        "Precision": ensemble_precision,
        "Recall": ensemble_recall,
        "F1 Score": ensemble_f1
    })

    # --------------------------------------------------------
    # TRAIN FINAL HYBRID ENSEMBLE
    # --------------------------------------------------------

    ensemble_training_start = time.perf_counter()

    hybrid_model.fit(
        X,
        y_xgb
    )

    ensemble_training_time = (
        time.perf_counter()
        - ensemble_training_start
    )

    # --------------------------------------------------------
    # SAVE HYBRID ENSEMBLE
    # --------------------------------------------------------

    joblib.dump(
        hybrid_model,
        MODELS_DIR
        / "hybrid_ensemble.joblib"
    )

    # Save label encoder separately.
    # The dashboard will need it when using the ensemble.
    joblib.dump(
        label_encoder,
        MODELS_DIR
        / "risk_label_encoder.joblib"
    )

    # --------------------------------------------------------
    # ENSEMBLE TIMING
    # --------------------------------------------------------

    timing_rows.append({
        "Model": "Hybrid Ensemble",
        "Training Time (seconds)": (
            ensemble_training_time
        ),
        "Cross-Validation Prediction Time (seconds)": (
            ensemble_prediction_time
        ),
        "Total Time (seconds)": (
            ensemble_training_time
            + ensemble_prediction_time
        )
    })

    print()
    print(
        f"Hybrid Ensemble Accuracy : "
        f"{ensemble_accuracy:.4f}"
    )

    print(
        f"Hybrid Ensemble Precision: "
        f"{ensemble_precision:.4f}"
    )

    print(
        f"Hybrid Ensemble Recall   : "
        f"{ensemble_recall:.4f}"
    )

    print(
        f"Hybrid Ensemble F1 Score : "
        f"{ensemble_f1:.4f}"
    )

    print(
        f"Training Time: "
        f"{ensemble_training_time:.4f} seconds"
    )

    print(
        f"CV Prediction Time: "
        f"{ensemble_prediction_time:.4f} seconds"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        RESULTS_DIR
        / "model_comparison.csv",
        index=False
    )

    # ========================================================
    # MODEL TIMING
    # ========================================================

    timing_df = pd.DataFrame(
        timing_rows
    )

    timing_df.to_csv(
        RESULTS_DIR
        / "model_timing.csv",
        index=False
    )

    # ========================================================
    # RANDOM FOREST CONFUSION MATRIX
    # ========================================================

    if rf_predictions is not None:

        class_labels = sorted(
            y.unique().tolist()
        )

        rf_cm = confusion_matrix(
            y,
            rf_predictions,
            labels=class_labels
        )

        confusion_df = pd.DataFrame(
            rf_cm,
            index=class_labels,
            columns=class_labels
        )

        confusion_df.index.name = "Actual"
        confusion_df.columns.name = "Predicted"

        confusion_df.to_csv(
            RESULTS_DIR
            / "confusion_matrix.csv"
        )

        print()
        print("=" * 60)
        print("RANDOM FOREST CONFUSION MATRIX")
        print("=" * 60)

        print(confusion_df)

    # ========================================================
    # HYBRID ENSEMBLE CONFUSION MATRIX
    # ========================================================

    class_labels = sorted(
        y.unique().tolist()
    )

    ensemble_cm = confusion_matrix(
        y,
        ensemble_predictions,
        labels=class_labels
    )

    ensemble_confusion_df = pd.DataFrame(
        ensemble_cm,
        index=class_labels,
        columns=class_labels
    )

    ensemble_confusion_df.index.name = "Actual"
    ensemble_confusion_df.columns.name = "Predicted"

    ensemble_confusion_df.to_csv(
        RESULTS_DIR
        / "ensemble_confusion_matrix.csv"
    )

    print()
    print("=" * 60)
    print("HYBRID ENSEMBLE CONFUSION MATRIX")
    print("=" * 60)

    print(ensemble_confusion_df)

    # ========================================================
    # RANDOM FOREST FEATURE IMPORTANCE
    # ========================================================

    rf_model = fitted_models[
        "Random Forest"
    ]

    rf_estimator = rf_model.named_steps[
        "model"
    ]

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf_estimator.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    importance_df.to_csv(
        RESULTS_DIR
        / "feature_importance.csv",
        index=False
    )

    # ========================================================
    # MODEL METADATA
    # ========================================================

    metadata = {
        "features": FEATURES,

        "classes": sorted(
            y.unique().tolist()
        ),

        "explainability_model": "Random Forest",

        "hybrid_ensemble": {
            "enabled": True,

            "base_models": [
                "Random Forest",
                "Gradient Boosting",
                "XGBoost"
            ],

            "meta_model": "Logistic Regression"
        },

        "xgboost_label_encoding": {
            "enabled": True,
            "mapping": {
                class_name: int(
                    label_encoder.transform(
                        [class_name]
                    )[0]
                )
                for class_name in label_encoder.classes_
            }
        },

        "note": (
            "Risk labels are proxy labels derived "
            "from 2023 state accident counts."
        )
    }

    with open(
        MODELS_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print("Generated result files:")

    print("✓ model_comparison.csv")
    print("✓ model_timing.csv")
    print("✓ confusion_matrix.csv")
    print("✓ ensemble_confusion_matrix.csv")
    print("✓ feature_importance.csv")

    print()
    print("Generated model files:")

    print("✓ logistic_regression.joblib")
    print("✓ decision_tree.joblib")
    print("✓ random_forest.joblib")
    print("✓ gradient_boosting.joblib")
    print("✓ xgboost.joblib")
    print("✓ hybrid_ensemble.joblib")
    print("✓ risk_label_encoder.joblib")

    print()
    print(
        "Models saved in:",
        MODELS_DIR
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_models()