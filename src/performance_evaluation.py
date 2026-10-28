import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from scipy.stats import spearmanr

import shap
from lime.lime_tabular import LimeTabularExplainer


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "accident_ml_dataset.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FEATURES
# =========================================================

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


# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(DATA_PATH)

X = data[FEATURES].copy()
y = data["risk_level"].astype(str)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))


# =========================================================
# ENCODE TARGET
# =========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

class_names = encoder.classes_


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced"
        )
}


# =========================================================
# MODEL EVALUATION
# =========================================================

results = []

predictions = {}

trained_models = {}


for name, model in models.items():

    # -----------------------------
    # Training time
    # -----------------------------

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time


    # -----------------------------
    # Prediction time
    # -----------------------------

    start_time = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start_time


    predictions[name] = y_pred

    trained_models[name] = model


    # -----------------------------
    # Classification metrics
    # -----------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # -----------------------------
    # Store results
    # -----------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "Training Time (ms)":
            training_time * 1000,

        "Prediction Time (ms)":
            prediction_time * 1000,

        "Prediction Time per Sample (ms)":
            (prediction_time / len(X_test)) * 1000
    })


# =========================================================
# SAVE PERFORMANCE RESULTS
# =========================================================

results_df = pd.DataFrame(results)

results_path = (
    RESULTS_DIR /
    "performance_evaluation.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ].round(4)
)


print("\n========================================")
print("EXECUTION TIME")
print("========================================")

print(
    results_df[
        [
            "Model",
            "Training Time (ms)",
            "Prediction Time (ms)",
            "Prediction Time per Sample (ms)"
        ]
    ].round(4)
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

rf_model = trained_models["Random Forest"]

rf_predictions = predictions["Random Forest"]

cm = confusion_matrix(
    y_test,
    rf_predictions
)


plt.figure(figsize=(7, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel(
    "Predicted Risk"
)

plt.ylabel(
    "Actual Risk"
)

plt.xticks(
    range(len(class_names)),
    class_names
)

plt.yticks(
    range(len(class_names)),
    class_names
)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()


cm_path = (
    RESULTS_DIR /
    "confusion_matrix_random_forest.png"
)

plt.savefig(
    cm_path,
    dpi=200
)

plt.close()


# =========================================================
# RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Importance":
        rf_model.feature_importances_

})


importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)


importance_path = (
    RESULTS_DIR /
    "feature_importance_evaluation.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# =========================================================
# SHAP EVALUATION
# =========================================================

try:

    shap_explainer = shap.TreeExplainer(
        rf_model
    )

    shap_values = shap_explainer.shap_values(
        X_test
    )


    if isinstance(shap_values, list):

        shap_array = np.mean(
            [
                np.abs(value)
                for value in shap_values
            ],
            axis=0
        )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            shap_array = np.mean(
                np.abs(shap_array),
                axis=2
            )

        else:

            shap_array = np.abs(
                shap_array
            )


    mean_shap = np.mean(
        shap_array,
        axis=0
    )


    shap_df = pd.DataFrame({

        "Feature": FEATURES,

        "Mean Absolute SHAP":
            mean_shap

    })


    shap_df = shap_df.sort_values(
        "Mean Absolute SHAP",
        ascending=False
    )


    shap_path = (
        RESULTS_DIR /
        "shap_evaluation.csv"
    )

    shap_df.to_csv(
        shap_path,
        index=False
    )


    # ---------------------------------------------
    # Compare RF importance ranking with SHAP ranking
    # ---------------------------------------------

    rf_rank = (
        importance_df
        .set_index("Feature")
        .loc[FEATURES]["Importance"]
        .rank(ascending=False)
    )


    shap_rank = (
        shap_df
        .set_index("Feature")
        .loc[FEATURES]["Mean Absolute SHAP"]
        .rank(ascending=False)
    )


    shap_correlation = spearmanr(
        rf_rank,
        shap_rank
    ).statistic


    print("\n========================================")
    print("SHAP EVALUATION")
    print("========================================")

    print(
        "SHAP vs Random Forest ranking correlation:",
        round(shap_correlation, 4)
    )


except Exception as e:

    shap_correlation = np.nan

    print(
        "SHAP evaluation warning:",
        e
    )


# =========================================================
# LIME EVALUATION
# =========================================================

try:

    lime_explainer = LimeTabularExplainer(

        X_train.values,

        feature_names=FEATURES,

        class_names=class_names,

        mode="classification",

        random_state=42
    )


    sample = X_test.iloc[0].values


    explanation = lime_explainer.explain_instance(

        sample,

        rf_model.predict_proba,

        num_features=len(FEATURES)
    )


    lime_fidelity = explanation.score


    print("\n========================================")
    print("LIME EVALUATION")
    print("========================================")

    print(
        "LIME local fidelity:",
        round(lime_fidelity, 4)
    )


    lime_path = (
        RESULTS_DIR /
        "lime_evaluation.txt"
    )


    with open(
        lime_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "LIME Local Fidelity (R2): "
            + str(round(lime_fidelity, 4))
        )


except Exception as e:

    lime_fidelity = np.nan

    print(
        "LIME evaluation warning:",
        e
    )


# =========================================================
# XAI SUMMARY
# =========================================================

xai_summary = pd.DataFrame({

    "Evaluation": [

        "SHAP-RF Rank Correlation",

        "LIME Local Fidelity"
    ],

    "Score": [

        shap_correlation,

        lime_fidelity
    ]

})


xai_summary_path = (
    RESULTS_DIR /
    "xai_evaluation.csv"
)

xai_summary.to_csv(
    xai_summary_path,
    index=False
)


# =========================================================
# COMPLETION
# =========================================================

print("\n========================================")
print("FILES GENERATED")
print("========================================")

print(
    "Performance:",
    results_path
)

print(
    "Confusion Matrix:",
    cm_path
)

print(
    "Feature Importance:",
    importance_path
)

print(
    "SHAP Evaluation:",
    RESULTS_DIR / "shap_evaluation.csv"
)

print(
    "LIME Evaluation:",
    RESULTS_DIR / "lime_evaluation.txt"
)

print(
    "XAI Summary:",
    xai_summary_path
)

print("\nPerformance evaluation completed successfully.")