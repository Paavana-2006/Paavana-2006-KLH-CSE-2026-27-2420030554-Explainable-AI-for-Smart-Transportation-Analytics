# ================================================================
# STRICT ML + DL HYBRID EXPERIMENT
# Smart Transportation Explainable AI Framework
# ================================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ================================================================
# PATHS AND SETTINGS
# ================================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "multimodal_state_year_dataset.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

# ================================================================
# LOAD DATA
# ================================================================

print("=" * 70)
print("STRICT ML + DL HYBRID EXPERIMENT")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset:", df.shape)

TARGET = "risk_trend"
GROUP = "State"

if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' not found.")

if GROUP not in df.columns:
    raise ValueError(f"Group '{GROUP}' not found.")

# ================================================================
# TARGET ENCODING
# ================================================================

classes = sorted(df[TARGET].astype(str).unique())

class_to_id = {
    name: i for i, name in enumerate(classes)
}

y = df[TARGET].astype(str).map(class_to_id).values
groups = df[GROUP].astype(str).values
N_CLASSES = len(classes)

if N_CLASSES < 2:
    raise ValueError("At least two target classes are required.")

if len(np.unique(groups)) < N_SPLITS:
    raise ValueError(
        f"Five-fold GroupKFold requires at least {N_SPLITS} "
        f"unique groups. Found {len(np.unique(groups))}."
    )

print("\nClass ID mapping:")
for class_id, class_name in enumerate(classes):
    print(f"{class_id}: {class_name}")

print("Number of groups:", len(np.unique(groups)))

# ================================================================
# REMOVE LEAKAGE-PRONE FEATURES
# ================================================================

LEAKAGE_FEATURES = [
    "risk_level",
    "risk_trend",
    "accident_yoy_change",
    "change_from_2022_to_2023",
    "% change from 2022 to 2023",
    "percent_change_2022_2023",
    "2019 Ranking",
    "2020 Ranking",
    "2021 Ranking",
    "2022 Ranking",
    "2023 Ranking",
    "accident_growth_2019_2022",
    "accident_growth",
    "accident_change",
    "trend",
    "target"
]

existing_leakage = [
    col for col in LEAKAGE_FEATURES if col in df.columns
]

print("\nRemoved target-derived / leakage-prone features:")

if existing_leakage:
    for col in existing_leakage:
        print(" -", col)
else:
    print(" - None found")

# ================================================================
# BUILD FEATURE MATRIX
# ================================================================

X_df = df.drop(
    columns=existing_leakage,
    errors="ignore"
)

X_df = X_df.drop(
    columns=[GROUP],
    errors="ignore"
)

X_df = X_df.select_dtypes(include=[np.number])

X_df = X_df.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

feature_names = list(X_df.columns)
X = X_df.values.astype(float)

if X.shape[1] == 0:
    raise ValueError("No numeric features remain after feature removal.")

print("\nIndependent numeric features:", X.shape[1])

for feature in feature_names:
    print(" -", feature)

# ================================================================
# MODEL FACTORIES
# ================================================================

def create_models():

    models = {}

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    models["Gradient Boosting"] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=3,
        random_state=RANDOM_STATE
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    models["MLP / FNN"] = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=RANDOM_STATE
        ))
    ])

    return models

# ================================================================
# PROBABILITY ALIGNMENT
# ================================================================

def get_probabilities(model, X_test):

    probabilities = model.predict_proba(X_test)

    result = np.zeros(
        (len(X_test), N_CLASSES)
    )

    for i, cls in enumerate(model.classes_):
        result[:, int(cls)] = probabilities[:, i]

    return result

# ================================================================
# METRICS
# ================================================================

def calculate_metrics(y_true, y_pred, probabilities):

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )
    }

    try:
        metrics["ROC-AUC"] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="weighted",
            labels=np.arange(N_CLASSES)
        )
    except Exception:
        metrics["ROC-AUC"] = np.nan

    return metrics

# ================================================================
# FEATURE SELECTION
# ================================================================

def create_selector(X_train):

    k = min(10, X_train.shape[1])

    return SelectKBest(
        score_func=lambda X_data, y_data:
            mutual_info_classif(
                X_data,
                y_data,
                random_state=RANDOM_STATE
            ),
        k=k
    )


def fit_model_with_selection(
    model_name,
    X_train,
    y_train,
    X_eval
):

    selector = create_selector(X_train)

    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )

    X_eval_selected = selector.transform(X_eval)

    model = create_models()[model_name]

    model.fit(
        X_train_selected,
        y_train
    )

    return get_probabilities(
        model,
        X_eval_selected
    )

# ================================================================
# BASELINE EVALUATION
# ================================================================

def evaluate_baseline(model_name):

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    outer_cv = GroupKFold(n_splits=N_SPLITS)

    true_values = []
    predictions = []
    probabilities = []
    fold_scores = []

    start = time.time()

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups),
        start=1
    ):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        prob = fit_model_with_selection(
            model_name,
            X_train,
            y_train,
            X_test
        )

        pred = np.argmax(prob, axis=1)

        fold_f1 = f1_score(
            y_test,
            pred,
            average="weighted",
            zero_division=0
        )

        fold_scores.append(fold_f1)

        print(f"Fold {fold}: F1={fold_f1:.4f}")

        true_values.extend(y_test)
        predictions.extend(pred)
        probabilities.extend(prob)

    elapsed = time.time() - start

    true_values = np.asarray(true_values)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    metrics = calculate_metrics(
        true_values,
        predictions,
        probabilities
    )

    print(f"Overall F1: {metrics['F1']:.4f}")

    return {
        "Model": model_name,
        "Type": (
            "Deep Learning"
            if model_name == "MLP / FNN"
            else "Machine Learning"
        ),
        **metrics,
        "Time": elapsed,
        "Fold F1 Mean": np.mean(fold_scores),
        "Fold F1 Std": np.std(fold_scores)
    }

# ================================================================
# INNER OOF PREDICTIONS FOR WEIGHT OPTIMIZATION
# ================================================================

def create_oof_predictions(
    model_name,
    X_train,
    y_train,
    groups_train
):

    oof = np.zeros(
        (len(X_train), N_CLASSES)
    )

    unique_groups = np.unique(groups_train)

    inner_splits = min(3, len(unique_groups))

    if inner_splits < 2:
        raise ValueError(
            "Not enough groups for inner GroupKFold."
        )

    inner_cv = GroupKFold(n_splits=inner_splits)

    for inner_train, inner_valid in inner_cv.split(
        X_train,
        y_train,
        groups_train
    ):

        X_inner_train = X_train[inner_train]
        X_inner_valid = X_train[inner_valid]
        y_inner_train = y_train[inner_train]

        oof[inner_valid] = fit_model_with_selection(
            model_name,
            X_inner_train,
            y_inner_train,
            X_inner_valid
        )

    return oof

# ================================================================
# WEIGHT OPTIMIZATION
# ================================================================

def optimize_weights(oof_predictions, y_train):

    number_of_models = len(oof_predictions)

    best_weights = None
    best_f1 = -1.0

    if number_of_models == 2:

        for weight_1 in np.arange(0.05, 1.00, 0.05):

            weight_2 = 1.0 - weight_1

            if weight_2 < 0.05:
                continue

            combined = (
                weight_1 * oof_predictions[0]
                + weight_2 * oof_predictions[1]
            )

            prediction = np.argmax(combined, axis=1)

            score = f1_score(
                y_train,
                prediction,
                average="weighted",
                zero_division=0
            )

            if score > best_f1:
                best_f1 = score
                best_weights = [weight_1, weight_2]

    elif number_of_models == 3:

        for weight_1 in np.arange(0.05, 1.00, 0.05):

            for weight_2 in np.arange(0.05, 1.00, 0.05):

                weight_3 = 1.0 - weight_1 - weight_2

                if weight_3 < 0.05:
                    continue

                combined = (
                    weight_1 * oof_predictions[0]
                    + weight_2 * oof_predictions[1]
                    + weight_3 * oof_predictions[2]
                )

                prediction = np.argmax(combined, axis=1)

                score = f1_score(
                    y_train,
                    prediction,
                    average="weighted",
                    zero_division=0
                )

                if score > best_f1:
                    best_f1 = score
                    best_weights = [
                        weight_1,
                        weight_2,
                        weight_3
                    ]

    else:
        raise ValueError(
            "Only 2 or 3 base models are supported."
        )

    if best_weights is None:
        raise RuntimeError("Could not find valid hybrid weights.")

    return best_weights, best_f1

# ================================================================
# HYBRID EVALUATION
# ================================================================

def evaluate_hybrid(hybrid_name, base_models):

    print("\n" + "=" * 70)
    print(hybrid_name)
    print("=" * 70)

    outer_cv = GroupKFold(n_splits=N_SPLITS)

    true_values = []
    predictions = []
    probabilities = []
    fold_scores = []
    selected_weights = []

    start = time.time()

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups),
        start=1
    ):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        # Step 1: Generate inner OOF probabilities
        oof_predictions = []

        for model_name in base_models:
            oof = create_oof_predictions(
                model_name,
                X_train,
                y_train,
                groups_train
            )

            oof_predictions.append(oof)

        # Step 2: Optimize weights using training data only
        best_weights, inner_f1 = optimize_weights(
            oof_predictions,
            y_train
        )

        selected_weights.append(best_weights)

        print(
            "Selected weights:",
            [round(value, 2) for value in best_weights]
        )

        print(f"Inner training F1: {inner_f1:.4f}")

        # Step 3: Train on outer training fold
        test_probabilities = []

        for model_name in base_models:

            test_probability = fit_model_with_selection(
                model_name,
                X_train,
                y_train,
                X_test
            )

            test_probabilities.append(test_probability)

        # Step 4: Weighted probability fusion
        final_probability = np.zeros_like(
            test_probabilities[0]
        )

        for weight, probability in zip(
            best_weights,
            test_probabilities
        ):
            final_probability += weight * probability

        final_prediction = np.argmax(
            final_probability,
            axis=1
        )

        # Step 5: Evaluate held-out fold
        fold_f1 = f1_score(
            y_test,
            final_prediction,
            average="weighted",
            zero_division=0
        )

        fold_scores.append(fold_f1)

        print(f"Fold {fold}: F1={fold_f1:.4f}")

        true_values.extend(y_test)
        predictions.extend(final_prediction)
        probabilities.extend(final_probability)

    elapsed = time.time() - start

    true_values = np.asarray(true_values, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    metrics = calculate_metrics(
        true_values,
        predictions,
        probabilities
    )

    average_weights = np.mean(
        np.asarray(selected_weights),
        axis=0
    )

    print(f"\nOverall F1: {metrics['F1']:.4f}")
    print(
        "Average weights:",
        [round(value, 3) for value in average_weights]
    )

    # Keep each candidate's predictions so we can save the
    # predictions for the best hybrid after all candidates finish.
    return {
        "result": {
            "Model": hybrid_name,
            "Type": "ML + DL Hybrid",
            **metrics,
            "Time": elapsed,
            "Fold F1 Mean": np.mean(fold_scores),
            "Fold F1 Std": np.std(fold_scores)
        },
        "y_true": true_values,
        "y_pred": predictions,
        "probabilities": probabilities,
        "average_weights": average_weights
    }

# ================================================================
# RUN BASELINE MODELS
# ================================================================

baseline_models = [
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "MLP / FNN"
]

baseline_results = []

for model_name in baseline_models:
    baseline_results.append(
        evaluate_baseline(model_name)
    )

# ================================================================
# HYBRID CANDIDATES
# ================================================================

hybrid_candidates = {
    "Random Forest + MLP": [
        "Random Forest",
        "MLP / FNN"
    ],

    "Gradient Boosting + MLP": [
        "Gradient Boosting",
        "MLP / FNN"
    ],

    "XGBoost + MLP": [
        "XGBoost",
        "MLP / FNN"
    ],

    "Random Forest + XGBoost + MLP": [
        "Random Forest",
        "XGBoost",
        "MLP / FNN"
    ],

    "Random Forest + Gradient Boosting + MLP": [
        "Random Forest",
        "Gradient Boosting",
        "MLP / FNN"
    ],

    "Gradient Boosting + XGBoost + MLP": [
        "Gradient Boosting",
        "XGBoost",
        "MLP / FNN"
    ]
}

# ================================================================
# RUN ALL HYBRIDS
# ================================================================

hybrid_runs = {}
hybrid_results = []

for hybrid_name, model_names in hybrid_candidates.items():

    run = evaluate_hybrid(
        hybrid_name,
        model_names
    )

    hybrid_runs[hybrid_name] = run
    hybrid_results.append(run["result"])

# ================================================================
# SAVE COMPARISON RESULTS
# ================================================================

all_results = pd.DataFrame(
    baseline_results + hybrid_results
)

all_results = all_results.sort_values(
    by="F1",
    ascending=False
).reset_index(drop=True)

comparison_csv = os.path.join(
    RESULTS_DIR,
    "strict_ml_dl_comparison.csv"
)

hybrid_csv = os.path.join(
    RESULTS_DIR,
    "strict_ml_dl_hybrids.csv"
)

baseline_csv = os.path.join(
    RESULTS_DIR,
    "strict_ml_dl_baselines.csv"
)

all_results.to_csv(comparison_csv, index=False)

pd.DataFrame(hybrid_results).to_csv(
    hybrid_csv,
    index=False
)

pd.DataFrame(baseline_results).to_csv(
    baseline_csv,
    index=False
)

# ================================================================
# SELECT BEST BASELINE AND BEST HYBRID
# ================================================================

baseline_df = all_results[
    all_results["Type"].isin([
        "Machine Learning",
        "Deep Learning"
    ])
]

hybrid_df = all_results[
    all_results["Type"] == "ML + DL Hybrid"
]

best_baseline = baseline_df.loc[
    baseline_df["F1"].idxmax()
]

best_hybrid = hybrid_df.loc[
    hybrid_df["F1"].idxmax()
]

improvement = (
    float(best_hybrid["F1"])
    - float(best_baseline["F1"])
)

if float(best_baseline["F1"]) != 0:
    improvement_percentage = (
        improvement / float(best_baseline["F1"])
    ) * 100
else:
    improvement_percentage = np.nan

status = (
    "Hybrid outperformed baseline"
    if improvement > 0
    else "No hybrid exceeded baseline"
)

print("\n" + "=" * 70)
print("FINAL HYBRID SELECTION")
print("=" * 70)

print(f"Best Baseline : {best_baseline['Model']}")
print(f"Baseline F1   : {best_baseline['F1']:.4f}")
print(f"\nBest Hybrid   : {best_hybrid['Model']}")
print(f"Hybrid F1     : {best_hybrid['F1']:.4f}")
print(f"\nImprovement   : {improvement:+.4f}")
print(f"Improvement % : {improvement_percentage:+.2f}%")
print(status)

selection = pd.DataFrame([{
    "Best Baseline": best_baseline["Model"],
    "Baseline F1": best_baseline["F1"],
    "Best Hybrid": best_hybrid["Model"],
    "Hybrid F1": best_hybrid["F1"],
    "Improvement": improvement,
    "Improvement Percentage": improvement_percentage,
    "Status": status
}])

selection_csv = os.path.join(
    RESULTS_DIR,
    "strict_ml_dl_selection.csv"
)

selection.to_csv(selection_csv, index=False)

# ================================================================
# SAVE BEST HYBRID PREDICTIONS AND CONFUSION MATRIX
# ================================================================

best_hybrid_name = str(best_hybrid["Model"])
best_run = hybrid_runs[best_hybrid_name]

best_y_true = best_run["y_true"]
best_y_pred = best_run["y_pred"]
best_probabilities = best_run["probabilities"]

# Save genuine held-out predictions for the dashboard
prediction_data = {
    "y_true": best_y_true,
    "y_pred": best_y_pred
}

for class_id, class_name in enumerate(classes):

    safe_name = "".join(
        char if char.isalnum() else "_"
        for char in str(class_name)
    )

    prediction_data[f"prob_{safe_name}"] = (
        best_probabilities[:, class_id]
    )

predictions_path = os.path.join(
    RESULTS_DIR,
    "strict_predictions.csv"
)

pd.DataFrame(prediction_data).to_csv(
    predictions_path,
    index=False
)

# Confusion matrix: rows are actual labels; columns are predictions
cm = confusion_matrix(
    best_y_true,
    best_y_pred,
    labels=np.arange(N_CLASSES)
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)

cm_df.index.name = "Actual"
cm_df.columns.name = "Predicted"

confusion_path = os.path.join(
    RESULTS_DIR,
    "strict_confusion_matrix.csv"
)

cm_df.to_csv(confusion_path)

print("\nBest hybrid confusion matrix:")
print(cm_df)

print("\nSaved prediction file:", predictions_path)
print("Saved confusion matrix:", confusion_path)

# Save class IDs so the dashboard uses the correct class order
class_mapping_path = os.path.join(
    RESULTS_DIR,
    "strict_class_mapping.csv"
)

pd.DataFrame({
    "class_id": np.arange(N_CLASSES),
    "class_name": classes
}).to_csv(class_mapping_path, index=False)

# ================================================================
# COMPARATIVE F1 CHART
# ================================================================

plot_df = all_results.sort_values(
    by="F1",
    ascending=True
)

plt.figure(figsize=(12, 7))

bars = plt.barh(
    plot_df["Model"],
    plot_df["F1"]
)

plt.xlabel("F1 Score")
plt.ylabel("Model")
plt.title("Comparative Analysis of ML, DL and Hybrid Models")
plt.xlim(0, 1)

for bar, value in zip(bars, plot_df["F1"]):
    plt.text(
        value + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.4f}",
        va="center"
    )

plt.tight_layout()

comparison_png = os.path.join(
    RESULTS_DIR,
    "strict_ml_dl_comparison.png"
)

plt.savefig(
    comparison_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# REFERENCE-STYLE COMPARISON CHART
# ================================================================

baseline_order = [
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "MLP / FNN"
]

hybrid_order = [
    "Random Forest + MLP",
    "Gradient Boosting + MLP",
    "XGBoost + MLP",
    "Random Forest + XGBoost + MLP",
    "Random Forest + Gradient Boosting + MLP",
    "Gradient Boosting + XGBoost + MLP"
]

ordered_models = [
    name for name in baseline_order + hybrid_order
    if name in all_results["Model"].values
]

plot_df = (
    all_results.set_index("Model")
    .loc[ordered_models]
    .reset_index()
)

fig, ax = plt.subplots(figsize=(18, 10))

x = np.arange(len(plot_df))
width = 0.24

bars_precision = ax.bar(
    x - width,
    plot_df["Precision"].values,
    width,
    label="Precision",
    color="#1f77b4"
)

bars_recall = ax.bar(
    x,
    plot_df["Recall"].values,
    width,
    label="Recall",
    color="#ff7f0e"
)

bars_f1 = ax.bar(
    x + width,
    plot_df["F1"].values,
    width,
    label="F1-Score",
    color="#2ca02c"
)

number_of_baselines = sum(
    name in baseline_order for name in ordered_models
)

ax.axvspan(
    number_of_baselines - 0.5,
    len(ordered_models) - 0.5,
    color="#dceaf3",
    alpha=0.65,
    zorder=0
)

ax.axvline(
    number_of_baselines - 0.5,
    color="gray",
    linestyle="--",
    linewidth=1.2,
    alpha=0.7
)


def add_value_labels(bars_to_label):

    for bar in bars_to_label:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.012,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )


add_value_labels(bars_precision)
add_value_labels(bars_recall)
add_value_labels(bars_f1)

best_baseline_position = ordered_models.index(
    str(best_baseline["Model"])
)

best_baseline_f1 = float(best_baseline["F1"])

ax.annotate(
    f"Best Baseline\nF1 = {best_baseline_f1:.3f}",
    xy=(
        best_baseline_position + width,
        best_baseline_f1
    ),
    xytext=(
        best_baseline_position - 0.35,
        0.825
    ),
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.0
    ),
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.9
    )
)

best_hybrid_position = ordered_models.index(
    str(best_hybrid["Model"])
)

best_hybrid_f1 = float(best_hybrid["F1"])

ax.annotate(
    f"Best Hybrid\nF1 = {best_hybrid_f1:.3f}",
    xy=(
        best_hybrid_position + width,
        best_hybrid_f1
    ),
    xytext=(
        best_hybrid_position,
        0.825
    ),
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.0
    ),
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="none",
        alpha=0.9
    )
)

display_labels = {
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "MLP / FNN": "MLP/FNN",
    "Random Forest + MLP": "H1: RF + MLP",
    "Gradient Boosting + MLP": "H2: GB + MLP",
    "XGBoost + MLP": "H3: XGB + MLP",
    "Random Forest + XGBoost + MLP": "H4: RF + XGB + MLP",
    "Random Forest + Gradient Boosting + MLP": "H5: RF + GB + MLP",
    "Gradient Boosting + XGBoost + MLP": "H6: GB + XGB + MLP"
}

ax.set_xticks(x)

ax.set_xticklabels(
    [display_labels.get(name, name) for name in ordered_models],
    rotation=25,
    ha="right",
    fontsize=9
)

ax.set_title(
    "Comparative Model Performance on "
    "Smart Transportation Risk-Trend Prediction",
    fontsize=16,
    fontweight="bold",
    pad=28
)

fig.text(
    0.5,
    0.895,
    f"Precision · Recall · F1-Score | "
    f"Multimodal Dataset ({X.shape[1]} Features, {len(df)} Samples)",
    ha="center",
    fontsize=10
)

ax.set_xlabel(
    "Models / Approaches",
    fontsize=12,
    fontweight="bold",
    labelpad=15
)

ax.set_ylabel(
    "Score",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylim(0, 0.90)
ax.grid(axis="y", linestyle="-", alpha=0.25)
ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.045),
    ncol=3,
    frameon=False,
    fontsize=10
)

fig.text(
    0.05,
    0.035,
    "Models Compared\nRF · GB · XGBoost · MLP/FNN · Hybrid Candidates",
    fontsize=9,
    fontweight="bold",
    va="bottom"
)

fig.text(
    0.36,
    0.035,
    f"Key Insights\n"
    f"Best baseline: {best_baseline['Model']} "
    f"(F1 = {best_baseline_f1:.3f})\n"
    f"Best hybrid: {best_hybrid['Model']} "
    f"(F1 = {best_hybrid_f1:.3f})\n"
    f"Improvement: {improvement:+.4f} F1",
    fontsize=9,
    va="bottom"
)

fig.text(
    0.73,
    0.035,
    f"Notes\n"
    f"• Same multimodal dataset\n"
    f"• {len(df)} samples, {X.shape[1]} independent features\n"
    f"• {N_SPLITS}-fold GroupKFold by State\n"
    f"• Hybrid = weighted probability fusion",
    fontsize=9,
    va="bottom"
)

plt.subplots_adjust(
    left=0.055,
    right=0.985,
    top=0.83,
    bottom=0.25
)

reference_png = os.path.join(
    RESULTS_DIR,
    "strict_ml_dl_comparison_reference.png"
)

plt.savefig(
    reference_png,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ================================================================
# FINISHED
# ================================================================

print("\n" + "=" * 70)
print("RESULT FILES")
print("=" * 70)

for path in [
    comparison_csv,
    hybrid_csv,
    baseline_csv,
    selection_csv,
    comparison_png,
    reference_png,
    predictions_path,
    confusion_path,
    class_mapping_path
]:
    print(path)

print("\nExperiment completed.")