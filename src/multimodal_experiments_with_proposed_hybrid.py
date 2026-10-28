from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FILES
# ============================================================

ROAD_FILE = RAW_DIR / "road_accidents.csv"
TRANSPORT_FILE = RAW_DIR / "transport.csv"
WEATHER_FILE = RAW_DIR / "weather.csv"
MAPPING_FILE = RAW_DIR / "city_state_mapping.csv"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_datasets():

    print("\nLoading datasets...")

    accidents = pd.read_csv(ROAD_FILE)
    transport = pd.read_csv(TRANSPORT_FILE)
    weather = pd.read_csv(WEATHER_FILE)
    mapping = pd.read_csv(MAPPING_FILE)

    print("Accident dataset:", accidents.shape)
    print("Transport dataset:", transport.shape)
    print("Weather dataset:", weather.shape)
    print("City-state mapping:", mapping.shape)

    return accidents, transport, weather, mapping


# ============================================================
# STANDARDIZE TEXT
# ============================================================

def clean_text(value):

    text = str(value).strip().lower()
    text = " ".join(text.split())

    # Handle common naming variants.
    replacements = {
        "madurai city": "madurai",
        "bengaluru": "bangalore",
        "bombay": "mumbai",
        "pimpri-chinchwad": "pimpri chichwad",
        "pimpri chinchwad": "pimpri chichwad",
    }

    return replacements.get(text, text)


# ============================================================
# GEOGRAPHIC HARMONIZATION
# ============================================================

def harmonize_geography(transport, weather, mapping):

    print("\nPerforming geographic harmonization...")

    transport = transport.copy()
    weather = weather.copy()
    mapping = mapping.copy()

    transport["City Key"] = transport["City Name"].map(clean_text)
    weather["City Key"] = weather["city"].map(clean_text)
    mapping["City Key"] = mapping["City Name"].map(clean_text)

    # Remove duplicate normalized mapping keys.
    mapping = mapping.drop_duplicates(
        subset=["City Key"],
        keep="first"
    )

    transport_cities = set(
        transport["City Key"].dropna()
    )

    weather_cities = set(
        weather["City Key"].dropna()
    )

    mapping_cities = set(
        mapping["City Key"].dropna()
    )

    missing_cities = sorted(
        (transport_cities | weather_cities) - mapping_cities
    )

    if missing_cities:

        print("\nERROR: The following cities do not have a state mapping:")

        for city in missing_cities:
            print(" -", city)

        print("\nAdd these cities to:")
        print(MAPPING_FILE)

        raise ValueError(
            "Geographic harmonization incomplete. "
            "Missing city-state mappings."
        )

    transport = transport.merge(
        mapping[["City Key", "State"]],
        on="City Key",
        how="left"
    )

    weather = weather.merge(
        mapping[["City Key", "State"]],
        on="City Key",
        how="left"
    )

    if transport["State"].isna().any():
        raise ValueError(
            "Some transport records still have missing state values."
        )

    if weather["State"].isna().any():
        raise ValueError(
            "Some weather records still have missing state values."
        )

    print("Geographic harmonization completed.")
    print("Transport states:", transport["State"].nunique())
    print("Weather states:", weather["State"].nunique())

    return transport, weather


# ============================================================
# TEMPORAL HARMONIZATION
# ============================================================

def harmonize_time(accidents, weather):

    print("\nPerforming temporal harmonization...")

    accidents = accidents.copy()
    weather = weather.copy()

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce"
    )

    if weather["date"].isna().any():
        raise ValueError(
            "Weather dataset contains invalid dates."
        )

    weather["Year"] = weather["date"].dt.year

    accident_years = [2019, 2020, 2021, 2022, 2023]

    long_records = []

    for _, row in accidents.iterrows():

        state = str(row["State"]).strip()

        for year in accident_years:

            column = f"{year} Accidents"

            if column in accidents.columns:

                long_records.append(
                    {
                        "State": state,
                        "Year": year,
                        "Accidents": pd.to_numeric(
                            row[column],
                            errors="coerce"
                        )
                    }
                )

    accident_long = pd.DataFrame(long_records)

    accident_long = accident_long.dropna(
        subset=["Accidents"]
    )

    weather_yearly = (
        weather
        .groupby(["State", "Year"], as_index=False)
        .agg(
            mean_temperature=("temperature_2m_max", "mean"),
            mean_min_temperature=("temperature_2m_min", "mean"),
            mean_precipitation=("precipitation_sum", "mean"),
            total_rain=("rain_sum", "sum"),
            max_rain=("rain_sum", "max"),
            mean_wind=("wind_speed_10m_max", "mean"),
            max_wind=("wind_speed_10m_max", "max"),
            rainy_days=(
                "rain_sum",
                lambda x: (x > 0).sum()
            ),
            extreme_rain_days=(
                "rain_sum",
                lambda x: (x >= 20).sum()
            )
        )
    )

    print("Temporal aggregation completed.")

    return accident_long, weather_yearly


# ============================================================
# TRANSPORT FEATURE ENGINEERING
# ============================================================

def create_transport_features(transport):

    print("\nEngineering transport features...")

    transport = transport.copy()

    numeric_columns = [
        "No. of buses of that type",
        "No. of bus terminals",
        "No. of bus stands",
        "No. of bus stops"
    ]

    for column in numeric_columns:

        transport[column] = pd.to_numeric(
            transport[column],
            errors="coerce"
        ).fillna(0)

    transport["bus_type"] = (
        transport["Type Of Bus (Ac / Non Ac)"]
        .fillna("Unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Correct handling of "Non AC".
    transport["is_ac"] = (
        transport["bus_type"]
        .str.contains("ac", regex=False)
        &
        ~transport["bus_type"].str.contains(
            "non ac",
            regex=False
        )
    ).astype(int)

    state_transport = (
        transport
        .groupby("State", as_index=False)
        .agg(
            total_buses=(
                "No. of buses of that type",
                "sum"
            ),
            total_bus_terminals=(
                "No. of bus terminals",
                "max"
            ),
            total_bus_stands=(
                "No. of bus stands",
                "max"
            ),
            total_bus_stops=(
                "No. of bus stops",
                "max"
            ),
            ac_bus_records=(
                "is_ac",
                "sum"
            )
        )
    )

    state_transport["buses_per_stop"] = (
        state_transport["total_buses"]
        /
        state_transport["total_bus_stops"].replace(
            0, np.nan
        )
    )

    state_transport["buses_per_terminal"] = (
        state_transport["total_buses"]
        /
        state_transport["total_bus_terminals"].replace(
            0, np.nan
        )
    )

    state_transport = state_transport.fillna(0)

    print("Transport feature engineering completed.")

    return state_transport


# ============================================================
# ACCIDENT FEATURE ENGINEERING
# ============================================================

def create_accident_features(accident_long):

    print("\nEngineering accident features...")

    accident_long = (
        accident_long
        .sort_values(["State", "Year"])
        .copy()
    )

    grouped = accident_long.groupby("State")["Accidents"]

    accident_long["previous_year_accidents"] = grouped.shift(1)

    accident_long["accident_yoy_change"] = (
        (
            accident_long["Accidents"]
            -
            accident_long["previous_year_accidents"]
        )
        /
        accident_long["previous_year_accidents"].replace(
            0, np.nan
        )
    ) * 100

    accident_long["rolling_accident_mean"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=5,
                min_periods=1
            )
            .mean()
        )
    )

    accident_long["accident_variation"] = (
        grouped
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=5,
                min_periods=2
            )
            .std()
        )
    )

    def classify_trend(value):

        if pd.isna(value):
            return np.nan

        if value < -5:
            return "Decreasing"

        if value > 5:
            return "Increasing"

        return "Stable"

    accident_long["risk_trend"] = (
        accident_long["accident_yoy_change"]
        .apply(classify_trend)
    )

    return accident_long


# ============================================================
# BUILD MULTIMODAL DATASET
# ============================================================

def build_multimodal_dataset(
    accident_features,
    transport_features,
    weather_features
):

    print("\nBuilding multimodal state-year dataset...")

    combined = accident_features.merge(
        transport_features,
        on="State",
        how="left"
    )

    combined = combined.merge(
        weather_features,
        on=["State", "Year"],
        how="left"
    )

    numeric_columns = [
        column
        for column in combined.columns
        if column not in ["State", "risk_trend"]
    ]

    for column in numeric_columns:

        combined[column] = pd.to_numeric(
            combined[column],
            errors="coerce"
        )

    combined = combined[
        combined["risk_trend"].notna()
    ].copy()

    feature_columns = [
        column
        for column in combined.columns
        if column not in [
            "State",
            "Year",
            "risk_trend"
        ]
    ]

    combined[feature_columns] = (
        combined[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    print(
        "Multimodal dataset shape:",
        combined.shape
    )

    return combined


# ============================================================
# SAVE HARMONIZATION SUMMARY
# ============================================================

def save_harmonization_summary(combined):

    summary = pd.DataFrame(
        {
            "Item": [
                "Geographic level",
                "Temporal level",
                "Accident source",
                "Transport source",
                "Weather source",
                "Final observation level",
                "Target",
                "Geographic harmonization",
                "Temporal harmonization",
                "Leakage prevention"
            ],
            "Description": [
                "State",
                "Year",
                "State-level annual accident data",
                "City-level transport aggregated to State",
                "City-level daily weather aggregated to State-Year",
                "State-Year",
                "Accident trend class",
                "City names normalized and mapped to State",
                "Weather dates converted to Year and aggregated",
                "Current-year accident count is not used as a predictor"
            ]
        }
    )

    summary.to_csv(
        RESULTS_DIR / "harmonization_summary.csv",
        index=False
    )

    print("\nSaved: harmonization_summary.csv")


# ============================================================
# DEFINE MODELS
# ============================================================

def get_models():

    models = {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=RANDOM_STATE, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (Literature Comparator)": Pipeline([("scaler", StandardScaler()), ("model", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE))]),
        "FNN/MLP (Literature Comparator)": Pipeline([("scaler", StandardScaler()), ("model", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.0001, learning_rate_init=0.001, max_iter=1000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=RANDOM_STATE))])
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss", random_state=RANDOM_STATE, objective="multi:softmax", num_class=3, tree_method="hist")

    return models


MODEL_ROLES = {
    "Logistic Regression": "Baseline",
    "Decision Tree": "Baseline",
    "Random Forest": "Baseline",
    "Gradient Boosting": "Baseline",
    "XGBoost": "Baseline",
    "SVM (Literature Comparator)": "Literature-derived comparator",
    "FNN/MLP (Literature Comparator)": "Literature-derived comparator",
}

MODEL_SOURCES = {
    "SVM (Literature Comparator)": "2025 IEEE IC2E3: Enhanced SVM for road accident prediction in India; standard SVM comparator, not exact reproduction.",
    "FNN/MLP (Literature Comparator)": "2026 Springer: FNN-based crash-severity prediction with SHAP; tabular FNN comparator, not exact reproduction.",
}


# ============================================================
# SELECT MODALITY FEATURES
# ============================================================

def get_modality_features(dataset, modality):

    accident_features = [
        "previous_year_accidents",
        "rolling_accident_mean",
        "accident_variation"
    ]

    transport_features = [
        "total_buses",
        "total_bus_terminals",
        "total_bus_stands",
        "total_bus_stops",
        "ac_bus_records",
        "buses_per_stop",
        "buses_per_terminal"
    ]

    weather_features = [
        "mean_temperature",
        "mean_min_temperature",
        "mean_precipitation",
        "total_rain",
        "max_rain",
        "mean_wind",
        "max_wind",
        "rainy_days",
        "extreme_rain_days"
    ]

    if modality == "Accident Only":
        return [
            column
            for column in accident_features
            if column in dataset.columns
        ]

    if modality == "Transport Only":
        return [
            column
            for column in transport_features
            if column in dataset.columns
        ]

    if modality == "Weather Only":
        return [
            column
            for column in weather_features
            if column in dataset.columns
        ]

    if modality == "Multimodal":
        return [
            column
            for column in (
                accident_features
                +
                transport_features
                +
                weather_features
            )
            if column in dataset.columns
        ]

    raise ValueError("Unknown modality.")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_modality(dataset, modality):

    print(f"\n{'=' * 60}")
    print(f"MODALITY: {modality}")
    print(f"{'=' * 60}")

    features = get_modality_features(dataset, modality)
    if not features:
        raise ValueError(f"No features available for {modality}")

    required_columns = features + ["risk_trend", "State"]
    data = dataset[required_columns].dropna().copy()
    X = data[features]
    y_original = data["risk_trend"].astype(str)
    groups = data["State"].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y_original), index=y_original.index, name="risk_trend_encoded")

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    if n_splits < 2:
        raise ValueError("Not enough states for grouped cross-validation.")

    cv = GroupKFold(n_splits=n_splits)
    models = get_models()
    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"\n{model_name}")
        start_time = time.perf_counter()

        predictions_encoded = cross_val_predict(model, X, y_encoded, cv=cv, groups=groups, method="predict")
        predictions_encoded = np.asarray(predictions_encoded).astype(int)
        predictions = label_encoder.inverse_transform(predictions_encoded)

        roc_auc = np.nan
        try:
            probabilities = cross_val_predict(model, X, y_encoded, cv=cv, groups=groups, method="predict_proba")
            roc_auc = roc_auc_score(y_encoded, probabilities, multi_class="ovr", average="weighted")
        except Exception as error:
            print("ROC-AUC unavailable:", error)

        elapsed = time.perf_counter() - start_time
        accuracy = accuracy_score(y_original, predictions)
        precision = precision_score(y_original, predictions, average="weighted", zero_division=0)
        recall = recall_score(y_original, predictions, average="weighted", zero_division=0)
        f1 = f1_score(y_original, predictions, average="weighted", zero_division=0)
        role = MODEL_ROLES.get(model_name, "Comparator")
        source = MODEL_SOURCES.get(model_name, "")

        print("Role     :", role)
        print("Accuracy :", f"{accuracy:.4f}")
        print("Precision:", f"{precision:.4f}")
        print("Recall   :", f"{recall:.4f}")
        print("F1 Score :", f"{f1:.4f}")
        print("ROC-AUC  :", f"{roc_auc:.4f}" if pd.notna(roc_auc) else "N/A")
        print("Time     :", f"{elapsed:.4f} sec")

        results.append({
            "Modality": modality,
            "Model": model_name,
            "Category": role,
            "Literature Source": source,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc,
            "Features": len(features),
            "Samples": len(data),
            "CV Folds": n_splits,
            "CV Protocol": "GroupKFold by State",
            "Execution Time": elapsed
        })

        model.fit(X, y_encoded)
        trained_models[model_name] = model

    return pd.DataFrame(results), trained_models, X, y_encoded, y_original, groups, features, label_encoder


# ============================================================
# OUT-OF-FOLD PERMUTATION IMPORTANCE
# ============================================================

def calculate_permutation_importance_oof(
    model,
    X,
    y_encoded,
    groups,
    features,
    n_splits
):

    print(
        "\nCalculating grouped out-of-fold permutation importance..."
    )

    cv = GroupKFold(
        n_splits=n_splits
    )

    records = []

    for fold_number, (train_index, test_index) in enumerate(
        cv.split(X, y_encoded, groups),
        start=1
    ):

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y_encoded.iloc[train_index]
        y_test = y_encoded.iloc[test_index]

        fold_model = get_models()["Random Forest"]

        fold_model.fit(
            X_train,
            y_train
        )

        baseline_predictions = fold_model.predict(
            X_test
        )

        baseline_score = f1_score(
            y_test,
            baseline_predictions,
            average="weighted",
            zero_division=0
        )

        for feature in features:

            X_permuted = X_test.copy()

            X_permuted[feature] = np.random.default_rng(
                RANDOM_STATE + fold_number
            ).permutation(
                X_permuted[feature].values
            )

            permuted_predictions = fold_model.predict(
                X_permuted
            )

            permuted_score = f1_score(
                y_test,
                permuted_predictions,
                average="weighted",
                zero_division=0
            )

            records.append(
                {
                    "Fold": fold_number,
                    "Feature": feature,
                    "Importance": (
                        baseline_score
                        -
                        permuted_score
                    )
                }
            )

    fold_df = pd.DataFrame(records)

    permutation_df = (
        fold_df
        .groupby("Feature", as_index=False)
        .agg(
            **{
                "Permutation Importance Mean": (
                    "Importance",
                    "mean"
                ),
                "Permutation Importance Std": (
                    "Importance",
                    "std"
                )
            }
        )
        .sort_values(
            "Permutation Importance Mean",
            ascending=False
        )
    )

    permutation_df.to_csv(
        RESULTS_DIR / "permutation_importance.csv",
        index=False
    )

    print(
        "Saved: permutation_importance.csv"
    )

    return permutation_df


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def calculate_feature_importance(model, features):

    print(
        "\nCalculating global feature importance..."
    )

    if hasattr(model, "feature_importances_"):

        importance_values = (
            model.feature_importances_
        )

    elif hasattr(model, "named_steps"):

        final_model = model.named_steps["model"]

        if hasattr(final_model, "coef_"):

            importance_values = np.mean(
                np.abs(final_model.coef_),
                axis=0
            )

        else:
            return None

    else:
        return None

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": importance_values
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    importance_df.to_csv(
        RESULTS_DIR / "multimodal_feature_importance.csv",
        index=False
    )

    print(
        "Saved: multimodal_feature_importance.csv"
    )

    return importance_df



# ============================================================
# ============================================================
# MULTIPLE HYBRID CANDIDATE EVALUATION
# ============================================================

def evaluate_hybrid_candidates(
    dataset,
    features,
    label_encoder
):

    print("\n" + "=" * 70)
    print("MULTIPLE HYBRID CANDIDATE EVALUATION")
    print("=" * 70)

    data = dataset[
        features + ["risk_trend", "State"]
    ].dropna().copy()

    X = data[features].reset_index(drop=True)

    y_original = (
        data["risk_trend"]
        .astype(str)
        .reset_index(drop=True)
    )

    groups = (
        data["State"]
        .astype(str)
        .reset_index(drop=True)
    )

    y = pd.Series(
        label_encoder.transform(y_original),
        name="risk_trend_encoded"
    )

    n_splits = min(5, groups.nunique())

    if n_splits < 2:
        raise ValueError(
            "Not enough states for grouped cross-validation."
        )

    outer_cv = GroupKFold(
        n_splits=n_splits
    )

    class_count = len(
        label_encoder.classes_
    )

    # ========================================================
    # HYBRID CANDIDATES
    # ========================================================

    candidate_definitions = {

        "Hybrid 1: RF + GB -> LR": (
            [
                "Random Forest",
                "Gradient Boosting"
            ],
            "Logistic Regression"
        ),

        "Hybrid 2: RF + XGB -> LR": (
            [
                "Random Forest",
                "XGBoost"
            ],
            "Logistic Regression"
        ),

        "Hybrid 3: GB + XGB -> LR": (
            [
                "Gradient Boosting",
                "XGBoost"
            ],
            "Logistic Regression"
        ),

        "Hybrid 4: RF + GB + XGB -> LR": (
            [
                "Random Forest",
                "Gradient Boosting",
                "XGBoost"
            ],
            "Logistic Regression"
        ),

        "Hybrid 5: RF + SVM -> LR": (
            [
                "Random Forest",
                "SVM (Literature Comparator)"
            ],
            "Logistic Regression"
        ),

        "Hybrid 6: RF + XGB -> XGB": (
            [
                "Random Forest",
                "XGBoost"
            ],
            "XGBoost"
        ),

        "Hybrid 7: RF + GB + XGB -> XGB": (
            [
                "Random Forest",
                "Gradient Boosting",
                "XGBoost"
            ],
            "XGBoost"
        ),

        "Hybrid 8: RF + GB -> XGB": (
            [
                "Random Forest",
                "Gradient Boosting"
            ],
            "XGBoost"
        )
    }

    # ========================================================
    # MODEL CREATION
    # ========================================================

    def make_model(model_name):

        model = get_models().get(model_name)

        if model is None:
            raise ValueError(
                f"Model '{model_name}' is unavailable."
            )

        return model

    # ========================================================
    # PROBABILITY ALIGNMENT
    # ========================================================

    def aligned_predict_proba(
        model,
        X_part
    ):

        probabilities = model.predict_proba(
            X_part
        )

        aligned = np.zeros(
            (
                len(X_part),
                class_count
            ),
            dtype=float
        )

        for column_index, class_value in enumerate(
            model.classes_
        ):

            aligned[
                :,
                int(class_value)
            ] = probabilities[
                :,
                column_index
            ]

        return aligned

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    all_candidate_results = []
    all_fold_results = []

    # ========================================================
    # EVALUATE EACH HYBRID
    # ========================================================

    for candidate_name, candidate_config in (
        candidate_definitions.items()
    ):

        base_names, meta_model_name = candidate_config

        print("\n" + "-" * 70)
        print(candidate_name)
        print(
            "Base Models:",
            " + ".join(base_names)
        )
        print(
            "Meta Model:",
            meta_model_name
        )
        print("-" * 70)

        all_pred = np.full(
            len(X),
            -1,
            dtype=int
        )

        all_prob = np.zeros(
            (
                len(X),
                class_count
            ),
            dtype=float
        )

        fold_records = []

        start_time = time.perf_counter()

        # ====================================================
        # OUTER GROUP K-FOLD
        # ====================================================

        for fold_number, (
            train_idx,
            test_idx
        ) in enumerate(
            outer_cv.split(
                X,
                y,
                groups
            ),
            start=1
        ):

            X_train = (
                X.iloc[train_idx]
                .reset_index(drop=True)
            )

            X_test = (
                X.iloc[test_idx]
                .reset_index(drop=True)
            )

            y_train = (
                y.iloc[train_idx]
                .reset_index(drop=True)
            )

            y_test = (
                y.iloc[test_idx]
                .reset_index(drop=True)
            )

            g_train = (
                groups.iloc[train_idx]
                .reset_index(drop=True)
            )

            # =================================================
            # INNER GROUP K-FOLD
            # =================================================

            inner_splits = min(
                4,
                g_train.nunique()
            )

            if inner_splits < 2:
                raise ValueError(
                    "Not enough training groups for "
                    "inner grouped stacking."
                )

            inner_cv = GroupKFold(
                n_splits=inner_splits
            )

            # =================================================
            # META TRAIN / TEST MATRICES
            # =================================================

            meta_train = np.zeros(
                (
                    len(X_train),
                    len(base_names) * class_count
                ),
                dtype=float
            )

            meta_test = np.zeros(
                (
                    len(X_test),
                    len(base_names) * class_count
                ),
                dtype=float
            )

            # =================================================
            # TRAIN BASE MODELS
            # =================================================

            for base_index, base_name in enumerate(
                base_names
            ):

                base_oof = np.zeros(
                    (
                        len(X_train),
                        class_count
                    ),
                    dtype=float
                )

                # ---------------------------------------------
                # INNER OOF PREDICTIONS
                # ---------------------------------------------

                for (
                    inner_train_idx,
                    inner_valid_idx
                ) in inner_cv.split(
                    X_train,
                    y_train,
                    g_train
                ):

                    base_model = make_model(
                        base_name
                    )

                    base_model.fit(
                        X_train.iloc[
                            inner_train_idx
                        ],
                        y_train.iloc[
                            inner_train_idx
                        ]
                    )

                    base_oof[
                        inner_valid_idx
                    ] = aligned_predict_proba(
                        base_model,
                        X_train.iloc[
                            inner_valid_idx
                        ]
                    )

                # ---------------------------------------------
                # STORE OOF PREDICTIONS
                # ---------------------------------------------

                start_col = (
                    base_index * class_count
                )

                end_col = (
                    start_col + class_count
                )

                meta_train[
                    :,
                    start_col:end_col
                ] = base_oof

                # ---------------------------------------------
                # TRAIN BASE MODEL ON FULL OUTER TRAIN SET
                # ---------------------------------------------

                final_base = make_model(
                    base_name
                )

                final_base.fit(
                    X_train,
                    y_train
                )

                meta_test[
                    :,
                    start_col:end_col
                ] = aligned_predict_proba(
                    final_base,
                    X_test
                )

            # =================================================
            # META MODEL
            # =================================================

            if meta_model_name == "Logistic Regression":

                meta_model = LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )

            elif meta_model_name == "XGBoost":

                if not XGBOOST_AVAILABLE:

                    raise ImportError(
                        "XGBoost is required for this "
                        "hybrid candidate."
                    )

                meta_model = XGBClassifier(
                    n_estimators=100,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    objective="multi:softprob",
                    num_class=class_count,
                    tree_method="hist"
                )

            else:

                raise ValueError(
                    f"Unknown meta model: "
                    f"{meta_model_name}"
                )

            # =================================================
            # TRAIN META MODEL
            # =================================================

            meta_model.fit(
                meta_train,
                y_train
            )

            # =================================================
            # PREDICT OUTER TEST SET
            # =================================================

            test_predictions = (
                meta_model.predict(
                    meta_test
                )
            )

            test_probabilities = (
                meta_model.predict_proba(
                    meta_test
                )
            )

            all_pred[
                test_idx
            ] = test_predictions

            all_prob[
                test_idx
            ] = test_probabilities

            # =================================================
            # FOLD F1
            # =================================================

            fold_f1 = f1_score(
                y_test,
                test_predictions,
                average="weighted",
                zero_division=0
            )

            fold_records.append(
                {
                    "Candidate": candidate_name,
                    "Fold": fold_number,
                    "Test Samples": len(test_idx),
                    "F1 Score": fold_f1
                }
            )

            print(
                f"Fold {fold_number}: "
                f"samples={len(test_idx)}, "
                f"F1={fold_f1:.4f}"
            )

        # ====================================================
        # OVERALL METRICS
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start_time
        )

        accuracy = accuracy_score(
            y,
            all_pred
        )

        precision = precision_score(
            y,
            all_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y,
            all_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y,
            all_pred,
            average="weighted",
            zero_division=0
        )

        try:

            roc_auc = roc_auc_score(
                y,
                all_prob,
                multi_class="ovr",
                average="weighted"
            )

        except Exception:

            roc_auc = np.nan

        fold_scores = [
            record["F1 Score"]
            for record in fold_records
        ]

        fold_mean = float(
            np.mean(fold_scores)
        )

        fold_std = (
            float(
                np.std(
                    fold_scores,
                    ddof=1
                )
            )
            if len(fold_scores) > 1
            else 0.0
        )

        # ====================================================
        # STORE CANDIDATE RESULT
        # ====================================================

        all_candidate_results.append(
            {
                "Modality": "Multimodal",
                "Model": candidate_name,
                "Category": "Hybrid candidate",
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
                "Features": len(features),
                "Samples": len(data),
                "CV Folds": n_splits,
                "CV Protocol": "Nested GroupKFold by State",
                "Execution Time": elapsed,
                "Mean Fold F1": fold_mean,
                "Std Fold F1": fold_std,
                "Base Models": " + ".join(base_names),
                "Meta Model": meta_model_name
            }
        )

        all_fold_results.extend(
            fold_records
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        predicted_labels = (
            label_encoder.inverse_transform(
                all_pred
            )
        )

        actual_labels = (
            label_encoder.inverse_transform(
                y.to_numpy()
            )
        )

        confusion = pd.crosstab(
            pd.Series(
                actual_labels,
                name="Actual"
            ),
            pd.Series(
                predicted_labels,
                name="Predicted"
            ),
            dropna=False
        )

        class_order = list(
            label_encoder.classes_
        )

        confusion = confusion.reindex(
            index=class_order,
            columns=class_order,
            fill_value=0
        )

        safe_name = (
            candidate_name
            .lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("->", "to")
            .replace("/", "_")
        )

        confusion.to_csv(
            RESULTS_DIR /
            f"{safe_name}_confusion_matrix.csv"
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "Accuracy :",
            f"{accuracy:.4f}"
        )

        print(
            "Precision:",
            f"{precision:.4f}"
        )

        print(
            "Recall   :",
            f"{recall:.4f}"
        )

        print(
            "F1 Score :",
            f"{f1:.4f}"
        )

        print(
            "ROC-AUC  :",
            f"{roc_auc:.4f}"
            if pd.notna(roc_auc)
            else "N/A"
        )

        print(
            "Mean Fold F1:",
            f"{fold_mean:.4f}"
        )

        print(
            "Std Fold F1 :",
            f"{fold_std:.4f}"
        )

        print(
            "Time     :",
            f"{elapsed:.4f} sec"
        )

    # ========================================================
    # CREATE RESULT DATAFRAMES
    # ========================================================

    candidate_df = pd.DataFrame(
        all_candidate_results
    )

    fold_df = pd.DataFrame(
        all_fold_results
    )

    # ========================================================
    # SORT BY F1 AND ROC-AUC
    # ========================================================

    candidate_df = (
        candidate_df
        .sort_values(
            by=[
                "F1 Score",
                "ROC-AUC"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    candidate_df["F1 Rank"] = (
        candidate_df[
            "F1 Score"
        ]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    candidate_df.to_csv(
        RESULTS_DIR /
        "hybrid_candidate_comparison.csv",
        index=False
    )

    candidate_df.to_csv(
        RESULTS_DIR /
        "proposed_model_comparison.csv",
        index=False
    )

    fold_df.to_csv(
        RESULTS_DIR /
        "hybrid_candidate_fold_results.csv",
        index=False
    )

    selection_summary = candidate_df[
        [
            "Model",
            "F1 Score",
            "ROC-AUC",
            "Accuracy",
            "Precision",
            "Recall",
            "Mean Fold F1",
            "Std Fold F1",
            "Execution Time",
            "Base Models",
            "Meta Model"
        ]
    ].copy()

    selection_summary.to_csv(
        RESULTS_DIR /
        "hybrid_selection_summary.csv",
        index=False
    )

    # ========================================================
    # PRINT COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("HYBRID CANDIDATE COMPARISON")
    print("=" * 70)

    print(
        candidate_df.to_string(
            index=False
        )
    )

    print(
        "\nBest candidate by F1 "
        "(NOT automatically the final proposed model):"
    )

    print(
        candidate_df.iloc[0][
            [
                "Model",
                "F1 Score",
                "ROC-AUC"
            ]
        ].to_string()
    )

    print("\nSaved:")
    print(
        "- hybrid_candidate_comparison.csv"
    )
    print(
        "- proposed_model_comparison.csv"
    )
    print(
        "- hybrid_candidate_fold_results.csv"
    )
    print(
        "- hybrid_selection_summary.csv"
    )

    return candidate_df



# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTIMODAL SMART TRANSPORTATION EXPERIMENT"
    )

    print(
        "=" * 70
    )

    (
        accidents,
        transport,
        weather,
        mapping
    ) = load_datasets()

    transport, weather = harmonize_geography(
        transport,
        weather,
        mapping
    )

    accident_long, weather_yearly = harmonize_time(
        accidents,
        weather
    )

    accident_features = create_accident_features(
        accident_long
    )

    transport_features = create_transport_features(
        transport
    )

    multimodal = build_multimodal_dataset(
        accident_features,
        transport_features,
        weather_yearly
    )

    multimodal_file = (
        PROCESSED_DIR /
        "multimodal_state_year_dataset.csv"
    )

    multimodal.to_csv(
        multimodal_file,
        index=False
    )

    print(
        "\nSaved:",
        multimodal_file
    )

    save_harmonization_summary(
        multimodal
    )

    modalities = [
        "Accident Only",
        "Transport Only",
        "Weather Only",
        "Multimodal"
    ]

    all_results = []

    final_X = None
    final_y_encoded = None
    final_y_original = None
    final_groups = None
    final_features = None
    final_label_encoder = None
    final_n_splits = None
    final_multimodal_model = None

    for modality in modalities:

        try:

            (
                result_df,
                trained_models,
                X,
                y_encoded,
                y_original,
                groups,
                features,
                label_encoder
            ) = evaluate_modality(
                multimodal,
                modality
            )

            all_results.append(result_df)

            if modality == "Multimodal":

                final_multimodal_model = (
                    trained_models.get(
                        "Random Forest"
                    )
                )

                final_X = X
                final_y_encoded = y_encoded
                final_y_original = y_original
                final_groups = groups
                final_features = features
                final_label_encoder = label_encoder

                final_n_splits = min(
                    5,
                    groups.nunique()
                )

        except Exception as error:

            print(
                f"\n{modality} experiment failed:"
            )

            print(error)

    proposed_metrics = None

    if final_X is not None and final_label_encoder is not None:
        proposed_metrics = evaluate_hybrid_candidates(
            multimodal,
            final_features,
            final_label_encoder
        )

    if all_results:

        comparison_df = pd.concat(
            all_results,
            ignore_index=True
        )

        comparison_df.to_csv(
            RESULTS_DIR /
            "modality_comparison.csv",
            index=False
        )

        print(
            "\nSaved: modality_comparison.csv"
        )

        print(
            "\nMODALITY COMPARISON"
        )

        print(
            comparison_df.to_string(
                index=False
            )
        )

        comparison_df[comparison_df["Category"] == "Literature-derived comparator"].to_csv(RESULTS_DIR / "literature_model_comparison.csv", index=False)
        comparison_df[comparison_df["Category"] == "Baseline"].to_csv(RESULTS_DIR / "baseline_model_comparison.csv", index=False)
        print("\nSaved: literature_model_comparison.csv")
        print("Saved: baseline_model_comparison.csv")

        if proposed_metrics is not None:
            final_comparison = pd.concat(
                [comparison_df, proposed_metrics],
                ignore_index=True,
                sort=False
            )
            final_comparison.to_csv(
                RESULTS_DIR / "final_model_comparison.csv",
                index=False
            )
            print("Saved: final_model_comparison.csv")

    if (
        final_multimodal_model is not None
        and final_X is not None
    ):

        calculate_permutation_importance_oof(
            final_multimodal_model,
            final_X,
            final_y_encoded,
            final_groups,
            final_features,
            final_n_splits
        )

        calculate_feature_importance(
            final_multimodal_model,
            final_features
        )

        joblib.dump(
            final_multimodal_model,
            MODELS_DIR /
            "multimodal_random_forest.joblib"
        )

        joblib.dump(
            final_label_encoder,
            MODELS_DIR /
            "multimodal_risk_label_encoder.joblib"
        )

        print(
            "\nSaved multimodal Random Forest model."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTIMODAL EXPERIMENT COMPLETED"
    )

    print(
        "=" * 70
    )

    print("\nGenerated files:")

    print(
        "1.",
        PROCESSED_DIR /
        "multimodal_state_year_dataset.csv"
    )

    print(
        "2.",
        RESULTS_DIR /
        "harmonization_summary.csv"
    )

    print(
        "3.",
        RESULTS_DIR /
        "modality_comparison.csv"
    )

    print(
        "4.",
        RESULTS_DIR /
        "permutation_importance.csv"
    )

    print(
        "5.",
        RESULTS_DIR /
        "multimodal_feature_importance.csv"
    )

    print(
        "6.",
        MODELS_DIR /
        "multimodal_random_forest.joblib"
    )

    print(
        "7.",
        MODELS_DIR /
        "multimodal_risk_label_encoder.joblib"
    )

    print(
        "8.",
        RESULTS_DIR /
        "literature_model_comparison.csv"
    )

    print(
        "9.",
        RESULTS_DIR /
        "baseline_model_comparison.csv"
    )

    print(
        "10.",
        RESULTS_DIR /
        "proposed_model_comparison.csv"
    )

    print(
        "11.",
        RESULTS_DIR /
        "proposed_hybrid_confusion_matrix.csv"
    )

    print(
        "12.",
        RESULTS_DIR /
        "proposed_hybrid_fold_results.csv"
    )

    print(
        "13.",
        RESULTS_DIR /
        "final_model_comparison.csv"
    )


if __name__ == "__main__":
    main()
