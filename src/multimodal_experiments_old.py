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

        "Logistic Regression":
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=RANDOM_STATE
                        )
                    )
                ]
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                max_depth=5
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=250,
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=RANDOM_STATE
            )
    }

    if XGBOOST_AVAILABLE:

        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            objective="multi:softmax",
            num_class=3,
            tree_method="hist"
        )

    return models


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

    features = get_modality_features(
        dataset,
        modality
    )

    if not features:
        raise ValueError(
            f"No features available for {modality}"
        )

    required_columns = features + [
        "risk_trend",
        "State"
    ]

    data = dataset[required_columns].dropna().copy()

    X = data[features]
    y_original = data["risk_trend"].astype(str)
    groups = data["State"].astype(str)

    print("Features:", len(features))
    print("Samples:", len(data))

    print("\nClasses:")
    print(y_original.value_counts())

    # --------------------------------------------------------
    # Encode labels for XGBoost and keep original labels
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = pd.Series(
        label_encoder.fit_transform(y_original),
        index=y_original.index,
        name="risk_trend_encoded"
    )

    print("\nLabel mapping:")

    for label, encoded in zip(
        label_encoder.classes_,
        range(len(label_encoder.classes_))
    ):
        print(
            f"{label} -> {encoded}"
        )

    # --------------------------------------------------------
    # Grouped CV by State
    # --------------------------------------------------------

    unique_groups = groups.nunique()

    n_splits = min(
        5,
        unique_groups
    )

    if n_splits < 2:
        raise ValueError(
            f"Not enough states for grouped cross-validation."
        )

    cv = GroupKFold(
        n_splits=n_splits
    )

    print(
        "\nCross-validation:",
        f"{n_splits}-fold GroupKFold by State"
    )

    models = get_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():

        print(f"\n{model_name}")

        start_time = time.perf_counter()

        predictions_encoded = cross_val_predict(
            model,
            X,
            y_encoded,
            cv=cv,
            groups=groups,
            method="predict"
        )

        predictions_encoded = np.asarray(
            predictions_encoded
        ).astype(int)

        predictions = label_encoder.inverse_transform(
            predictions_encoded
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        accuracy = accuracy_score(
            y_original,
            predictions
        )

        precision = precision_score(
            y_original,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_original,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_original,
            predictions,
            average="weighted",
            zero_division=0
        )

        print("Accuracy :", f"{accuracy:.4f}")
        print("Precision:", f"{precision:.4f}")
        print("Recall   :", f"{recall:.4f}")
        print("F1 Score :", f"{f1:.4f}")

        results.append(
            {
                "Modality": modality,
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Features": len(features),
                "Samples": len(data),
                "CV Folds": n_splits,
                "CV Protocol": "GroupKFold by State",
                "Execution Time": elapsed
            }
        )

        # Train final model on all data.
        # XGBoost receives encoded labels.
        model.fit(
            X,
            y_encoded
        )

        trained_models[model_name] = model

    return (
        pd.DataFrame(results),
        trained_models,
        X,
        y_encoded,
        y_original,
        groups,
        features,
        label_encoder
    )


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


if __name__ == "__main__":
    main()
