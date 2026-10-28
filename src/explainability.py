from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "accident_ml_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest.joblib"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "2019 Accidents", "2020 Accidents",
    "2021 Accidents", "2022 Accidents",
    "average_accidents", "total_accidents_2019_2022",
    "accident_growth_2019_2022", "accident_variation"
]

def run_explainability():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    model = joblib.load(MODEL_PATH)

    # Extract the RandomForest model from the pipeline.
    rf = model.named_steps["model"]
    X_values = X.fillna(X.median())

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_values)

        # Save a simple global importance table that works across SHAP versions.
        if isinstance(shap_values, list):
            absolute_values = np.mean(
                np.abs(np.stack(shap_values, axis=0)), axis=(0, 1)
            )
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                absolute_values = np.mean(np.abs(arr), axis=(0, 2))
            else:
                absolute_values = np.mean(np.abs(arr), axis=0)

        shap_importance = pd.DataFrame({
            "Feature": FEATURES,
            "Mean Absolute SHAP": absolute_values
        }).sort_values("Mean Absolute SHAP", ascending=False)

        shap_importance.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
        print("SHAP explanation saved.")
    except Exception as error:
        print("SHAP explanation could not be generated:", error)

    try:
        from lime.lime_tabular import LimeTabularExplainer
        explainer = LimeTabularExplainer(
            X_values.to_numpy(),
            feature_names=FEATURES,
            class_names=list(model.classes_),
            mode="classification",
            discretize_continuous=True,
            random_state=42
        )
        explanation = explainer.explain_instance(
            X_values.iloc[0].to_numpy(),
            model.predict_proba,
            num_features=len(FEATURES)
        )
        lime_df = pd.DataFrame(
            explanation.as_list(),
            columns=["Rule", "Weight"]
        )
        lime_df.to_csv(RESULTS_DIR / "lime_example.csv", index=False)
        print("LIME example saved.")
    except Exception as error:
        print("LIME explanation could not be generated:", error)

if __name__ == "__main__":
    run_explainability()
