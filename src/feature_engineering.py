from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def create_accident_features():
    df = pd.read_csv(CLEAN_DIR / "road_accidents_clean.csv")

    year_cols = [
        "2019 Accidents", "2020 Accidents",
        "2021 Accidents", "2022 Accidents", "2023 Accidents"
    ]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Features use only 2019–2022 values.
    df["average_accidents"] = df[
        ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents"]
    ].mean(axis=1)

    df["total_accidents_2019_2022"] = df[
        ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents"]
    ].sum(axis=1)

    df["accident_growth_2019_2022"] = (
        (df["2022 Accidents"] - df["2019 Accidents"])
        / df["2019 Accidents"].replace(0, np.nan)
    ) * 100

    df["accident_variation"] = df[
        ["2019 Accidents", "2020 Accidents", "2021 Accidents", "2022 Accidents"]
    ].std(axis=1)

    # Proxy target created from 2023 accident counts.
    # This is an academic demonstration label, not a ground-truth incident label.
    q1 = df["2023 Accidents"].quantile(0.33)
    q2 = df["2023 Accidents"].quantile(0.66)

    def assign_risk(value):
        if value <= q1:
            return "Low"
        if value <= q2:
            return "Medium"
        return "High"

    df["risk_level"] = df["2023 Accidents"].apply(assign_risk)

    feature_cols = [
        "2019 Accidents", "2020 Accidents",
        "2021 Accidents", "2022 Accidents",
        "average_accidents", "total_accidents_2019_2022",
        "accident_growth_2019_2022", "accident_variation"
    ]

    output = df[["State", "2023 Accidents", "risk_level"] + feature_cols]
    output.to_csv(PROCESSED_DIR / "accident_ml_dataset.csv", index=False)

    print("Feature engineering completed")
    print("Output shape:", output.shape)
    print(output["risk_level"].value_counts())
    return output

if __name__ == "__main__":
    create_accident_features()
