from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ROAD_FILE = BASE_DIR / "data" / "cleaned" / "road_accidents_clean.csv"
TRANSPORT_FILE = BASE_DIR / "data" / "cleaned" / "transport_clean.csv"
WEATHER_FILE = BASE_DIR / "data" / "cleaned" / "weather_clean.csv"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DATA COMPLETENESS
# ============================================================

def completeness(df):
    total = df.shape[0] * df.shape[1]

    if total == 0:
        return 0

    return round((df.notna().sum().sum() / total) * 100, 2)


# ============================================================
# 1. ROAD ACCIDENT DATASET
# ============================================================

road = pd.read_csv(ROAD_FILE)

accident_columns = [
    "2019 Accidents",
    "2020 Accidents",
    "2021 Accidents",
    "2022 Accidents",
    "2023 Accidents"
]

for column in accident_columns:
    road[column] = pd.to_numeric(
        road[column],
        errors="coerce"
    )

road_metrics = {
    "Dataset": "Road Accident",
    "Records": len(road),
    "Features": len(road.columns),
    "Data Completeness (%)": completeness(road),
    "States Covered": road["State"].nunique(),
    "Average Accidents": round(
        road[accident_columns].mean().mean(),
        2
    ),
    "Average 2023 Accidents": round(
        road["2023 Accidents"].mean(),
        2
    ),
    "Total 2023 Accidents": int(
        road["2023 Accidents"].sum()
    )
}


# ============================================================
# 2. TRANSPORT DATASET
# ============================================================

transport = pd.read_csv(TRANSPORT_FILE)

transport_columns = [
    "No. of buses of that type",
    "No. of bus terminals",
    "No. of bus stands",
    "No. of bus stops"
]

for column in transport_columns:
    transport[column] = pd.to_numeric(
        transport[column],
        errors="coerce"
    )

transport_metrics = {
    "Dataset": "Transport",
    "Records": len(transport),
    "Features": len(transport.columns),
    "Data Completeness (%)": completeness(transport),
    "Cities Covered": transport["City Name"].nunique(),
    "Average Buses": round(
        transport["No. of buses of that type"].mean(),
        2
    ),
    "Average Bus Terminals": round(
        transport["No. of bus terminals"].mean(),
        2
    ),
    "Average Bus Stands": round(
        transport["No. of bus stands"].mean(),
        2
    ),
    "Average Bus Stops": round(
        transport["No. of bus stops"].mean(),
        2
    )
}


# ============================================================
# 3. WEATHER DATASET
# ============================================================

weather = pd.read_csv(WEATHER_FILE)

weather["date"] = pd.to_datetime(
    weather["date"],
    errors="coerce"
)

weather_columns = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max"
]

for column in weather_columns:
    weather[column] = pd.to_numeric(
        weather[column],
        errors="coerce"
    )

valid_dates = weather["date"].notna().sum()

weather_metrics = {
    "Dataset": "Weather",
    "Records": len(weather),
    "Features": len(weather.columns),
    "Data Completeness (%)": completeness(weather),
    "Cities Covered": weather["city"].nunique(),
    "Valid Date Records (%)": round(
        (valid_dates / len(weather)) * 100,
        2
    ),
    "Average Max Temperature": round(
        weather["temperature_2m_max"].mean(),
        2
    ),
    "Average Min Temperature": round(
        weather["temperature_2m_min"].mean(),
        2
    ),
    "Average Precipitation": round(
        weather["precipitation_sum"].mean(),
        2
    ),
    "Average Wind Speed": round(
        weather["wind_speed_10m_max"].mean(),
        2
    )
}


# ============================================================
# SAVE RESULTS
# ============================================================

pd.DataFrame([road_metrics]).to_csv(
    RESULTS_DIR / "road_dataset_metrics.csv",
    index=False
)

pd.DataFrame([transport_metrics]).to_csv(
    RESULTS_DIR / "transport_dataset_metrics.csv",
    index=False
)

pd.DataFrame([weather_metrics]).to_csv(
    RESULTS_DIR / "weather_dataset_metrics.csv",
    index=False
)

summary = pd.DataFrame([
    road_metrics,
    transport_metrics,
    weather_metrics
])

summary.to_csv(
    RESULTS_DIR / "dataset_performance_summary.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DATASET PERFORMANCE EVALUATION")
print("=" * 60)

print("\nROAD ACCIDENT DATASET")
print("-" * 30)

for key, value in road_metrics.items():
    print(f"{key}: {value}")

print("\nTRANSPORT DATASET")
print("-" * 30)

for key, value in transport_metrics.items():
    print(f"{key}: {value}")

print("\nWEATHER DATASET")
print("-" * 30)

for key, value in weather_metrics.items():
    print(f"{key}: {value}")

print("\nResults saved in the results folder.")