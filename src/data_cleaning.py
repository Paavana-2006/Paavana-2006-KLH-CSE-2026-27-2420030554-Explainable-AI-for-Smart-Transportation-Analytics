from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

def clean_accidents():
    path = RAW_DIR / "road_accidents.csv"
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    text_cols = ["State"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    numeric_cols = [c for c in df.columns if c != "State"]
    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["State"]).drop_duplicates()
    df.to_csv(CLEAN_DIR / "road_accidents_clean.csv", index=False)
    return df

def clean_transport():
    path = RAW_DIR / "transport.csv"
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    if "City Name" in df.columns:
        df["City Name"] = df["City Name"].astype(str).str.strip()

    for col in df.columns:
        if col != "City Name" and col != "Type Of Bus (Ac / Non Ac)":
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["City Name"]).drop_duplicates()
    df.to_csv(CLEAN_DIR / "transport_clean.csv", index=False)
    return df

def clean_weather():
    path = RAW_DIR / "weather.csv"
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    df["city"] = df["city"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in df.columns:
        if col not in ["city", "date"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["city", "date"]).drop_duplicates()
    df.to_csv(CLEAN_DIR / "weather_clean.csv", index=False)
    return df

if __name__ == "__main__":
    a = clean_accidents()
    t = clean_transport()
    w = clean_weather()
    print("Cleaning completed")
    print("Road accidents:", a.shape)
    print("Transport:", t.shape)
    print("Weather:", w.shape)
