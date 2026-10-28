from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE_DIR / "data" / "cleaned"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def save_plots():
    accidents = pd.read_csv(CLEAN_DIR / "road_accidents_clean.csv")
    weather = pd.read_csv(CLEAN_DIR / "weather_clean.csv", parse_dates=["date"])
    transport = pd.read_csv(CLEAN_DIR / "transport_clean.csv")

    plt.figure(figsize=(12, 6))
    top = accidents.sort_values("2023 Accidents", ascending=False).head(10)
    plt.bar(top["State"], top["2023 Accidents"])
    plt.xticks(rotation=75)
    plt.title("Top 10 States by Road Accidents in 2023")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "top_states_2023.png", dpi=150)
    plt.close()

    weather_daily = weather.groupby("date", as_index=False)["temperature_2m_max"].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(weather_daily["date"], weather_daily["temperature_2m_max"])
    plt.title("Average Daily Maximum Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "temperature_trend.png", dpi=150)
    plt.close()

    city_transport = transport.groupby("City Name", as_index=False)["No. of bus stops"].sum()
    top_cities = city_transport.sort_values("No. of bus stops", ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    plt.bar(top_cities["City Name"], top_cities["No. of bus stops"])
    plt.xticks(rotation=75)
    plt.title("Top Cities by Number of Bus Stops")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bus_stops_by_city.png", dpi=150)
    plt.close()

    print("EDA plots saved in results/")

if __name__ == "__main__":
    save_plots()
