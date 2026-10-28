import requests


# ============================================================
# WEATHER CODE DESCRIPTION
# ============================================================

def weather_description(code):

    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail"
    }

    return descriptions.get(
        int(code),
        "Unknown"
    )


# ============================================================
# GET LOCATION COORDINATES
# ============================================================

def get_coordinates(city):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
        "countryCode": "IN"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        results = data.get(
            "results",
            []
        )

        if not results:
            return None

        location = results[0]

        return {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "name": location["name"],
            "state": location.get(
                "admin1",
                ""
            )
        }

    except Exception:

        return None


# ============================================================
# GET LIVE WEATHER
# ============================================================

def get_live_weather(city):

    location = get_coordinates(
        city
    )

    if location is None:

        return None

    url = "https://api.open-meteo.com/v1/forecast"

    params = {

        "latitude": location["latitude"],

        "longitude": location["longitude"],

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m,"
            "wind_gusts_10m"
        ),

        "timezone": "auto"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        current = data.get(
            "current"
        )

        if current is None:

            return None

        temperature = float(
            current.get(
                "temperature_2m",
                0
            )
        )

        humidity = float(
            current.get(
                "relative_humidity_2m",
                0
            )
        )

        precipitation = float(
            current.get(
                "precipitation",
                0
            )
        )

        rain = float(
            current.get(
                "rain",
                0
            )
        )

        wind = float(
            current.get(
                "wind_speed_10m",
                0
            )
        )

        gust = float(
            current.get(
                "wind_gusts_10m",
                0
            )
        )

        weather_code = int(
            current.get(
                "weather_code",
                0
            )
        )

        # ----------------------------------------------------
        # WEATHER RISK
        # ----------------------------------------------------

        if (
            rain >= 4
            or wind >= 40
            or gust >= 55
            or weather_code in [65, 82, 95, 96, 99]
        ):

            status = "Adverse"

        elif (
            rain > 0
            or wind >= 25
            or weather_code in [
                51,
                53,
                55,
                61,
                63,
                80,
                81
            ]
        ):

            status = "Moderate"

        else:

            status = "Normal"

        return {

            "city": location["name"],

            "state": location["state"],

            "latitude": location["latitude"],

            "longitude": location["longitude"],

            "temperature": temperature,

            "humidity": humidity,

            "precipitation": precipitation,

            "rain": rain,

            "wind": wind,

            "wind_gust": gust,

            "weather_code": weather_code,

            "condition": weather_description(
                weather_code
            ),

            "status": status
        }

    except Exception:

        return None