"""
Weather Tool — fetches current weather using WeatherAPI.com.
Falls back to a helpful message if no API key is configured.
"""
import requests
from config import WEATHER_API_KEY


def get_weather(location: str) -> str:
    """
    Get current weather for a location using WeatherAPI.com.

    Args:
        location: City name or coordinates, e.g. 'Mumbai' or 'Bangalore'.

    Returns:
        Formatted weather report string.
    """
    if not WEATHER_API_KEY:
        return (
            f"Weather API key not configured. "
            f"Add WEATHER_API_KEY to your .env file (free at weatherapi.com). "
            f"Requested location: {location}"
        )

    try:
        resp = requests.get(
            "https://api.weatherapi.com/v1/current.json",
            params={"key": WEATHER_API_KEY, "q": location, "aqi": "no"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        loc = data["location"]
        cur = data["current"]
        return (
            f"Weather in {loc['name']}, {loc['country']}:\n"
            f"  Condition : {cur['condition']['text']}\n"
            f"  Temp      : {cur['temp_c']}°C (feels like {cur['feelslike_c']}°C)\n"
            f"  Humidity  : {cur['humidity']}%\n"
            f"  Wind      : {cur['wind_kph']} km/h {cur['wind_dir']}\n"
            f"  Visibility: {cur['vis_km']} km"
        )

    except requests.HTTPError as e:
        return f"Weather API error ({e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        return f"Weather error: {e}"
