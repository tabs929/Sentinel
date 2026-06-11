"""Weather tool — uses open-meteo.com, which needs no API key.

This is the smoke-test tool: it is always "available" so you can exercise the
full agent/observability pipeline without configuring any credentials.
"""

from __future__ import annotations

from typing import Any

import httpx

from sentinel.tools.base import BaseTool, ToolError

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
}


class WeatherTool(BaseTool):
    name = "get_weather"
    description = (
        "Get the current weather for a city or place. Use this whenever the "
        "user asks about weather, temperature, or conditions somewhere. "
        "No credentials required."
    )

    retryable_exceptions = (ToolError, httpx.HTTPError, ConnectionError, TimeoutError)

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or place name, e.g. 'San Francisco' or 'Tokyo, Japan'.",
                }
            },
            "required": ["location"],
        }

    async def _call(self, location: str | None = None, **_: Any) -> tuple[str, dict[str, Any]]:
        if not location or not location.strip():
            raise ToolError("a non-empty 'location' is required")

        timeout = self.settings.tool_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            geo_resp = await client.get(
                GEOCODE_URL, params={"name": location, "count": 1, "language": "en"}
            )
            geo_resp.raise_for_status()
            geo = geo_resp.json()
            results = geo.get("results") or []
            if not results:
                raise ToolError(f"no location found matching '{location}'")
            place = results[0]
            lat, lon = place["latitude"], place["longitude"]

            wx_resp = await client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                },
            )
            wx_resp.raise_for_status()
            current = wx_resp.json().get("current", {})

        code = current.get("weather_code")
        condition = _WEATHER_CODES.get(code, f"code {code}")
        name = ", ".join(
            p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p
        )
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")

        content = (
            f"Weather in {name}: {condition}, {temp}°C, "
            f"humidity {humidity}%, wind {wind} km/h."
        )
        data = {
            "location": name,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": temp,
            "humidity_pct": humidity,
            "wind_kph": wind,
            "condition": condition,
            "weather_code": code,
        }
        return content, data
