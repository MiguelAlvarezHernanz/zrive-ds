import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
from module_1_extra_fun import plot_comparative_temperature, plot_comparative_precipitation

API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def api_call(url, params, retries=3):
    for attempt in range(retries):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("Rate limit reached. Retrying...")
            time.sleep(2 ** attempt)
        else:
            response.raise_for_status()
    raise RuntimeError("API request failed after retries")


def get_data_meteo_api(city, start_year=2010, end_year=2020):
    latitude = COORDINATES[city]["latitude"]
    longitude = COORDINATES[city]["longitude"]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "daily": ",".join(VARIABLES),
        "timezone": "auto",
    }
    response = api_call(API_URL, params)
    if "daily" not in response or not all(var in response["daily"] for var in VARIABLES):
        raise ValueError("API response schema is invalid or incomplete")
    return pd.DataFrame(response["daily"])


def process_data(df):
    # Resample monthly
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df.resample("ME").mean()  


def plot_data(data, city):
    legend = ["Temperature: °C", "Precipitation: mm", "Wind speed: m/s"]

    plt.figure(figsize=(12, 8))
    for i,var in enumerate(VARIABLES):
        plt.plot(data.index, data[var], label=legend[i])
    plt.title(f"Climatic Variables in {city}")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    results_raw = {}
    results = {}
    for city in COORDINATES.keys():
        print(f"Fetching data for {city}...")
        raw_data = get_data_meteo_api(city)
        results_raw[city] = raw_data
        processed_data = process_data(raw_data)
        results[city] = processed_data
        plot_data(processed_data, city)
    
#    plot_comparative_temperature(results_raw)
#    plot_comparative_precipitation(results_raw)


if __name__ == "__main__":
    main()
