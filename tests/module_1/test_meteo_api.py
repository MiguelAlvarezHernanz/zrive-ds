from src.module_1.module_1_meteo_api import api_call, get_data_meteo_api, process_data
import pytest
import pandas as pd
from unittest.mock import patch

API_URL = "https://archive-api.open-meteo.com/v1/archive?"
MOCK_RESPONSE = {
    "daily": {
        "time": ["2020-01-01", "2020-01-02"],
        "temperature_2m_mean": [5, 6],
        "precipitation_sum": [0.1, 0.2],
        "wind_speed_10m_max": [3, 4],
    }
}


def test_api_call_success():
    params = {"latitude": 40.416775, "longitude": -3.703790, "daily": "temperature_2m_mean"}
    with patch("meteo_api.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = MOCK_RESPONSE

        response = api_call(API_URL, params)
        assert response == MOCK_RESPONSE


def test_api_call_rate_limit():
    params = {"latitude": 40.416775, "longitude": -3.703790, "daily": "temperature_2m_mean"}
    with patch("meteo_api.requests.get") as mock_get:
        mock_get.return_value.status_code = 429
        mock_get.return_value.headers = {"Retry-After": "1"}
        with pytest.raises(RuntimeError):
            api_call(API_URL, params)


def test_get_data_meteo_api():
    with patch("meteo_api.api_call", return_value=MOCK_RESPONSE):
        df = get_data_meteo_api("Madrid")
        assert isinstance(df, pd.DataFrame)
        assert "temperature_2m_mean" in df.columns
        assert "precipitation_sum" in df.columns
        assert len(df) == 2


def test_process_data():
    data = pd.DataFrame({
        "time": ["2020-01-01", "2020-01-02"],
        "temperature_2m_mean": [5, 6],
        "precipitation_sum": [0.1, 0.2],
        "wind_speed_10m_max": [3, 4],
    })
    df = process_data(data)
    assert isinstance(df, pd.DataFrame)
    assert df.index.freq == "M"  # Comprueba que la frecuencia es mensual


def test_main():
    raise NotImplementedError
