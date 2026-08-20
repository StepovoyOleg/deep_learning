import pytest

from oil_analyst.forecasting import EiaBrentDataProvider, forecast_oil_price


@pytest.mark.parametrize("method", ["exponential_smoothing", "sarima"])
def test_real_brent_forecast_has_three_points_and_intervals(method):
    result = forecast_oil_price(EiaBrentDataProvider("data/prices/RBRTEd.xls"), "Brent", 3, method)
    assert len(result.forecast) == 3
    assert all(point.lower_bound < point.value < point.upper_bound for point in result.forecast)
