from oil_analyst.forecasting import EiaBrentDataProvider


def test_real_eia_loader_returns_monthly_brent():
    series = EiaBrentDataProvider("data/prices/RBRTEd.xls").load("Brent")
    assert len(series) >= 120 and series.index.is_monotonic_increasing and not series.isna().any()
