from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .models import ForecastPoint, ForecastResult


class PriceDataProvider(Protocol):
    def load(self, instrument: str) -> pd.Series: ...


class CsvPriceDataProvider:
    def __init__(self, path: str | Path): self.path = Path(path)

    def load(self, instrument: str) -> pd.Series:
        if not self.path.exists(): raise FileNotFoundError(f"Price CSV not found: {self.path}")
        frame = pd.read_csv(self.path)
        required = {"date", "instrument", "price"}
        if not required.issubset(frame.columns): raise ValueError(f"CSV requires columns: {sorted(required)}")
        subset = frame[frame.instrument.str.casefold() == instrument.casefold()].copy()
        if len(subset) < 12: raise ValueError("At least 12 observations are required")
        subset["date"] = pd.to_datetime(subset.date)
        return subset.sort_values("date").set_index("date").price.astype(float).asfreq("MS").interpolate()


class EiaBrentDataProvider:
    """Load the official EIA RBRTE daily XLS and normalize it to monthly means."""
    def __init__(self, path: str | Path, training_years: int = 10):
        self.path, self.training_years = Path(path), training_years

    def load(self, instrument: str) -> pd.Series:
        if instrument.casefold() != "brent":
            raise ValueError("The bundled EIA dataset contains Brent only")
        if not self.path.exists():
            raise FileNotFoundError(f"EIA Brent XLS not found: {self.path}")
        frame = pd.read_excel(self.path, sheet_name="Data 1", skiprows=3,
                              names=["date", "price"], usecols=[0, 1])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame = frame.dropna().drop_duplicates("date", keep="last").sort_values("date")
        cutoff = frame.date.max() - pd.DateOffset(years=self.training_years)
        series = frame.loc[frame.date >= cutoff].set_index("date").price.resample("MS").mean()
        series = series.interpolate(limit=2)
        if series.isna().any() or len(series) < 24:
            raise ValueError("EIA series cannot be normalized into a complete monthly history")
        series.name = "Brent"
        return series


def _metrics(series: pd.Series, fitted: np.ndarray) -> dict[str, float | None]:
    actual = series.to_numpy()[-len(fitted):]
    rmse = float(np.sqrt(np.mean((actual - fitted) ** 2)))
    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs((actual - fitted) / denom)) * 100)
    return {"rmse_in_sample": rmse, "mape_in_sample_pct": mape}


def forecast_oil_price(provider: PriceDataProvider, instrument: str, horizon: int,
                       method: str = "exponential_smoothing") -> ForecastResult:
    if horizon < 1 or horizon > 36: raise ValueError("forecast horizon must be between 1 and 36 months")
    series = provider.load(instrument)
    if method == "sarima":
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False).fit(disp=False)
        prediction = fit.get_forecast(horizon); mean = prediction.predicted_mean
        interval = prediction.conf_int(alpha=0.2); lower, upper = interval.iloc[:, 0], interval.iloc[:, 1]
        fitted = fit.fittedvalues.iloc[2:].to_numpy()
    elif method == "exponential_smoothing":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if len(series) >= 24 else None
        fit = ExponentialSmoothing(series, trend="add", seasonal=seasonal, seasonal_periods=12 if seasonal else None).fit()
        mean = fit.forecast(horizon); residual_std = float(np.std(fit.resid, ddof=1))
        scale = 1.2816 * residual_std * np.sqrt(np.arange(1, horizon + 1)); lower, upper = mean - scale, mean + scale
        fitted = np.asarray(fit.fittedvalues)
    else: raise ValueError("method must be 'sarima' or 'exponential_smoothing'")
    points = [ForecastPoint(period=str(idx.date()), value=float(v), lower_bound=float(lo), upper_bound=float(hi)) for idx, v, lo, hi in zip(mean.index, mean, lower, upper)]
    return ForecastResult(instrument=instrument, forecast_horizon=horizon, method=method, forecast=points,
        lower_bound=[p.lower_bound for p in points], upper_bound=[p.upper_bound for p in points], metrics=_metrics(series, fitted),
        interpretation="Статистический базовый прогноз, не учитывающий будущие шоки и решения OPEC+.",
        assumptions=["Историческая динамика сохраняет прогностическую ценность", "Интервал — ориентировочный 80% диапазон"])
