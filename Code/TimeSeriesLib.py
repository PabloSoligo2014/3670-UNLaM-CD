import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class BaseRollingForecast:
    """Use fit/predict for final forecasts and backtest for rolling evaluation."""

    def __init__(self, **kwargs):
        self.window = kwargs["window"]
        self.train_percent = kwargs["train_percent"]

    def _calc_score(self, y, y_pred):
        return mean_squared_error(y, y_pred)

    def score(self):
        return self.score_

    def _backtest_start(self, n_samples):
        train_len = round(n_samples * self.train_percent)
        if train_len < 1 or train_len >= n_samples:
            raise ValueError("train_percent must leave at least one training and one test observation.")
        if not isinstance(self.window, (int, np.integer)) or self.window < 1:
            raise ValueError("window must be at least 1.")
        
        self.train_len_ = train_len
        return train_len

    def _set_backtest_results(self, y, predictions, start):
        index = y.index[start:]
        self._sresult = pd.Series(predictions, index=index, name="predicted_mean")
        actual = y.iloc[start:]
        self.resid_ = actual - self._sresult
        self.score_ = self._calc_score(actual, self._sresult)
        return self._sresult

    @staticmethod
    def _forecast_index(history, steps, supplied_index=None):
        if supplied_index is not None:
            return supplied_index
        index = history.index
        if isinstance(index, pd.RangeIndex):
            return pd.RangeIndex(index[-1] + index.step, index[-1] + index.step * (steps + 1), index.step)
        if isinstance(index, pd.DatetimeIndex):
            freq = index.freq
            if freq is None and len(index) >= 3:
                freq = pd.infer_freq(index)
            if freq is not None:
                return pd.date_range(start=index[-1], periods=steps + 1, freq=freq)[1:]
        return pd.RangeIndex(len(history), len(history) + steps)

    @staticmethod
    def _as_series(X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 0:
                raise ValueError("X must contain at least one column.")
            X = X.iloc[:, 0]
        elif not isinstance(X, pd.Series):
            X = pd.Series(X)
        if X.empty:
            raise ValueError("X must contain at least one observation.")
        return X

    @staticmethod
    def _validate_steps(steps):
        if not isinstance(steps, (int, np.integer)) or steps < 1:
            raise ValueError("steps must be a positive integer.")


class MeanRollingForecast(BaseRollingForecast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_steps = kwargs.get("mean_steps", 0)

    def fit(self, X):
        self.y_ = self._as_series(X).copy()
        self.is_fitted_ = True
        return self

    def predict(self, steps):
        self._validate_steps(steps)
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("Call fit before predict.")
        start = max(0, len(self.y_) - self.mean_steps) if self.mean_steps > 0 else 0
        forecast = np.full(steps, self.y_.iloc[start:].mean())
        return pd.Series(forecast, index=self._forecast_index(self.y_, steps), name="predicted_mean")

    def backtest(self, X):
        y = self._as_series(X)
        train_len = self._backtest_start(len(y))
        
        predictions = []
        for i in range(train_len, len(y), self.window):
            steps = min(self.window, len(y) - i)
            start = max(0, i - self.mean_steps) if self.mean_steps > 0 else 0
            mean = y.iloc[start:i].mean()
            predictions.extend([mean] * steps)
        return self._set_backtest_results(y, predictions, train_len)


class LastRollingForecast(BaseRollingForecast):
    def fit(self, X):
        self.y_ = self._as_series(X).copy()
        self.is_fitted_ = True
        return self

    def predict(self, steps):
        self._validate_steps(steps)
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("Call fit before predict.")
        forecast = np.full(steps, self.y_.iloc[-1])
        return pd.Series(forecast, index=self._forecast_index(self.y_, steps), name="predicted_mean")

    def backtest(self, X):
        y = self._as_series(X)
        train_len = self._backtest_start(len(y))
        predictions = []
        for i in range(train_len, len(y), self.window):
            steps = min(self.window, len(y) - i)
            predictions.extend([y.iloc[i - 1]] * steps)
        return self._set_backtest_results(y, predictions, train_len)


class SarimaxRollingForecast(BaseRollingForecast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = kwargs.get("order", (
            kwargs.get("p", 1),
            kwargs.get("d", 0),
            kwargs.get("q", 0),
        ))
        self.seasonal_order = kwargs.get("seasonal_order", (0, 0, 0, 0))
        self.target_col = kwargs.get("target_col")

    def _split_input(self, X, exog=None):
        if isinstance(X, pd.DataFrame):
            if self.target_col is None:
                raise ValueError("target_col is required when X is a DataFrame.")
            if self.target_col not in X.columns:
                raise ValueError(f"Target column {self.target_col!r} is not present in X.")
            if exog is not None:
                raise ValueError("Pass exogenous variables either in X or through exog, not both.")
            y = X[self.target_col]
            exog = X.drop(columns=[self.target_col])
            if exog.empty:
                exog = None
        else:
            y = self._as_series(X)
        exog = self._coerce_exog(exog, y.index)
        if exog is not None and len(exog) != len(y):
            raise ValueError("y and exog must have the same number of observations.")
        return y, exog

    @staticmethod
    def _coerce_exog(exog, index, columns=None):
        if exog is None or isinstance(exog, pd.DataFrame):
            return exog
        values = np.asarray(exog)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise ValueError("exog must be a 2D array or DataFrame.")
        if len(values) != len(index):
            raise ValueError("y and exog must have the same number of observations.")
        if columns is None:
            columns = [f"x{i}" for i in range(values.shape[1])]
        if values.shape[1] != len(columns):
            raise ValueError("exog must have the same number of columns as the training exog.")
        return pd.DataFrame(values, index=index, columns=columns)

    @staticmethod
    def _scaled_exog_frame(scaler, exog):
        return pd.DataFrame(
            scaler.transform(exog),
            index=exog.index,
            columns=exog.columns,
        )

    def fit(self, X, exog=None):
        y, exog = self._split_input(X, exog)
        if exog is not None:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(exog)
            scaled_exog = self._scaled_exog_frame(self.scaler_, exog)
        else:
            self.scaler_ = None
            scaled_exog = None

        result = SARIMAX(
            y,
            exog=scaled_exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
        ).fit(disp=False)
        self.res_ = result
        self.y_ = y.copy()
        self.exog_columns_ = list(exog.columns) if exog is not None else None
        self.aic_ = result.aic
        return self

    def predict(self, steps, exog_future=None):
        self._validate_steps(steps)
        if not hasattr(self, "res_"):
            raise RuntimeError("Call fit before predict.")
        if self.scaler_ is None:
            if exog_future is not None:
                raise ValueError("This model was fitted without exogenous variables.")
            scaled_exog = None
        else:
            if exog_future is None:
                raise ValueError("exog_future is required because the model was fitted with exogenous variables.")
            if len(exog_future) != steps:
                raise ValueError("steps must match the number of rows in exog_future.")
            if isinstance(exog_future, pd.DataFrame) and list(exog_future.columns) != self.exog_columns_:
                raise ValueError("exog_future must have the same columns in the same order as the training exog.")
            exog_future = self._coerce_exog(
                exog_future,
                self._forecast_index(self.y_, steps),
                self.exog_columns_,
            )
            scaled_exog = self._scaled_exog_frame(self.scaler_, exog_future)

        forecast = self.res_.get_forecast(steps=steps, exog=scaled_exog).predicted_mean
        if exog_future is not None:
            forecast.index = exog_future.index
        return forecast

    def backtest(self, X, exog=None):
        y, exog = self._split_input(X, exog)
        train_len = self._backtest_start(len(y))
        predictions = []
        fold_aic = []

        for i in range(train_len, len(y), self.window):
            steps = min(self.window, len(y) - i)
            train_exog = exog.iloc[:i] if exog is not None else None
            forecast_exog = exog.iloc[i:i + steps] if exog is not None else None
            if train_exog is not None:
                scaler = StandardScaler()
                scaler.fit(train_exog)
                train_exog = self._scaled_exog_frame(scaler, train_exog)
                forecast_exog = self._scaled_exog_frame(scaler, forecast_exog)

            result = SARIMAX(
                y.iloc[:i],
                exog=train_exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
            ).fit(disp=False)
            fold_aic.append(result.aic)
            forecast = result.get_forecast(steps=steps, exog=forecast_exog).predicted_mean
            predictions.extend(forecast.tolist())

        self.backtest_aic_ = fold_aic
        return self._set_backtest_results(y, predictions, train_len)
