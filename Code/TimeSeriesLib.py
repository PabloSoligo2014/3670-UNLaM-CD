import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler


class BaseRollingForecast():
    def __init__(self, **kwargs):
        self.window         = kwargs["window"]
        self.train_percent  = kwargs["train_percent"]
        

    def fit(self, X=None):
        #X es el entrenamiento puro y es la parte de prediccion o entrenamiento
        self.total_len  = X.shape[0]
        self.end_idx    = X.shape[0]
        self.train_len  = round(X.shape[0]*self.train_percent)
        self.horizon    = X.shape[0] - self.train_len;  

    def _calc_score(self, y, y_pred):
        return mean_squared_error(y, y_pred)    

    def score(self):
        return self.score_
        
        


class MeanRollingForecast(BaseRollingForecast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mean_steps = 0
        if "mean_steps" in kwargs:
            self.mean_steps = kwargs["mean_steps"]

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 0:
                raise ValueError("X must contain at least one column.")
            X = X.iloc[:, 0]

        pred_mean = []
        for i in range(self.train_len, self.total_len, self.window):
            steps = min(self.window, self.total_len - i)
            
            if (self.mean_steps>0) and (i-self.mean_steps)>0:
                start_point = i-self.mean_steps
            else:
                start_point = 0

            mean = np.mean(X[start_point:i])
            pred_mean.extend(mean for _ in range(steps))

        index = X.index[self.train_len:self.total_len]
            
        self._sresult = pd.Series(pred_mean, index=index)
        actual = X.iloc[self.train_len:self.total_len]
        self.resid_ = actual - self._sresult
        self.score_ = self._calc_score(actual, self._sresult)
        return self._sresult
    

class LastRollingForecast(BaseRollingForecast):

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 0:
                raise ValueError("X must contain at least one column.")
            X = X.iloc[:, 0]

        pred_last_value = []
        for i in range(self.train_len, self.total_len, self.window):
            steps = min(self.window, self.total_len - i)
            last_value = X[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(steps))
        index = X.index[self.train_len:self.total_len]

        self._sresult = pd.Series(pred_last_value, index=index)
        actual = X.iloc[self.train_len:self.total_len]
        self.resid_ = actual - self._sresult
        self.score_ = self._calc_score(actual, self._sresult)
        return self._sresult


class SarimaxRollingForecast(BaseRollingForecast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


        if "order" in kwargs:
            self.order = kwargs["order"]
        else:
            self.order = (1,0,0)

        if "seasonal_order" in kwargs:
            self.seasonal_order = kwargs["seasonal_order"]
        else:
            self.seasonal_order = (0, 0, 0, 0)


        if "target_col" in kwargs:
            self.target_col = kwargs["target_col"]
        else:
            self.target_col = None
            
         
    def predict(self, X):
        self.scaler_ = StandardScaler()
        pred_ARMA = []

        if isinstance(X, pd.DataFrame):
            if self.target_col is None:
                raise ValueError("target_col is required when X is a DataFrame.")
            if self.target_col not in X.columns:
                raise ValueError(f"Target column {self.target_col!r} is not present in X.")

            y = X[self.target_col]
            exog = X.drop(columns=[self.target_col])
            if exog.empty:
                exog = None
        else:
            y = X
            exog = None

        for i in range(self.train_len, len(y), self.window):
            steps = min(self.window, len(y) - i)

            train_exog = exog.iloc[:i] if exog is not None else None
            forecast_exog = exog.iloc[i:i + steps] if exog is not None else None
            

            if exog is not None:
                #Sin fugas ;)
                self.scaler_.fit(train_exog)
                train_exog      =   self.scaler_.transform(train_exog)
                forecast_exog   =   self.scaler_.transform(forecast_exog)
            #TODO: paralelizar esto si se puede
            model = SARIMAX(
                y.iloc[:i],
                exog=train_exog,
                order=self.order,
                seasonal_order=self.seasonal_order
            )
            self.res_ = model.fit(disp=False)
            oos_pred = self.res_.get_forecast(
                steps=steps,
                exog=forecast_exog,
            ).predicted_mean
            pred_ARMA.extend(oos_pred)

        #aic del ultimo modelo o la media de los aic
        self.aic_   = self.res_.aic
                    
        index = y.index[self.train_len:self.total_len]
        self._sresult = pd.Series(pred_ARMA, index=index)
        actual = y.iloc[self.train_len:self.total_len]
        self.resid_ = actual - self._sresult
        self.score_ = self._calc_score(actual, self._sresult)
        return self._sresult
  