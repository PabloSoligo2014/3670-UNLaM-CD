import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        self.score_ = self._calc_score(X.iloc[self.train_len:self.total_len], self._sresult)
        return self._sresult
    

class LastRollingForecast(BaseRollingForecast):

    def predict(self, X):
        pred_last_value = []
        for i in range(self.train_len, self.total_len, self.window):
            steps = min(self.window, self.total_len - i)
            last_value = X[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(steps))
        index = X.index[self.train_len:self.total_len]

        self._sresult = pd.Series(pred_last_value, index=index)
        self.score_ = self._calc_score(X.iloc[self.train_len:self.total_len], self._sresult)
        return self._sresult


class SarimaxRollingForecast(BaseRollingForecast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "p" in kwargs:
            self.p = kwargs["p"]
        else:
            self.p = 1

        if "d" in kwargs:
            self.d = kwargs["d"]
        else:
            self.d = 0

        if "q" in kwargs:
            self.q = kwargs["q"]
        else:
            self.q = 1
     

         
    def predict(self, X):

        pred_ARMA = []
        
        for i in range(self.train_len, self.total_len, self.window):
            steps = min(self.window, self.total_len - i)
            model = SARIMAX(X[:i], order=(self.p,self.d,self.q))
            res = model.fit(disp=False)
            oos_pred = res.get_forecast(steps=steps).predicted_mean
            pred_ARMA.extend(oos_pred)
        index = X.index[self.train_len:self.total_len]
        self._sresult = pd.Series(pred_ARMA, index=index)
        self.score_ = self._calc_score(X.iloc[self.train_len:self.total_len], self._sresult)
        return self._sresult
  