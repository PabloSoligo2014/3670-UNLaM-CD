from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class ColOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, percent=1.5, strategy="remove", columns=[]):
        super().__init__()
        self.percent = percent
        self.columns = columns
        self.strategy = strategy
        self.Qs = {}
    def fit(self, X, y=None):
        for c in self.columns:
            self.Qs[c] = (X[c].quantile(0.25), X[c].quantile(0.75))
        return self
    def  transform(self, X):
        Xc = X.copy()
        for c in self.columns:
            Q1, Q3 = self.Qs[c]
            iqr = Q3 - Q1
            upper_limit = Q3 + self.percent * iqr
            lower_limit = Q1 - self.percent * iqr
            if self.strategy=="remove":

                Xc = Xc[(Xc[c] > lower_limit) & (Xc[c] < upper_limit)]
                
            elif self.strategy=="limit":
                Xc.loc[(Xc[c] < lower_limit), c] = lower_limit
                Xc.loc[(Xc[c] > upper_limit), c] = upper_limit
            elif self.strategy=="mean":
                raise Exception("Strategy not implemented yet")
            else:
                raise Exception("Strategy not implemented yet")
            
        return Xc