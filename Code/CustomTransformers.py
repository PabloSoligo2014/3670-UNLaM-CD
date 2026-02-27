from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class ColumnScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None, columns=None):
        super().__init__()
        self.columns = columns
        self.scaler = scaler
    
    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        self.scaler.fit(X[self.columns])
        return self
    
    def get_feature_names_out(self, input_features=None):
        return self.columns
        
    def transform(self, X, y=None):
        Xc = X.copy()
        Xc.loc[:,self.columns] = self.scaler.transform(Xc[self.columns])
        return Xc

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, **kargs):
        super().__init__()
        self.columns = kargs["columns"]
    
    def fit(self, X, y=None):
        self.input_columns = X.columns
        return self
    
    def get_feature_names_out(self, input_features=None):
        return [col for col in self.input_columns if col not in self.columns]
        
    def transform(self, X, y=None):
        Xc = X.copy()
        return Xc.drop(columns=self.columns, axis=1)

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
                raise Exception("It's not working yet")
                #Xc = Xc[(Xc[c] > lower_limit) & (Xc[c] < upper_limit)]
                
            elif self.strategy=="limit":

                dtype = Xc[c].dtype
                if np.issubdtype(dtype, np.integer):
                    lower_limit = int(lower_limit)
                    upper_limit = int(upper_limit)



                Xc.loc[(Xc[c] < lower_limit), c] = lower_limit
                Xc.loc[(Xc[c] > upper_limit), c] = upper_limit
            elif self.strategy=="mean":
                raise Exception("Strategy not implemented yet")
            else:
                raise Exception("Strategy not implemented yet")
            
        return Xc
    
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns
    
    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns
        return self
    
    def get_feature_names_out(self, input_features=None):
        return self.columns
        
    def transform(self, X, y=None):
        Xc = X.copy()
        return Xc[self.columns]

class CollinearityDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, min_coef=0.99, method="pearson"):
        super().__init__()
        self.columns =  columns
        self.min_coef = min_coef  
        self.columns_to_drop = [] 
        self.method = method   
    
    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=['number']).columns
        
        #method{‘pearson’, ‘kendall’, ‘spearman’} 
        correlation_matrix = X[self.columns].corr(method=self.method)
        fc = correlation_matrix.shape[1]
        for i in range(fc):
            for j in range(i+1, fc):
                if abs(correlation_matrix.iloc[i, j]) > self.min_coef and correlation_matrix.columns[j] not in self.columns_to_drop:
                    self.columns_to_drop.append(correlation_matrix.columns[j])
        return self
    
    def get_feature_names_out(self, input_features=None):
        return self.columns.drop(columns=self.columns_to_drop, axis=1)
        
    def transform(self, X, y=None):
        Xc = X.copy()
        return Xc.drop(columns=self.columns_to_drop)