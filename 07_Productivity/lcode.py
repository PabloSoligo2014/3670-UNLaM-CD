import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer

#Actualizado el 28/09/2026 para ajustarlo a sklearn 1.2 y la posibilidad de trabajar directamente con pandas.
class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype_map: dict):
        self.dtype_map = dtype_map

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TypeSelector requiere un pandas DataFrame como entrada.")
        
        # Validar que las columnas a transformar existan en X
        missing_cols = set(self.dtype_map.keys()) - set(X.columns)
        if missing_cols:
            raise KeyError(f"Las siguientes columnas no están en el DataFrame: {missing_cols}")
            
        self.feature_names_in_  = list(X.columns)
        self.n_features_in_     = len(self.feature_names_in_)
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["feature_names_in_"])
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TypeSelector requiere un pandas DataFrame como entrada.")

        X_out = X.copy()
        
        # Aplicar casteo solo a las columnas definidas en dtype_map
        for col, dtype in self.dtype_map.items():
            if col in X_out.columns:
                # Manejo de fechas si se especifica 'datetime'
                if dtype in ['datetime', 'datetime64', 'datetime64[ns]']:
                    X_out[col] = pd.to_datetime(X_out[col])
                else:
                    X_out[col] = X_out[col].astype(dtype)
                    
        return X_out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["feature_names_in_"])
        return self.feature_names_in_


class IntegerImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean', number=999):
        super().__init__()
        self.strategy   = strategy
        self.number     = number
        self.imputer    = None

    def _replace_number(self, X):
        if isinstance(X, pd.DataFrame):
            return X.replace(self.number, np.nan)
        if isinstance(X, np.ndarray):
            return np.where(X == self.number, np.nan, X)
        raise TypeError("Input type not supported. Expected pandas DataFrame or numpy array.")

    def fit(self, X, y=None):
         # Agregado 28/09/2026 sklearn 1.2
        self.feature_names_in_  = list(X.columns)
        self.n_features_in_     = len(self.feature_names_in_)
        
        Xc = self._replace_number(X)
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.set_output(transform="pandas")
        self.imputer.fit(Xc)


        return self

    def get_feature_names_out(self, input_features=None):
        # Agregado 28/09/2026 sklearn 1.2
        check_is_fitted(self, attributes=["feature_names_in_"])
        return self.feature_names_in_
        

    def transform(self, X):
        Xc = self._replace_number(X)
        Xt = self.imputer.transform(Xc)
        return Xt.astype(int)