from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

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


#TODO: Actualizado a versiones nuevas de sklearn salida pandas 
class CollinearityDropper(BaseEstimator, TransformerMixin):
    """
    Parámetros:
    -----------
    min_coef : float, default=0.8
        Umbral absoluto de correlación (0 a 1). Se eliminan columnas con |r| >= min_coef.
    method : str, default='pearson'
        Método de correlación de pandas ('pearson', 'spearman', 'kendall').
    """
    def __init__(self, min_coef=0.8, method="pearson"):
        self.min_coef = min_coef
        self.method = method

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CollinearityDropper requiere un DataFrame de Pandas como entrada.")

        # Atributos estándar con guión bajo final para check_is_fitted
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = list(X.columns)

        # Cálculo de la matriz de correlación absoluta
        corr_matrix = X.corr(method=self.method).abs()

        # Matriz triangular superior para no evaluar pares duplicados
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identificación de columnas redundantes
        self.cols_to_drop_ = [
            column for column in upper_tri.columns if any(upper_tri[column] >= self.min_coef)
        ]
        
        # Columnas conservadas
        self.feature_names_out_ = [col for col in X.columns if col not in self.cols_to_drop_]

        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["cols_to_drop_"])

        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.cols_to_drop_)
        else:
            # Soporte en caso de que X llegue como ndarray de NumPy
            indices_to_keep = [
                i for i, col in enumerate(self.feature_names_in_) if col not in self.cols_to_drop_
            ]
            return X[:, indices_to_keep]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["feature_names_out_"])
        return np.array(self.feature_names_out_, dtype=object)
