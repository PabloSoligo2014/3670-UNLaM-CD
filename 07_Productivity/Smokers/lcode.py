
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier # vecinos más cercanos para clasificación
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.feature_selection import SequentialFeatureSelector, RFECV, SelectFromModel
from sklearn import set_config
set_config(transform_output="pandas")
from sklearn.utils.validation import check_is_fitted


#Multicolumn FE. Si fuese sobre una unica columna usar lambdas o cosas mas simples,
#Desarrollado el 25/09/2026, sujeto a pruebas

class MCFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, new_feature_name="", drop_features=False, operation=None):
        self.operation = operation
        self.new_feature_name = new_feature_name
        self.drop_features = drop_features
        self.eps = 1e-6

    def fit(self, X, y=None):
        # 1. Guardar n_features_in_ (estándar de sklearn)
        if hasattr(X, "shape"):
            self.n_features_in_ = X.shape[1]
        
        # 2. Atributos estimados terminan en guión bajo '_' para pass de check_is_fitted
        if isinstance(X, pd.DataFrame):
            self.current_features_ = list(X.columns)
        else:
            self.current_features_ = [f"x{i}" for i in range(self.n_features_in_)]
            
        return self

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["current_features_"])
        
        if input_features is None:
            input_features = self.current_features_
        else:
            input_features = list(input_features)

        if self.drop_features:
            return np.array([self.new_feature_name], dtype=object)
        else:
            return np.array(list(input_features) + [self.new_feature_name], dtype=object)

    def transform(self, X):
        # Valida explícitamente que .fit() haya sido ejecutado
        check_is_fitted(self, attributes=["current_features_"])
        
        if self.operation is None:
            raise ValueError("Operation function was not specified.")

        # Manejo consistente para Pandas DataFrame
        if isinstance(X, pd.DataFrame):
            Xc = X.copy()
            Xc[self.new_feature_name] = self.operation(X, self.eps)
            
            if self.drop_features:
                cols_to_drop = [col for col in X.columns if col != self.new_feature_name]
                Xc.drop(columns=cols_to_drop, inplace=True)
                
            return Xc
        
        # Manejo en caso de recibir arreglos de NumPy
        else:
            new_col = self.operation(X, self.eps)
            if new_col.ndim == 1:
                new_col = new_col.reshape(-1, 1)
                
            if self.drop_features:
                return new_col
            else:
                return np.hstack([X, new_col])



# ¿Para que esto? Porque si se pretende usar SFS + KNN ambos deben compartir el mismo n_neighbors
class SFSKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, weights='uniform',  metric='euclidean', n_features_to_select='auto', direction='forward'):
        self.n_neighbors            = n_neighbors
        self.weights                = weights
        self.metric                 = metric

        
        
        self.n_features_to_select   = n_features_to_select
        self.direction              = direction
        
    def fit(self, X, y):
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.sfs_ = SequentialFeatureSelector(
            estimator=self.knn_,
            n_features_to_select=self.n_features_to_select,
            direction=self.direction
        )
        
        X_sfs = self.sfs_.fit_transform(X, y)
        
        # --- GUARDAR LAS COLUMNAS AQUÍ ---
        if isinstance(X, pd.DataFrame):
            # Si X es un DataFrame, guardamos directamente las columnas seleccionadas
            self.selected_features_ = X.columns[self.sfs_.get_support()].tolist()
        else:
            # Si X es una matriz de NumPy, guardamos los índices de las columnas
            self.selected_features_ = self.sfs_.get_support(indices=True)
            
        self.knn_.fit(X_sfs, y)
        return self

    def predict(self, X):
        X_sfs = self.sfs_.transform(X)
        return self.knn_.predict(X_sfs)

    def predict_proba(self, X):
        X_sfs = self.sfs_.transform(X)
        return self.knn_.predict_proba(X_sfs)


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


    
def get_bmi(X, eps):
    height_m = X['height(cm)'] / 100.0
    return X['weight(kg)'] / (height_m ** 2 + eps)
def get_WHtR(X, eps):
    return X['Cholesterol'] / (X['HDL'] + eps)
def get_TG_HDL_Ratio(X, eps):
    return X['triglyceride'] / (X['HDL'] + eps) 
def get_Non_HDL_Cholesterol(X, eps):
    return X['Cholesterol'] - X['HDL']
#Presion
def get_Pulse_Pressure(X, eps):
    return X['systolic'] - X['relaxation']
def get_MAP(X, eps):
    # Este es ligeramente distino, necesita de Pulse_Pressure que debe ser calculado previamente!!
    return  X['relaxation'] + (X['Pulse_Pressure'] / 3.0)
#Mas
def get_AST_ALT_Ratio(X, eps):
    return X['AST'] / (X['ALT'] + eps)
#No usado        
def get_eyesight_mean(X, eps):
    return (X['eyesight(left)'] + X['eyesight(right)']) / 2.0