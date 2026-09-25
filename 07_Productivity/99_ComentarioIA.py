# Como comentario, la IA propone esto ¿Esta mal? Y la verdad que no, pero es una solución pobre
# tiene un alcance limitado y una reusabilidad casi nula.
#  
class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-6):
        self.eps = eps  # Previene divisiones por cero

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. Antropometría
        height_m = X_out['height(cm)'] / 100.0
        X_out['BMI'] = X_out['weight(kg)'] / (height_m ** 2 + self.eps)
        X_out['WHtR'] = X_out['waist(cm)'] / (X_out['height(cm)'] + self.eps)
        X_out['Chol_HDL_Ratio'] = X_out['Cholesterol'] / (X_out['HDL'] + self.eps)
        X_out['TG_HDL_Ratio'] = X_out['triglyceride'] / (X_out['HDL'] + self.eps)
        
        # 3. Presión Arterial
        X_out['Pulse_Pressure'] = X_out['systolic'] - X_out['relaxation']
        X_out['MAP'] = X_out['relaxation'] + (X_out['Pulse_Pressure'] / 3.0)

        return X_out