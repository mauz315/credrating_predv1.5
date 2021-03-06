# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:05:46 2019

@author: P900017
"""

import pandas as pd
from sklearn.externals import joblib
from rating_functions import rating_prediction
from lime_explainer import explain_tree

# Rutas para archivos de modelos y encoders
feat_key = pd.read_csv('data/features.csv', sep=',', index_col = ["Feature"], encoding = "latin1")
le = pd.read_csv('data/lab_encoder.csv', sep=',', index_col = 0, encoding = "latin1")
model_file = 'model/actual_rf_em.sav' # Modelo.
#model_pure_file = folder + 'model/rating_random_forest_pure.sav' # Modelo.
train_set = 'explainer/X_train_actual.sav'
sov_encoder_file = 'model/sov_lab_encoder_em.sav' # Encoder de rating soberano.
#scaler_dir = 'model/my_scaler.sav'
# Datos de carga de modelos:
rf = joblib.load(model_file)
#rf_pure = joblib.load(model_pure_file)
X_train = joblib.load(train_set)
#scaler = joblib.load(scaler_dir)
sov_lab_encoder = joblib.load(sov_encoder_file)
output_pred = 'output/kallpa_actual.csv' # Nombre de archivo donde se publican los resultados.

#Nueva data para predecir
data = pd.read_csv('input/rating_pred_kallpa.csv', sep=',', index_col = ["Indicadores"], encoding = "latin1")

# Predicción original de rating_functions (imprime resultados y guarda archivo)
rating_prediction(data, rf, feat_key, le, sov_lab_encoder, output_pred)

lime_explain = False
print_exp = False

##############################################################################
##############################################################################
####     LIME EXPLAIN
##############################################################################
##############################################################################
# Opciones de usuario
period = 0 # Periodo a explicar
ratings = 2 # Ratings a explicar (n opciones más probables para el modelo)

if lime_explain:
    # Crea el archivo /explainer/lime_output.html
   explain_tree(data, period, ratings, rf, X_train, sov_lab_encoder, 
                le, feat_key, print_exp)
