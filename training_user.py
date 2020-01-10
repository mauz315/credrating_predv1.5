# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 09:54:30 2019

@author: P900017
"""
import pandas as pd
from rating_functions import model_training, feat_elim
#import matplotlib.pyplot as plt

financials = True
bank_suffix = ''
train_data = 'data_em_1212_0619.csv'

if financials == True:
    bank_suffix = '_banks'
    train_data = 'research_data_banks.csv'

## Cargando archivosy bases de datos necesarias
# Rutas, nombres de ratios / nombres de ratios pure
feat_key = pd.read_csv('data/features' + bank_suffix + '.csv', sep=',', index_col = ["Feature"], encoding = "latin1")
# Encoder para calificaciones:
le = pd.read_csv('data/lab_encoder' + bank_suffix + '.csv', sep=',', index_col = 0, encoding = "latin1")
# Datos de entrenamiento:
data_em = pd.read_csv('data/' + train_data, sep=',', index_col = ["Fecha", 'Ticker'], encoding = "latin1")

## Visualización de clases para mayor insight (histograma y estadisticos basicos)
#print(data_em.describe())
#data_em['IssuerRating'].value_counts().plot(kind='bar')
#plt.show()

## Reduccion de variables
# Variables disponibles para agregar a to_del
#   "Ratio1", "Ratio2", "Ratio3", "Ratio4", "Ratio5", "Ratio6", 
#   "Ratio7", "Ratio8", "Ratio9", "Ratio10", "Ratio11", "Ratio12", "Ratio13"
# Automáticamente toma los LTM +13
#Para no reducir variables, to_del =[]
# Primeras opciones: Ratio4, Ratio9, Ratio13, Ratio7
to_del = []
if to_del:
    data_em, feat_key = feat_elim(data_em, feat_key, to_del)

# Eliminar NaN desde ya
data_em['NA'] = False
for i in range(len(data_em)):
    for obs in list(data_em.iloc[i]):
        if pd.isnull(obs):
            data_em.iat[i, data_em.columns.get_loc('NA')] = True
data_em = data_em[data_em.NA == False]
del data_em["NA"]

## Remover desviaciones atípicas para algunos ratios, necesita NaNs eliminados
# Lista de ratios solo comprueba el if
#Usar | para varias condiciones
#p = 5
#qhigh3 = np.percentile(data_em.Ratio3,100 - p/2)
#qlow3 = np.percentile(data_em.Ratio3,p/2)
#qhigh6 = np.percentile(data_em.Ratio6,100 - p/2)
#qlow6 = np.percentile(data_em.Ratio6,p/2)
#qhigh8 = np.percentile(data_em.Ratio8,100 - p/2)
#qlow8 = np.percentile(data_em.Ratio8,p/2)
## Recomendado: Ratio3, Ratio6, Ratio8, Ratio13
#critical_feat = True
#if critical_feat:
#    data_em = data_em[data_em.Ratio3 < qhigh3]
#    data_em = data_em[data_em.Ratio3 > qlow3]
#    data_em = data_em[data_em.Ratio6 < qhigh6]
#    data_em = data_em[data_em.Ratio6 > qlow6]
#    data_em = data_em[data_em.Ratio8 < qhigh8]
#    data_em = data_em[data_em.Ratio8 > qlow8]
# inputs para las funciones de training
remove_nan = True # Remover filas con datos faltantes.
n_estimators = 1000 # Número de árboles de entrenamiento
min_samples_leaf = 2
model_file = 'model/actual_rf_em' + bank_suffix + '.sav' # Modelo.
sov_encoder_file = 'model/sov_lab_encoder_em' + bank_suffix + '.sav' # Encoder de rating soberano.
output_test = 'output/pred_test.csv' # Archivo de salida con prediciones.
#LIME train set
train_set = 'explainer/X_train_actual' + bank_suffix + '.sav' # training set, depende del modelo utilizado

# Training original de rating_functions
model_training(data_em, feat_key, le, remove_nan, output_test, 
               model_file, train_set, sov_encoder_file,
               n_estimators = n_estimators, min_samples_leaf = min_samples_leaf,
               permut=True, shuffle_sample=False, conf_matrix = True)
