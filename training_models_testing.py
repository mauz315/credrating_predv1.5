# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:00:40 2020

@author: P900017
"""

# SVM testing script w/o functions

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def is_string(s):
    try:
        float(s)
        return False
    except ValueError:
        return True
    
## Cargando archivosy bases de datos necesarias
# Rutas, nombres de ratios / nombres de ratios pure
feat_key = pd.read_csv('data/features_banks.csv', sep=',', index_col = ["Feature"], encoding = "latin1")
# Encoder para calificaciones:
le = pd.read_csv('data/lab_encoder_banks.csv', sep=',', index_col = 0, encoding = "latin1")
# Datos de entrenamiento:
data_em = pd.read_csv('data/research_data_banks.csv', sep=',', index_col = ["Fecha", 'Ticker'], encoding = "latin1")

# Eliminar NaN desde ya
data_em['NA'] = False
for i in range(len(data_em)):
    for obs in list(data_em.iloc[i]):
        if pd.isnull(obs):
            data_em.iat[i, data_em.columns.get_loc('NA')] = True
data_em = data_em[data_em.NA == False]
del data_em["NA"]

remove_nan = True # Remover filas con datos faltantes.
model_file = 'model/actual_svm_em_banks.sav' # Modelo.
sov_encoder_file = 'model/sov_lab_encoder_em.sav' # Encoder de rating soberano.
output_test = 'output/pred_test_svm.csv' # Archivo de salida con prediciones.

data_index = data_em.index # Se crea la variable data_index para publicar el output.
y_ = np.array(data_em.pop('IssuerRating'))
X_ = np.array(data_em[feat_key["Key"]])

# Remove observations with no output
ind_valid_out = [is_string(yi) for yi in y_]
X = X_[ind_valid_out]
y = y_[ind_valid_out]
data_index = data_index[ind_valid_out]

a = []
for yi in y_:
    if is_string(yi):
        a.append(le.loc[yi])
    else:
        float('NaN')

y = np.array(a)

# Encode Sovereign Rating
sr = feat_key[feat_key["Key"] == 'SovereignRating']
if len(sr)>0:
    pos_sr = feat_key.index.get_loc(sr.index[0])# Position sovereign rating
    pos_str = [is_string(x) for x in X[:,pos_sr]]
    labels = np.unique(X[pos_str,pos_sr])
#        labels = np.array(order_ratings(labels))
    le_X = LabelEncoder()
    le_X.fit(labels)
    X[pos_str,pos_sr] = le_X.transform(X[pos_str,pos_sr])
    joblib.dump(le_X, sov_encoder_file)# Save sovereign label encoder

# Remove NaN
if remove_nan:
    ind_not_na = [not np.isnan(np.sum(x)) for x in X]
    X = X[ind_not_na]
    y = y[ind_not_na]
    data_index = data_index[ind_not_na]

y = np.ravel(y)

clf = SVC(gamma='auto', decision_function_shape='ovo')
clf.fit(X, y)

joblib.dump(clf, model_file)

plt.scatter(X[:, 11], X[:, 14], c = y, cmap='YlOrRd', edgecolors='k')
#plt.gray()
#plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.Paired, edgecolors='k')        

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

plt.show()

