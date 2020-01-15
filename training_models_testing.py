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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

# Train and test samples:
perc_train_size = 0.8
train_size = int(X.shape[0] * perc_train_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, shuffle=True)

#param_grid = [{'C': list(range(1:20)), 'gamma': ['auto'], 'kernel': ['rbf']},]

print('Tamaño de la muestra de entrenamiento: ' + str(np.shape(X_train)[0]))
print('Tamaño de la muestra de testing: ' + str(np.shape(X_test)[0]))

#score = 'precision'
clf = SVC(gamma= 'auto', kernel='poly', probability=True)
#clf = GridSearchCV(SVC(), param_grid, scoring='%s_macro' % score)
clf.fit(X, y)

#print("Best parameters set found on development set:")
#print()
#print(clf.best_params_)
#print()
#print("Grid scores on development set:")
#print()
#means = clf.cv_results_['mean_test_score']
#stds = clf.cv_results_['std_test_score']
#for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#    print("%0.3f (+/-%0.03f) for %r"
#          % (mean, std * 2, params))
#print()
#Kernel results so far:
# rbf: Train = .78      Test = .72
# poly:Train = .94      Test = .94

#Gamma --> higher leads to overfitting
#C --> penalty: higher leads to overfitting
#degree --> polynomial degree, = 1 means linear.

#Degree results so far:
#degree = 3 is default, degree = 4 led to 1.0 score on training, 0.995 on testing


joblib.dump(clf, model_file)

plt.scatter(X[:, 11], X[:, 14], c = y, cmap='YlOrRd', edgecolors='k')
#plt.gray()
#plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.Paired, edgecolors='k')        

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
#YY, XX = np.meshgrid(yy, xx)
#xy = np.vstack([XX.ravel(), YY.ravel()]).T
#
#Z = clf.decision_function(xy).reshape(XX.shape)
#
## plot decision boundary and margins
#a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

plt.show()

# Prediction files por training and test sets
pred_train = clf.predict(X_train)
pred_test = clf.predict(X_test)
#pred_o = clf.predict(X_o)

print('Train Accuracy:', metrics.accuracy_score(y_train, pred_train))
print('Test Accuracy:', metrics.accuracy_score(y_test, pred_test))
#print('Original Set Accuracy:', metrics.accuracy_score(y_o, pred_o))
print(classification_report(y_test, pred_test))

mse_train = metrics.mean_squared_error(y_train, pred_train)
mse_test = metrics.mean_squared_error(y_test, pred_test)
print("Train MSE: {}".format(mse_train))
print("Test MSE: {}".format(mse_test))

#Confusion matrix of test data 
conf_matrix = False
if conf_matrix:
    conf_mat = confusion_matrix(y_test, pred_test)
    print(conf_mat)

# output file:

pred_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in clf.predict(X_test)])
y_test_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in y_test])

if len(sr)>0:
    X_test[:, pos_sr] = le_X.inverse_transform(X_test[:, pos_sr].astype('int')) # Inverse transform of sov. ratingsS


# Testing for sovereign rating dependency



