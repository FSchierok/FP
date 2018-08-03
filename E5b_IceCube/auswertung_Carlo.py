import numpy as np
import pandas as pd
from sklearn import metrics
np.random.seed(2)

# Daten einlesen mit pandas. Konvertierung in numpy array. Format:
# (N_samples, N_features)
signal_pandas = pd.read_csv('data/signal_angeglichen+bereinigt.csv', sep=',')
signal = signal_pandas.values
background_pandas = pd.read_csv('data/background_angeglichen+bereinigt.csv', sep=',')
background = background_pandas.values
# signal_pandas = pd.read_csv('data/signal.csv', sep=';')
# signal = signal_pandas.values
# background_pandas = pd.read_csv('data/signal.csv', sep=';')
# background = background_pandas.values

# Klassifikation in erste Spalte (für Signal "1", für Background "0")
signal = np.column_stack((np.ones(len(signal)), signal))
background = np.column_stack((np.zeros(len(background)), background))

# Fasse signal und backgound als data zusammen
data = np.concatenate((signal, background), axis = 0)

# Bereinigung: Entferne alle Nans und Infs
len_data, width_data = data.shape
for i in range(len_data):
    for j in range(width_data):
        if (np.isnan(data[i,j]) or np.isinf(data[i,j])):
            data[i,j] = 10**100

# X sind die Beispiele und y ihre Klassifikation
data_X = data[:,1:]
data_y = data[:,0]

# TODO: Feature Selection
data_X_selected = data_X[:,:15]

# Permutiere die Indizes von data zufällig beim Aufteilen in Trainings- und
# Testdaten (90% und 10% von Gesamtdaten).
perm = np.random.permutation(len(data))
data_train_y = data_y[perm[:-len(data_y) // 10]]
data_test_y  = data_y[perm[-len(data_y) // 10:]]
data_train_X = data_X_selected[perm[:-len(data_X_selected) // 10]]
data_test_X  = data_X_selected[perm[-len(data_X_selected) // 10:]]


# Lernen und überprüfen
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(data_train_X, data_train_y)
expected = data_test_y
predicted = model.predict(data_test_X)
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
