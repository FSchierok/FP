import numpy as np
import pandas as pd
np.random.seed(0)

# Daten einlesen mit pandas. Konvertierung in numpy array
signal_pandas = pd.read_csv('data/signal_angeglichen+bereinigt.csv', sep=',')
signal = signal_pandas.values
background_pandas = pd.read_csv('data/background_angeglichen+bereinigt.csv', sep=',')
background = background_pandas.values


# Klassifikation in erste Spalte (für Signal "1", für Background "0")
signal = np.column_stack((np.ones(len(signal)), signal))
background = np.column_stack((np.zeros(len(background)), background))

# Fasse signal und backgound als data zusammen
data = np.concatenate((signal, background), axis = 0)

# Bereinigung: Entferne alle Nans und Infs
len_data, width_data = data.shape
for i in range(len_data):
    for j in range(width_data):
        if np.isnan(data[i,j]) or np.isinf(data[i,j]):
            data[i,j] = 10**50

# X sind die Beispiele und y ihre Klassifikation
data_X = data[:,1:]
data_y = data[:,0]

# TODO: Feature Selection
data_X_selected = data_X[:, 140:]

# Permutiere die Indizes von data zufällig beim Aufteilen in Trainings- und
# Testdaten (90% und 10% von Gesamtdaten).
perm = np.random.permutation(len(data))
data_train_y = data[perm[:-len(data_y) // 10]]
data_test_y  = data[perm[-len(data_y) // 10:]]
data_train_X = data[perm[:-len(data_X_selected) // 10]]
data_test_X  = data[perm[-len(data_X_selected) // 10:]]

# Lernen und überprüfen
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(data_train_X, data_train_y).predict(data_test_X)
total = len(data_test_y)
matches = 0
for i in range(total):
    if data_test_y == y_pred:
        matches += 1
print("Number of correctly labeled points out of a total ", total, " points : ", matches)
