from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
np.random.seed(0)

# Daten einlesen mit pandas. Konvertierung in numpy array
signal_pandas = pd.read_csv('data/signal.csv', sep=';')
signal = signal_pandas.values
background_pandas = pd.read_csv('data/signal.csv', sep=';')
background = background_pandas.values

# Klassifikation in erste Spalte (für Signal "1", für Background "0")
signal = np.column_stack((np.ones(len(signal)), signal))
background = np.column_stack((np.zeros(len(background)), background))

# Fasse signal und backgound als data zusammen und permutiere dann die Indizes
# von data zufällig beim Aufteilen in Trainings- und Testdaten (90% und 10% von
# Gesamtdaten). Dabei sind X die Beispiele und y ihre Klassifikation
data = np.concatenate((signal, background), axis = 0)
perm = np.random.permutation(len(data))
data_train = data[perm[:-len(data) // 10]]
data_test  = data[perm[-len(data) // 10:]]
data_train_X = data_train[:,1:]
data_train_y = data_train[:,0]
data_test_X = data_test[:,1:]
data_test_y = data_test[:,0]

# TODO: NaNs und Infs entfernen; Attribute entfernen, die nur in einem der bei-
# den Datensätze vorkommen; keine Monte-Carlo-Wahrheiten; keine Eventidenti-
# fikationsnummern; keine Gewichte --> in calc entsprechende Spalten finden und
# per Hand rausnehmen... dauert lange bei 280 Spalten. Schnellere Möglichkeit?


# Lernen und überprüfen
# nn = NearestNeighbors(1).fit(data_train_X)
# distances, indices = nn.kneighbors(XTest)
# print(indices.shape)
# total = 0
# match = 0
# for i in range(len(data_test_y)):
#     total += 1
#     if data_test_y[i] == data_train_y[indices[i]]:
#             match += 1;
# print(total, match)
