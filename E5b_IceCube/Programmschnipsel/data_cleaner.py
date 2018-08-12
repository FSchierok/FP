import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

# Daten einlesen
signal_pandas = pd.read_csv("data/signal_auch_noch_ohne_IDs.csv", delimiter=",")
background_pandas = pd.read_csv("data/background_auch_noch_ohne_IDs.csv", delimiter=",")
signal = signal_pandas.values
background = background_pandas.values

len_sig, width_sig = signal.shape
for i in range(len_sig):
    for j in range(width_sig):
        if np.isnan(signal[i,j]):
            signal[i,j] = np.nanmedian(signal[i,:])

len_bg, width_bg = background.shape
for i in range(len_bg):
    for j in range(width_bg):
        if np.isnan(background[i,j]):
            background[i,j] = np.nanmedian(background[i,:])

np.savetxt("data/signal_bereinigt.csv", signal, delimiter=",")
np.savetxt("data/background_bereinigt.csv", background, delimiter=",")
