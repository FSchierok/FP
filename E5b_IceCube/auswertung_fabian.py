import sklearn as skl
import numpy as np
import pandas as pd

signal = pd.read_csv("data/signal.csv", delimiter=";")
bg = pd.read_csv("data/background.csv", delimiter=";")
# Enfernt alle Spalten, die nicht im Background sind, sowie alle mit Weight, MC oder Event im Namen. Alle ungültigen Werte werden durch den Mittelwert ersetzt
for category in signal:
    if category not in bg:
        del signal[category]
    elif "Weight" in str(category):
        del signal[category]
    elif "MC" in str(category):
        del signal[category]
    elif "Event" in str(category):
        del signal[category]
    else:
        for value in signal[category]:
            if np.isnan(value) or np.isinf(value):
                value = np.mean(signal[category])
for category in bg:
    if category not in signal:
        del bg[category]
signal.to_csv("data/signal_fabian.csv", sep=";")
bg.to_csv("data/bg_fabian.csv", sep=";")

data = pd.concat([signal, bg])
