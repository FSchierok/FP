import matplotlib as mpl
import time
mpl.use('pgf')
mpl.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'lualatex',
    'pgf.preamble': r'\usepackage{unicode-math}\usepackage{siunitx}',
    'errorbar.capsize': 3
})
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
import scipy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import pandas as pd
from sklearn import metrics
np.random.seed(2)

# Input
signal_pandas = pd.read_csv('data/signal_auch_noch_ohne_IDs.csv', sep=',')
signal = signal_pandas.values
background_pandas = pd.read_csv('data/background_auch_noch_ohne_IDs.csv', sep=',')
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
        if np.isnan(data[i,j]):
            data[i,j] = np.nanmedian(data[i,:])

# X sind die Beispiele und y ihre Klassifikation
data_X = data[:,1:]
data_y = data[:,0]

# feature selection
from sklearn.feature_selection import SelectKBest
def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier



sel = SelectKBest(score_func=f_regression, k=26)
data_X_selected = sel.fit_transform(data_X, data_y)
#print(sel.get_support(indices = True) )

# Teile in Test- und Trainingsdaten auf. Shuffle mit seed 42
data_train_X, data_test_X, data_train_y, data_test_y  = train_test_split(data_X_selected, data_y, test_size=0.25, random_state=42)

auc = np.zeros(200)

for n_trees in range(1, 200):
    # Lernen und überprüfen mit k next neighbors
    forest = RandomForestClassifier(n_estimators=n_trees)
    forest.fit(data_train_X, data_train_y)
    expected = data_test_y
    predicted_probs = forest.predict_proba(data_test_X)
    print("# = " ,n_trees, ": AUC = ", roc_auc_score(expected, predicted_probs[:, 1]))
    auc[n_trees] = roc_auc_score(expected, predicted_probs[:, 1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(1, 200), n_trees, "-")
ax.set_ylabel("AUC")
ax.set_xlabel("Anzahl Entscheidungsbäume")
ax.set_xlim([1,200])
fig.savefig("plots/number_trees.pdf")
