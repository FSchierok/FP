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
signal_pandas = pd.read_csv('data/signal_auch_ohne_MC.csv', sep=',')
signal = signal_pandas.values
background_pandas = pd.read_csv('data/background_auch_ohne_MC.csv', sep=',')
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
        if (np.isnan(data[i,j]) or np.isinf(data[i,j])):
            data[i,j] = np.nanmedian(data[i,:])

# X sind die Beispiele und y ihre Klassifikation
data_X = data[:,1:]
data_y = data[:,0]

# feature selection
from sklearn.feature_selection import SelectKBest
def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False)
data_X_selected = SelectKBest(score_func=f_regression, k=26).fit_transform(data_X, data_y)



from sklearn.model_selection import train_test_split
data_train_X, data_test_X, data_train_y, data_test_y  = train_test_split(data_X_selected, data_y, test_size=0.25, random_state=42)


# Lernen und überprüfen mit Naive Bayes
from sklearn.naive_bayes import GaussianNB
nbayes = GaussianNB()
nbayes.fit(data_train_X, data_train_y)
expected = data_test_y
predicted = nbayes.predict(data_test_X)
predicted_probs = nbayes.predict_proba(data_test_X)


# ROC
from sklearn.metrics import roc_curve, roc_auc_score
y_score = predicted_probs[:, 1]
fprate, tprate, threshold = roc_curve(expected, y_score)
auc = roc_auc_score(expected, y_score)
# Plot ROC curve
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.set_title('Receiver Operating Characteristic')
ax1.plot(fprate, tprate, label = "AUC = %0.2f" % auc)
ax1.plot([0, 1], ls="--")
ax1.set_ylabel('True Positive Rate')
ax1.set_xlabel('False Positive Rate')
ax1.legend()
fig1.savefig("plots/bayes/ROC.pdf")

# Plot Scoreverteilung, Ziel: https://docs.aws.amazon.com/machine-learning/latest/dg/binary-classification.html
from sklearn.metrics import confusion_matrix
def num_tp(score):
    predicted = (predicted_probs[:,1] > score).astype(bool) # https://stackoverflow.com/questions/49785904/how-to-set-threshold-to-scikit-learn-random-forest-model
    return confusion_matrix(expected, predicted)[1, 0]

def num_tn(score):
    predicted = (predicted_probs[:,1] >= score).astype(bool)
    return confusion_matrix(expected, predicted)[0, 1]

score = np.linspace(0, 1, 1000)
num_tp1 = np.zeros(1000)
num_tn1 = np.zeros(1000)
for i in range(len(score)):
    num_tp1[i] = num_tp(score[i])
    num_tn1[i] = num_tn(score[i])

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.set_title('Scoreverteilung')
score = np.linspace(0, 1, 1000)
ax3.plot(score, num_tp1, "-", label = "Signal")
ax3.plot(score, num_tn1, "-", label = "Background")
ax3.set_ylabel("Anzahl")
ax3.set_xlabel("Score")
ax3.legend()
fig3.savefig("plots/bayes/Scoredistribution.pdf")

# precision recall threshold curve
# https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn, http://www.scikit-yb.org/en/latest/api/classifier/threshold.html
from yellowbrick.classifier import DiscriminationThreshold
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)
visualizer = DiscriminationThreshold(nbayes, exclude = ("queue_rate", "fscore"), ax = ax5)
visualizer.fit(data_train_X, data_train_y)  # Fit the training data to the visualizer
visualizer.poof(outpath="plots/bayes/prec_reca_thresh.pdf")     # Draw/show/poof the data

print(time.clock())

print(confusion_matrix(expected, (predicted_probs[:,1] > 0.6).astype(bool)))
from sklearn.metrics import classification_report
print(classification_report(expected, (predicted_probs[:,1] > 0.6).astype(bool)))
