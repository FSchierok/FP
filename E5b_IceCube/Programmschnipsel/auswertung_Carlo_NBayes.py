import matplotlib as mpl
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



import numpy as np
import pandas as pd
from sklearn import metrics
np.random.seed(2)

# Daten einlesen mit pandas. Konvertierung in numpy array. Format:
# (N_samples, N_features)
signal_pandas = pd.read_csv('data/signal_auch_noch_ohne_IDs.csv', sep=',')
signal = signal_pandas.values
background_pandas = pd.read_csv('data/background_auch_noch_ohne_IDs.csv', sep=',')
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

# Bereinigung: Entferne alle Nans
len_data, width_data = data.shape
for i in range(len_data):
    for j in range(width_data):
        if np.isnan(data[i,j]):
            data[i,j] = np.nanmedian(data[i,:])# median

# X sind die Beispiele und y ihre Klassifikation
data_X = data[:,1:]
data_y = data[:,0]

# Feature Selection
# http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
# https://stackoverflow.com/questions/15484011/scikit-learn-feature-selection-for-regression-data#
from sklearn.feature_selection import SelectKBest
def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False) #center=True (the default) would not work ("ValueError: center=True only allowed for dense data") but should presumably work in general
data_X_selected = SelectKBest(score_func=f_regression, k=10).fit_transform(data_X, data_y) # k = 20 da dadurch höchstmöglicher Jaccard-Index von J = 0.98325
# sklearn.feature_selection.VarianceThreshold
# http://www.scikit-yb.org/en/latest/api/features/rfecv.html

# Permutiere die Indizes von data zufällig beim Aufteilen in Trainings- und
# Testdaten (90% und 10% von Gesamtdaten).
#from sklearn.model_selection import train_test_split
perm = np.random.permutation(len(data))
data_train_y = data_y[perm[:-len(data_y) // 10]]
data_test_y  = data_y[perm[-len(data_y) // 10:]]
data_train_X = data_X_selected[perm[:-len(data_X_selected) // 10]]
data_test_X  = data_X_selected[perm[-len(data_X_selected) // 10:]]


# Lernen und überprüfen mit Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import jaccard_similarity_score
nbayes = GaussianNB()
nbayes.fit(data_train_X, data_train_y)
expected = data_test_y
predicted = nbayes.predict(data_test_X)
predicted_probs = nbayes.predict_proba(data_test_X)
#
# # ROC
# from sklearn.metrics import roc_curve, roc_auc_score
# y_score = predicted_probs[:, 1]
# fprate, tprate, threshold = roc_curve(expected, y_score)
# auc = roc_auc_score(expected, y_score)
# # Plot ROC curve
# fig1 = plt.figure(1)
# ax1 = fig1.add_subplot(111)
# ax1.set_title('Receiver Operating Characteristic')
# ax1.plot(fprate, tprate, label = "AUC = %0.2f" % auc)
# ax1.plot([0, 1], ls="--")
# ax1.set_ylabel('True Positive Rate')
# ax1.set_xlabel('False Positive Rate')
# ax1.legend()
# fig1.savefig("plots/ROC.pdf")
#
# # precision recall curve
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import average_precision_score
# average_precision = average_precision_score(expected, y_score)
# precision, recall, _ = precision_recall_curve(expected, y_score)
# fig2 = plt.figure(2)
# ax2 = fig2.add_subplot(111)
# ax2.step(recall, precision, color='b', alpha=0.2, where='post')
# ax2.fill_between(recall, precision, step='post', alpha=0.2, color='b')
# ax2.set_xlabel('Recall')
# ax2.set_ylabel('Precision')
# ax2.set_ylim([0.0, 1.05])
# ax2.set_xlim([0.0, 1.0])
# ax2.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
# fig2.savefig("plots/precision_recall_curve.pdf")
#
# print("Classification report:\n", metrics.classification_report(expected, predicted), "\n")
# print("Confusion Matrix:\n", metrics.confusion_matrix(expected, predicted), "\n")
# print("Jaccard-Index:\n", jaccard_similarity_score(expected,predicted), "\n")
#
# print("berechnete Wahrscheinlichkeiten:\n", predicted_probs, "\n")

from sklearn.metrics import confusion_matrix
def num_tp(score):
    predicted = (predicted_probs[:,1] >= score).astype(bool) # https://stackoverflow.com/questions/49785904/how-to-set-threshold-to-scikit-learn-random-forest-model
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


# Plot Scoreverteilung, Ziel: https://docs.aws.amazon.com/machine-learning/latest/dg/binary-classification.html
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.set_title('Scoreverteilung')
score = np.linspace(0, 1, 1000)
ax3.plot(score, num_tp1, "-", label = "\# of true positives")
ax3.plot(score, num_tn1, "-", label = "\# of true negatives")
#ax3.set_xlim([0, 1])
ax3.set_xlabel("Score")
ax3.legend()
fig3.savefig("plots/Scoredistribution.pdf")

# precision recall threshold cureve: https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn, http://www.scikit-yb.org/en/latest/api/classifier/threshold.html
# most important features: http://www.scikit-yb.org/en/latest/api/features/importances.html

# # precision recall threshold curve_fit
# from sklearn.linear_model import LogisticRegression
# from yellowbrick.classifier import DiscriminationThreshold
#
# # Instantiate the classification model and visualizer
# visualizer = DiscriminationThreshold(nbayes, exclude = ("queue_rate", "fscore"))
# visualizer.fit(data_train_X, data_train_y)  # Fit the training data to the visualizer
# visualizer.poof(outpath="plots/prec_reca_thresh.pdf")     # Draw/show/poof the data
