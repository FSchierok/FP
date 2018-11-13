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



x, y = np.genfromtxt("auc_n_trees.txt", unpack = True)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(x, y, "-")
ax1.set_ylabel("AUC")
ax1.set_xlabel("Anzahl Entscheidungsbäume")
ax1.set_xlim([1,75])
fig1.savefig("plots/number_trees.pdf")
