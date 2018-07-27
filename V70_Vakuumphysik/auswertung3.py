# Tabellenprogramm https://github.com/TheChymera/matrix2latex
import os
import sys
sys.path.insert(0, '../global/matrix2latex-master/')
from matrix2latex import matrix2latex

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
import math as m


def fehler(x):
    return np.std(x, ddof=1) / len(x)

def lin(x, a, b):
    return a*x + b

x_1 = np.linspace(0.06, 5, 1000)
y_1 = np.ones(len(x_1))*8.7
y_1eo = np.ones(len(x_1))*(8.7 + 0.8)
y_1eu = np.ones(len(x_1))*(8.7 - 0.8)
x_2 = np.linspace(0.02, 0.04, 1000)
y_2 = np.ones(len(x_1))*1.0
y_2eo = np.ones(len(x_1))*(1.0 + 0.2)
y_2eu = np.ones(len(x_1))*(1.0 - 0.2)
plt.plot(x_1, y_1, "-", color = "C1", label="Sauggeschwindigkeit aus Evakuierungskurve")
plt.plot(x_1, y_1eo, "--", color = "C1")
plt.plot(x_1, y_1eu, "--", color = "C1")
plt.plot(x_2, y_2, "-", color = "C1")
plt.plot(x_2, y_2eo, "--", color = "C1")
plt.plot(x_2, y_2eu, "--", color = "C1")
plt.errorbar([0.05, 0.1, 0.15, 0.2], [15, 26, 26, 27], xerr = [0.005, 0.01, 0.015, 0.02], yerr = [2, 3, 3, 3], fmt = ".", label="Sauggeschwindigkeit aus Leckratenmessung", color = "C0")
plt.legend(loc="best")
plt.grid()
#plt.xlim(-0.8, 21)
#plt.ylim(-0.2, 11.4)
plt.xlabel(r"$p \, / \, \text{µbar}$")
plt.ylabel(r"$S \, / \, \text{(l / s)}$")
plt.xscale("log")
plt.tight_layout()
plt.savefig("img/TurboSaug.pdf")
