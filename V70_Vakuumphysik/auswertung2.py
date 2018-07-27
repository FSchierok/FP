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

x_1 = np.linspace(0.06, 0.4, 1000)
y_1 = np.ones(len(x_1))*0.64
y_1eo = np.ones(len(x_1))*(0.64 + 0.05)
y_1eu = np.ones(len(x_1))*(0.64 - 0.05)
x_2 = np.linspace(0.6, 10, 1000)
y_2 = np.ones(len(x_1))*1.08
y_2eo = np.ones(len(x_1))*(1.08 + 0.08)
y_2eu = np.ones(len(x_1))*(1.08 - 0.08)
x_3 = np.linspace(20, 1000, 1000)
y_3 = np.ones(len(x_1))*0.78
y_3eo = np.ones(len(x_1))*(0.78 + 0.09)
y_3eu = np.ones(len(x_1))*(0.78 - 0.09)
plt.plot(x_1, y_1, "-", color = "C1", label="Sauggeschwindigkeit aus Evakuierungskurve")
plt.plot(x_1, y_1eo, "--", color = "C1")
plt.plot(x_1, y_1eu, "--", color = "C1")
plt.plot(x_2, y_2, "-", color = "C1")
plt.plot(x_2, y_2eo, "--", color = "C1")
plt.plot(x_2, y_2eu, "--", color = "C1")
plt.plot(x_3, y_3, "-", color = "C1")
plt.plot(x_3, y_3eo, "--", color = "C1")
plt.plot(x_3, y_3eu, "--", color = "C1")
plt.errorbar([0.1, 0.4, 0.6, 1], [0.56, 1.0, 1.2, 1.3], xerr = [0.01, 0.04, 0.06, 0.1], yerr = [0.07, 0.1, 0.2, 0.2], fmt = ".", label="Sauggeschwindigkeit aus Leckratenmessung", color = "C0")
plt.legend(loc="best")
plt.grid()
#plt.xlim(-0.8, 21)
#plt.ylim(-0.2, 11.4)
plt.xlabel(r"$p \, / \, \text{mbar}$")
plt.ylabel(r"$S \, / \, \text{(l / s)}$")
plt.xscale("log")
plt.tight_layout()
plt.savefig("img/DrehSaug.pdf")
