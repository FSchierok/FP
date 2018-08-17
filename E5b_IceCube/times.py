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

# Import

nb = np.array([24.13, 21.87, 23.22, 23.37, 22.77])
nn = np.array([13.22, 12.91, 14.88, 14.64, 14.64])
rf = np.array([13.64, 13.47, 14.08, 15.11, 12.72])
print(np.mean(nb), fehler(nb))
print(np.mean(nn), fehler(nn))
print(np.mean(rf), fehler(rf))
