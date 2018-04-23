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


# Daten einlesen und ausgeben:
# x = np.genfromtxt("data/x.txt", unpack=True) [Skalar oder Vektor]
# y = ufloat(y-Nominalwert, y-Fehler) [Skalar mit Fehler]
# z = unp.uarray(vec1, vec2) [Vektor mit Fehler]
# print(x)

# Fehler:
# noms(x) [Nominalwert]
# stds(x) [Fehlerwert]

# linfit:
# def f1(x, a, b):
#    return a*x + b
#
# params, cov = curve_fit(f2, x, y) [x und y = f(x) sind Eingabe]
# error = np.sqrt(np.diag(cov))

# plot:
# SHOW  = False
#
# x = np.linspace(0, 1, 1000)
# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, "-r", label=r"$f(x)$")
# plt.plot(x1, f1(x1, *params), "-g", label="Fit")
# plt.yticks([0, 10, 20, 30, 40, 50])
# plt.ylim(0, 60)
# plt.xticks([0, 2 * 10**-9], ["0", "2"])  [für z.b. nano-Einteilung]
# plt.legend(loc="best")
# plt.grid()
# plt.xlabel(r"$C_\mathrm{K} / \mathrm{nF}$")
# plt.ylabel(r"$f / \mathrm{kHz}$")
# plt.tight_layout()
# plt.show() if SHOW else plt.savefig("img/xyz.pdf")

# LaTex-Tabellen
# output = open("tab/ausgabe.tex","w")
# R = 50
# V = np.array([1, 20, 100])
# I = V/R
# cap = r"""Berechnete Ströme durch einen $\SI{%g}{\ohm}$-Widerstand.""" % R
# hr = [[r"$U$", r"$I$"], [r"\si{V}", r"\si{A}"]]
# t = matrix2latex([V, I], caption = cap, transpose = False, alignment = "S", format = "%.2f", headerRow = hr, label = "Labeltext")
# output.write(t)
# print(t)
