# Tabellenprogramm https://github.com/TheChymera/matrix2latex
# martin.berger(at)tu-
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


def np2ufl(x):
    return ufloat(np.mean(x), fehler(x))


def uarr2ufl(x):
    nom = noms(x)
    std = stds(x)
    mean = sum(nom / std**2) / sum(1 / std**2)
    err = np.sqrt(1 / sum(1 / std**2))
    return ufloat(mean, err)


# Import
luft = np.genfromtxt("data/luft.txt", unpack=True)
glas = np.genfromtxt("data/glas.txt", unpack=True)
T, lam, t, L_m, L_err = np.genfromtxt("data/setup.txt", unpack=True)

# Umrechnen
glas[1] = np.deg2rad(glas[1])  # deg -> rad
luft[3] = 100 * luft[3]  # mbar -> Pa
T = T + 273.15  # C -> K
lam = lam * 1e-9  # nm -> m
t = t * 1e-3  # mm -> m
L = ufloat(L_m, L_err) / 1000  # Fehler, mm -> m

# glas
n_glas_raw = 1 / (1 - ((glas[0] * lam) / (2 * t * glas[1]**2)))
print("n Glas: " + str(n_glas_raw))
n_glas = np2ufl(n_glas_raw)
print("n Glas Fehler: ", str(n_glas))


cap = r"Messwerte und daraus bestimmten Brechungsindeces $n_\text{Glas}$"
hr = [[r"Messung", r"\Theta", r"$M$", r"$n_\text{Glas}$"], [r"1", r"rad", r" 1", r"1 "]]
with open("tab/glas.tex", "w") as file:
    file.write(matrix2latex([np.arange(len(glas[0])), glas[1], glas[0], n_glas_raw], caption=cap, headerRow=hr, alignment="l", label=r"tab:glas", transpose=False))

# luft
n_luft_raw = unp.uarray([1, 1, 1], [0, 0, 0])
for i in range(3):
    n_luft_raw[i] = (luft[i][-1] * lam / L) + 1
    print("Messreihe " + str(i) + ": " + str(n_luft_raw[i]))
n_luft = uarr2ufl(n_luft_raw)
print("n_Luft: " + str(n_luft))
cap = r"Messwerte und daraus bestimmten Brechungsindeces $n_\text{Luft}$"
hr = [[r"Messung", r"$M$", r"$n_\text{Luft}$", r"\sigma"], [r" 1", r"1 ", r" 1 ", r"1"]]
with open("tab/luft.tex", "w") as file:
    file.write(matrix2latex([np.arange(len(n_luft_raw)), [luft[0][-1], luft[1][-1], luft[2][-1]], noms(n_luft_raw), stds(n_luft_raw)], caption=cap, headerRow=hr, alignment="l", label=r"tab:luft", transpose=False))
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
