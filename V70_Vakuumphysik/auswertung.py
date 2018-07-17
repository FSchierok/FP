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

# Drehschieber Evakuierungskurve
drehschieber_pmess, drehschieber_t1, drehschieber_t2, drehschieber_t3, drehschieber_t4, drehschieber_t5 = np.genfromtxt("data/drehschieber.txt", unpack=True)
drehschieber_meant= 0.2 * (drehschieber_t1 + drehschieber_t2 + drehschieber_t3 + drehschieber_t4 + drehschieber_t5)
drehschieber_errt = np.sqrt(1/(5*(5-1)) * ((drehschieber_meant - drehschieber_t1)**2 +
 (drehschieber_meant - drehschieber_t2)**2 +
 (drehschieber_meant - drehschieber_t3)**2 +
 (drehschieber_meant - drehschieber_t4)**2 +
 (drehschieber_meant - drehschieber_t5)**2
 ))
drehschieber_t = unp.uarray(drehschieber_meant, drehschieber_errt)
drehschieber_pende = 0.05 #mbar
drehschieber_pendeerr = 0.005 #mbar
drehschieber_pstart = 1013 #mbar
drehschieber_pstarterr = 3 #mbar
drehschieber_errpmess = 0.10 * drehschieber_pmess
drehschieber_p = unp.uarray(drehschieber_pmess, np.sqrt(
(drehschieber_errpmess/(drehschieber_pmess - drehschieber_pende))**2
+ (drehschieber_pstarterr/(drehschieber_pstart - drehschieber_pende))**2
+ ((drehschieber_pmess - drehschieber_pstart)/(drehschieber_pmess - drehschieber_pende)/(drehschieber_pstart - drehschieber_pende) * drehschieber_pendeerr)**2
))

drehschieber_linreg0p = drehschieber_p[0:1]
drehschieber_linreg0t = drehschieber_t[0:1]

drehschieber_linreg1p = drehschieber_p[1:5]
drehschieber_linreg1t = drehschieber_t[1:5]

drehschieber_linreg2p = drehschieber_p[5:13]
drehschieber_linreg2t = drehschieber_t[5:13]

drehschieber_linreg3p = drehschieber_p[13:18]
drehschieber_linreg3t = drehschieber_t[13:18]

params_drehschieber_1, cov_drehschieber1 = curve_fit(lin, noms(drehschieber_linreg1t), np.log((noms(drehschieber_linreg1p)-drehschieber_pende)/(drehschieber_pstart-drehschieber_pende)))
error_drehschieber1 = np.sqrt(np.diag(cov_drehschieber1))

params_drehschieber_2, cov_drehschieber2 = curve_fit(lin, noms(drehschieber_linreg2t), np.log((noms(drehschieber_linreg2p)-drehschieber_pende)/(drehschieber_pstart-drehschieber_pende)))
error_drehschieber2 = np.sqrt(np.diag(cov_drehschieber2))

params_drehschieber_3, cov_drehschieber3 = curve_fit(lin, noms(drehschieber_linreg3t), np.log((noms(drehschieber_linreg3p)-drehschieber_pende)/(drehschieber_pstart-drehschieber_pende)))
error_drehschieber3 = np.sqrt(np.diag(cov_drehschieber3))

plt.figure(1)
x = np.linspace(-20, 200, 1000)

plt.errorbar(noms(drehschieber_linreg0t), np.log((noms(drehschieber_linreg0p)-drehschieber_pende)/(drehschieber_pstart-drehschieber_pende)), xerr = stds(drehschieber_linreg0t), yerr = stds(drehschieber_linreg0p),fmt = ".", label="Messdaten", color = "black")


plt.plot(x, lin(x, *params_drehschieber_1), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(drehschieber_linreg1t), np.log((noms(drehschieber_linreg1p)-drehschieber_pende)/(drehschieber_pstart-drehschieber_pende)), xerr = stds(drehschieber_linreg1t), yerr = stds(drehschieber_linreg1p),fmt = ".", label="Messdaten", color = "C0")

plt.plot(x, lin(x, *params_drehschieber_2), "-", label="Regressionsgerade", color = "C1")
plt.errorbar(noms(drehschieber_linreg2t), np.log((noms(drehschieber_linreg2p)-drehschieber_pende)/(drehschieber_pstart-drehschieber_pende)), xerr = stds(drehschieber_linreg2t), yerr = stds(drehschieber_linreg2p),fmt = ".", label="Messdaten", color = "C1")

plt.plot(x, lin(x, *params_drehschieber_3), "-", label="Regressionsgerade", color = "C2")
plt.errorbar(noms(drehschieber_linreg3t), np.log((noms(drehschieber_linreg3p)-drehschieber_pende)/(drehschieber_pstart-drehschieber_pende)), xerr = stds(drehschieber_linreg3t), yerr = stds(drehschieber_linreg3p),fmt = ".", label="Messdaten", color = "C2")


plt.title("Evakuierungskurve der Drehschieberpumpe")
plt.legend(loc="best")
plt.grid()
plt.xlim(-10, 162)
plt.ylim(-12.8, 0.8)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$\ln \left( \frac{p(t) - p_\text{E}}{p_0 - p_\text{E}}\right)$")
plt.savefig("img/drehschieber.pdf")





# Turbo Evakuierungskurve
turbo_pmess, turbo_t1, turbo_t2, turbo_t3, turbo_t4, turbo_t5 = np.genfromtxt("data/turbo.txt", unpack=True)
turbo_meant= 0.2 * (turbo_t1 + turbo_t2 + turbo_t3 + turbo_t4 + turbo_t5)
turbo_errt = np.sqrt(1/(5*(5-1)) * ((turbo_meant - turbo_t1)**2 +
 (turbo_meant - turbo_t2)**2 +
 (turbo_meant - turbo_t3)**2 +
 (turbo_meant - turbo_t4)**2 +
 (turbo_meant - turbo_t5)**2
 ))
turbo_t = unp.uarray(turbo_meant, turbo_errt)
turbo_pende = 0.015#µbar
turbo_pendeerr = 0.005 #µbar
turbo_pstart = 5 #µbar
turbo_pstarterr = 1 #µbar
turbo_errpmess = 0.10 * turbo_pmess
turbo_p = unp.uarray(turbo_pmess, np.sqrt(
(turbo_errpmess/(turbo_pmess - turbo_pende))**2
+ (turbo_pstarterr/(turbo_pstart - turbo_pende))**2
+ ((turbo_pmess - turbo_pstart)/(turbo_pmess - turbo_pende)/(turbo_pstart - turbo_pende) * turbo_pendeerr)**2
))


turbo_linreg1p = turbo_p[0:6]
turbo_linreg1t = turbo_t[0:6]

turbo_linreg2p = turbo_p[6:9]
turbo_linreg2t = turbo_t[6:9]


params_turbo_1, cov_turbo1 = curve_fit(lin, noms(turbo_linreg1t), np.log((noms(turbo_linreg1p)-turbo_pende)/(turbo_pstart-turbo_pende)))
error_turbo1 = np.sqrt(np.diag(cov_turbo1))

params_turbo_2, cov_turbo2 = curve_fit(lin, noms(turbo_linreg2t), np.log((noms(turbo_linreg2p)-turbo_pende)/(turbo_pstart-turbo_pende)))
error_turbo2 = np.sqrt(np.diag(cov_turbo2))

plt.figure(2)

plt.plot(x, lin(x, *params_turbo_1), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(turbo_linreg1t), np.log((noms(turbo_linreg1p)-turbo_pende)/(turbo_pstart-turbo_pende)), xerr = stds(turbo_linreg1t), yerr = stds(turbo_linreg1p),fmt = ".", label="Messdaten", color = "C0")

plt.plot(x, lin(x, *params_turbo_2), "-", label="Regressionsgerade", color = "C1")
plt.errorbar(noms(turbo_linreg2t), np.log((noms(turbo_linreg2p)-turbo_pende)/(turbo_pstart-turbo_pende)), xerr = stds(turbo_linreg2t), yerr = stds(turbo_linreg2p),fmt = ".", label="Messdaten", color = "C1")

#plt.errorbar(noms(turbo_t), np.log((noms(turbo_p)-turbo_pende)/(turbo_pstart-turbo_pende)), xerr = stds(turbo_t), yerr = stds(turbo_p),fmt = ".", label="Messdaten")
plt.title("Evakuierungskurve der Turbomolekularpumpe")
plt.legend(loc="best")
plt.grid()
plt.xlim(-1, 26)
plt.ylim(-8.3, 0.9)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$\ln \left( \frac{p(t) - p_\text{E}}{p_0 - p_\text{E}}\right)$")
plt.savefig("img/turbo.pdf")


# Leckratenmessung Drehschieber 0.1mbar
drehLeck01_pmess, drehLeck01_t1, drehLeck01_t2, drehLeck01_t3 = np.genfromtxt("data/drehLeck01.txt", unpack=True)
drehLeck01_meant= 1/3 * (drehLeck01_t1 + drehLeck01_t2 + drehLeck01_t3)
drehLeck01_errt = np.sqrt(1/(3*(3-1)) * ((drehLeck01_meant - drehLeck01_t1)**2 +
 (drehLeck01_meant - drehLeck01_t2)**2 +
 (drehLeck01_meant - drehLeck01_t3)**2
 ))
drehLeck01_t = unp.uarray(drehLeck01_meant, drehLeck01_errt)

drehLeck01_errp = 0.1 * drehLeck01_pmess
drehLeck01_p = unp.uarray(drehLeck01_pmess, drehLeck01_errp)

params_drehLeck01, cov_drehLeck01 = curve_fit(lin, noms(drehLeck01_t), noms(drehLeck01_p))
error_drehLeck01 = np.sqrt(np.diag(cov_drehLeck01))

plt.figure(3)
plt.subplot(2, 2, 1)
#plt.rcParams['figure.figsize'] = (10, 10)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_drehLeck01), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(drehLeck01_t), noms(drehLeck01_p), xerr = stds(drehLeck01_t), yerr = stds(drehLeck01_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-10, 180)
plt.ylim(0.05, 1.19)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 0.1 \, \text{mbar}$")



# Leckratenmessung Drehschieber 0.4mbar
drehLeck04_pmess, drehLeck04_t1, drehLeck04_t2, drehLeck04_t3 = np.genfromtxt("data/drehLeck04.txt", unpack=True)
drehLeck04_meant= 1/3 * (drehLeck04_t1 + drehLeck04_t2 + drehLeck04_t3)
drehLeck04_errt = np.sqrt(1/(3*(3-1)) * ((drehLeck04_meant - drehLeck04_t1)**2 +
 (drehLeck04_meant - drehLeck04_t2)**2 +
 (drehLeck04_meant - drehLeck04_t3)**2
 ))
drehLeck04_t = unp.uarray(drehLeck04_meant, drehLeck04_errt)

drehLeck04_errp = 0.1 * drehLeck04_pmess
drehLeck04_p = unp.uarray(drehLeck04_pmess, drehLeck04_errp)

params_drehLeck04, cov_drehLeck04 = curve_fit(lin, noms(drehLeck04_t), noms(drehLeck04_p))
error_drehLeck04 = np.sqrt(np.diag(cov_drehLeck04))

plt.subplot(2, 2, 2)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_drehLeck04), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(drehLeck04_t), noms(drehLeck04_p), xerr = stds(drehLeck04_t), yerr = stds(drehLeck04_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-5, 110)
plt.ylim(0.1, 4.6)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 0.4 \, \text{mbar}$")




# Leckratenmessung Drehschieber 0.6mbar
drehLeck06_pmess, drehLeck06_t1, drehLeck06_t2, drehLeck06_t3 = np.genfromtxt("data/drehLeck06.txt", unpack=True)
drehLeck06_meant= 1/3 * (drehLeck06_t1 + drehLeck06_t2 + drehLeck06_t3)
drehLeck06_errt = np.sqrt(1/(3*(3-1)) * ((drehLeck06_meant - drehLeck06_t1)**2 +
 (drehLeck06_meant - drehLeck06_t2)**2 +
 (drehLeck06_meant - drehLeck06_t3)**2
 ))
drehLeck06_t = unp.uarray(drehLeck06_meant, drehLeck06_errt)

drehLeck06_errp = 0.1 * drehLeck06_pmess
drehLeck06_p = unp.uarray(drehLeck06_pmess, drehLeck06_errp)

params_drehLeck06, cov_drehLeck06 = curve_fit(lin, noms(drehLeck06_t), noms(drehLeck06_p))
error_drehLeck06 = np.sqrt(np.diag(cov_drehLeck06))

plt.subplot(2, 2, 3)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_drehLeck06), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(drehLeck06_t), noms(drehLeck06_p), xerr = stds(drehLeck06_t), yerr = stds(drehLeck06_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-4, 90)
plt.ylim(0.1, 6.9)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 0.6 \, \text{mbar}$")




# Leckratenmessung Drehschieber 1mbar
drehLeck1_pmess, drehLeck1_t1, drehLeck1_t2, drehLeck1_t3 = np.genfromtxt("data/drehLeck1.txt", unpack=True)
drehLeck1_meant= 1/3 * (drehLeck1_t1 + drehLeck1_t2 + drehLeck1_t3)
drehLeck1_errt = np.sqrt(1/(3*(3-1)) * ((drehLeck1_meant - drehLeck1_t1)**2 +
 (drehLeck1_meant - drehLeck1_t2)**2 +
 (drehLeck1_meant - drehLeck1_t3)**2
 ))
drehLeck1_t = unp.uarray(drehLeck1_meant, drehLeck1_errt)

drehLeck1_errp = 0.1 * drehLeck1_pmess
drehLeck1_p = unp.uarray(drehLeck1_pmess, drehLeck1_errp)

params_drehLeck1, cov_drehLeck1 = curve_fit(lin, noms(drehLeck1_t), noms(drehLeck1_p))
error_drehLeck1 = np.sqrt(np.diag(cov_drehLeck1))

plt.subplot(2, 2, 4)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_drehLeck1), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(drehLeck1_t), noms(drehLeck1_p), xerr = stds(drehLeck1_t), yerr = stds(drehLeck1_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-3, 80)
plt.ylim(0.2, 11.8)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 1 \, \text{mbar}$")
plt.tight_layout()
plt.savefig("img/drehLeck.pdf")

# Leckratenmessung Turbo 0.05 mbar
turboLeck005_pmess, turboLeck005_t1, turboLeck005_t2, turboLeck005_t3 = np.genfromtxt("data/turboLeck005.txt", unpack=True)
turboLeck005_meant= 1/3 * (turboLeck005_t1 + turboLeck005_t2 + turboLeck005_t3)
turboLeck005_errt = np.sqrt(1/(3*(3-1)) * ((turboLeck005_meant - turboLeck005_t1)**2 +
 (turboLeck005_meant - turboLeck005_t2)**2 +
 (turboLeck005_meant - turboLeck005_t3)**2
 ))
turboLeck005_t = unp.uarray(turboLeck005_meant, turboLeck005_errt)

turboLeck005_errp = 0.1 * turboLeck005_pmess
turboLeck005_p = unp.uarray(turboLeck005_pmess, turboLeck005_errp)

params_turboLeck005, cov_turboLeck005 = curve_fit(lin, noms(turboLeck005_t), noms(turboLeck005_p))
error_turboLeck005 = np.sqrt(np.diag(cov_turboLeck005))

plt.figure(4)
plt.subplot(2, 2, 1)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_turboLeck005), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(turboLeck005_t), noms(turboLeck005_p), xerr = stds(turboLeck005_t), yerr = stds(turboLeck005_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-3, 63)
plt.ylim(-0.1, 4.5)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 0.05 \, \text{mbar}$")


# Leckratenmessung Turbo 0.1 mbar
turboLeck01_pmess, turboLeck01_t1, turboLeck01_t2, turboLeck01_t3 = np.genfromtxt("data/turboLeck01.txt", unpack=True)
turboLeck01_meant= 1/3 * (turboLeck01_t1 + turboLeck01_t2 + turboLeck01_t3)
turboLeck01_errt = np.sqrt(1/(3*(3-1)) * ((turboLeck01_meant - turboLeck01_t1)**2 +
 (turboLeck01_meant - turboLeck01_t2)**2 +
 (turboLeck01_meant - turboLeck01_t3)**2
 ))
turboLeck01_t = unp.uarray(turboLeck01_meant, turboLeck01_errt)

turboLeck01_errp = 0.1 * turboLeck01_pmess
turboLeck01_p = unp.uarray(turboLeck01_pmess, turboLeck01_errp)

params_turboLeck01, cov_turboLeck01 = curve_fit(lin, noms(turboLeck01_t), noms(turboLeck01_p))
error_turboLeck01 = np.sqrt(np.diag(cov_turboLeck01))

plt.subplot(2, 2, 2)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_turboLeck01), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(turboLeck01_t), noms(turboLeck01_p), xerr = stds(turboLeck01_t), yerr = stds(turboLeck01_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-2, 43)
plt.ylim(-0.3, 11.7)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 0.1 \, \text{mbar}$")


# Leckratenmessung Turbo 0.15 mbar
turboLeck015_pmess, turboLeck015_t1, turboLeck015_t2, turboLeck015_t3 = np.genfromtxt("data/turboLeck015.txt", unpack=True)
turboLeck015_meant= 1/3 * (turboLeck015_t1 + turboLeck015_t2 + turboLeck015_t3)
turboLeck015_errt = np.sqrt(1/(3*(3-1)) * ((turboLeck015_meant - turboLeck015_t1)**2 +
 (turboLeck015_meant - turboLeck015_t2)**2 +
 (turboLeck015_meant - turboLeck015_t3)**2
 ))
turboLeck015_t = unp.uarray(turboLeck015_meant, turboLeck015_errt)

turboLeck015_errp = 0.1 * turboLeck015_pmess
turboLeck015_p = unp.uarray(turboLeck015_pmess, turboLeck015_errp)

params_turboLeck015, cov_turboLeck015 = curve_fit(lin, noms(turboLeck015_t), noms(turboLeck015_p))
error_turboLeck015 = np.sqrt(np.diag(cov_turboLeck015))

plt.subplot(2, 2, 3)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_turboLeck015), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(turboLeck015_t), noms(turboLeck015_p), xerr = stds(turboLeck015_t), yerr = stds(turboLeck015_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-1, 30)
plt.ylim(-0.3, 11.2)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 0.15 \, \text{mbar}$")


# Leckratenmessung Turbo 0.2 mbar
turboLeck02_pmess, turboLeck02_t1, turboLeck02_t2, turboLeck02_t3 = np.genfromtxt("data/turboLeck02.txt", unpack=True)
turboLeck02_meant= 1/3 * (turboLeck02_t1 + turboLeck02_t2 + turboLeck02_t3)
turboLeck02_errt = np.sqrt(1/(3*(3-1)) * ((turboLeck02_meant - turboLeck02_t1)**2 +
 (turboLeck02_meant - turboLeck02_t2)**2 +
 (turboLeck02_meant - turboLeck02_t3)**2
 ))
turboLeck02_t = unp.uarray(turboLeck02_meant, turboLeck02_errt)

turboLeck02_errp = 0.1 * turboLeck02_pmess
turboLeck02_p = unp.uarray(turboLeck02_pmess, turboLeck02_errp)

params_turboLeck02, cov_turboLeck02 = curve_fit(lin, noms(turboLeck02_t), noms(turboLeck02_p))
error_turboLeck02 = np.sqrt(np.diag(cov_turboLeck02))

plt.subplot(2, 2, 4)
x = np.linspace(-20, 200, 1000)
plt.plot(x, lin(x, *params_turboLeck02), "-", label="Regressionsgerade", color = "C0")
plt.errorbar(noms(turboLeck02_t), noms(turboLeck02_p), xerr = stds(turboLeck02_t), yerr = stds(turboLeck02_p),fmt = ".", label="Messdaten", color = "C0")
plt.legend(loc="best")
plt.grid()
plt.xlim(-0.8, 21)
plt.ylim(-0.2, 11.4)
plt.xlabel(r"$t \, / \, \text{s}$")
plt.ylabel(r"$p(t) \, / \, \text{mbar}$")
plt.title(r"$p_0 = 0.2 \, \text{mbar}$")
plt.tight_layout()
plt.savefig("img/turboLeck.pdf")




# Gesamtauswertung
# Evakuierung
Vdreh = unp.uarray(12.0, 0.9) # Liter
a_dreh1 = unp.uarray(params_drehschieber_1[0], error_drehschieber1[0])
a_dreh2 = unp.uarray(params_drehschieber_2[0], error_drehschieber2[0])
a_dreh3 = unp.uarray(params_drehschieber_3[0], error_drehschieber3[0])
S_dreh1 = - Vdreh * a_dreh1
S_dreh2 = - Vdreh * a_dreh2
S_dreh3 = - Vdreh * a_dreh3

Vturbo = unp.uarray(12.5, 0.9) # Liter
a_turbo1 = unp.uarray(params_turbo_1[0], error_turbo1[0])
a_turbo2 = unp.uarray(params_turbo_2[0], error_turbo2[0])
S_turbo1 = - Vturbo * a_turbo1
S_turbo2 = - Vturbo * a_turbo2

#Leckratenmessung
p0_dreh01 = drehLeck01_p[0]
p0_dreh04 = drehLeck04_p[0]
p0_dreh06 = drehLeck06_p[0]
p0_dreh1 = drehLeck1_p[0]
p0_turbo01 = turboLeck01_p[0]
p0_turbo02 = turboLeck02_p[0]
p0_turbo005 = turboLeck005_p[0]
p0_turbo015 = turboLeck015_p[0]

a_drehLeck01 = unp.uarray(params_drehLeck01[0], error_drehLeck01[0])
a_drehLeck04 = unp.uarray(params_drehLeck04[0], error_drehLeck04[0])
a_drehLeck06 = unp.uarray(params_drehLeck06[0], error_drehLeck06[0])
a_drehLeck1 = unp.uarray(params_drehLeck1[0], error_drehLeck1[0])
a_turboLeck01 = unp.uarray(params_turboLeck01[0], error_turboLeck01[0])
a_turboLeck02 = unp.uarray(params_turboLeck02[0], error_turboLeck02[0])
a_turboLeck005 = unp.uarray(params_turboLeck005[0], error_turboLeck005[0])
a_turboLeck015 = unp.uarray(params_turboLeck015[0], error_turboLeck015[0])

Q_drehLeck01 = Vdreh * a_drehLeck01
Q_drehLeck04 = Vdreh * a_drehLeck04
Q_drehLeck06 = Vdreh * a_drehLeck06
Q_drehLeck1 = Vdreh * a_drehLeck1
Q_turboLeck01 = Vturbo * a_turboLeck01
Q_turboLeck02 = Vturbo * a_turboLeck02
Q_turboLeck005 = Vturbo * a_turboLeck005
Q_turboLeck015 = Vturbo * a_turboLeck015

S_drehLeck01 = Q_drehLeck01 / p0_dreh01
S_drehLeck04 = Q_drehLeck04 / p0_dreh04
S_drehLeck06 = Q_drehLeck06 / p0_dreh06
S_drehLeck1 = Q_drehLeck1 / p0_dreh1
S_turboLeck01 = Q_turboLeck01 / p0_turbo01
S_turboLeck02 = Q_turboLeck02 / p0_turbo02
S_turboLeck005 = Q_turboLeck005 / p0_turbo005
S_turboLeck015 = Q_turboLeck015 / p0_turbo015


print(S_drehLeck01)
print(S_drehLeck04)
print(S_drehLeck06)
print(S_drehLeck1)
print(S_turboLeck01)
print(S_turboLeck02)
print(S_turboLeck005)
print(S_turboLeck015)
