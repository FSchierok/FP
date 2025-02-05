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
freq, freqerr, revstat1, revsweep1, revstat2, revsweep2 = np.genfromtxt("data/dips.txt", unpack=True)

# Fehler, Umrechnung in Ströme
f       = 1e3 * unp.uarray(freq, freqerr)                               # kilohertz
Istat1  = 0.3 * unp.uarray(revstat1,    np.ones(len(revstat1))  * 0.02) # 0.3 Ampere / Umdrehung mal Anzahl Umdrehungen
Isweep1 = 0.1 * unp.uarray(revsweep1,   np.ones(len(revsweep1)) * 0.02) # 0.1 ...
Istat2  = 0.3 * unp.uarray(revstat2,    np.ones(len(revstat2))  * 0.02)
Isweep2 = 0.1 * unp.uarray(revsweep2,   np.ones(len(revsweep2)) * 0.02)

# Umrechnung von Strom in B Feld
# B = mu0 8/sqrt(125) I N / R  = c I N / R   mit mu0 = 4 pi 10^-7 H/m
c = 4 * np.pi * 1e-7 * 8 / np.sqrt(125)
Nstat   = 154
Nsweep  = 11
Rstat   = 0.1579 # m
Rsweep  = 0.1639
Bstat1  = c * Istat1    * Nstat     / Rstat
Bsweep1 = c * Isweep1   * Nsweep    / Rsweep
Bstat2  = c * Istat2    * Nstat     / Rstat
Bsweep2 = c * Isweep2   * Nsweep    / Rsweep
B1 = Bstat1 + Bsweep1
B2 = Bstat2 + Bsweep2

#linfit
def lin(x, a, b):
    return a*x + b
params1, cov1 = curve_fit(lin, noms(f), noms(B1))
error1 = np.sqrt(np.diag(cov1))
params2, cov2 = curve_fit(lin, noms(f), noms(B2))
error2 = np.sqrt(np.diag(cov2))

print("Steigung 1 (Rb87) in 10⁻¹⁰ T/Hz:", params1[0]*1e10, "+-", error1[0]*1e10)
print("Steigung 2 (Rb85) in 10⁻¹⁰ T/Hz:", params2[0]*1e10, "+-", error2[0]*1e10)
print("Achsenabschnitt 1 (Rb87) in 10⁻⁵ T:", params1[1]*1e5, "+-", error1[1]*1e5)
print("Achsenabschnitt 2 (Rb85) in 10⁻⁵ T:", params2[1]*1e5, "+-", error2[1]*1e5)

# plot
#SHOW  = True
#plt.figure(1)
plt.errorbar(noms(f)*1e-3, noms(B1)*1e6, xerr=stds(f)*1e-3, yerr=stds(B1)*1e6, fmt = ".", label = r"$\mathrm{Messwerte}\;{}^{87}\mathrm{Rb}$")
#plt.plot(noms(f)*1e-3, noms(B1)*1e6, "+", label=r"$B_1$")
plt.errorbar(noms(f)*1e-3, noms(B2)*1e6, xerr=stds(f)*1e-3, yerr=stds(B2)*1e6, fmt = ".", label = r"$\mathrm{Messwerte}\;{}^{85}\mathrm{Rb}$")
#plt.plot(noms(f)*1e-3, noms(B2)*1e6, "+", label=r"$B_2$")
x = np.linspace(-30, 1030, 1000)
plt.plot(x, 1e6*lin(x*1e3, *params1), "C0", label=r"$\text{linearer Fit}\;{}^{87}\mathrm{Rb}$")
plt.plot(x, 1e6*lin(x*1e3, *params2), "C1", label=r"$\text{linearer Fit}\;{}^{85}\mathrm{Rb}$")
#plt.yticks([0, 10, 20, 30, 40, 50])
plt.xlim(-30, 1030)
#plt.xticks([0, 2 * 10**-9], ["0", "2"])  [für z.b. nano-Einteilung]
plt.legend(loc="best")
plt.grid()
plt.xlabel(r"$f / \mathrm{kHz}$")
plt.ylabel(r"$B / \mathrm{µ T}$")
plt.savefig("img/plot1.pdf")

# Erdmagnetfeld horizontal
Bhor1 = unp.uarray(params1[1], error1[1])
Bhor2 = unp.uarray(params2[1], error2[1])
Bhor = 0.5 * (Bhor1 + Bhor2)
print("Erdmagnetfeld in horizontale Richtung in µT:", Bhor*1e6)

# Landé-Faktoren
# gF = h/muB / Steigung aus Linfit mit h / muB = h / (e hbar / 2m) = h / (e h / 4m pi)  = 4 m pi / e = 7.14477345 s T
g1F = 7.14477345 * 1e-11 / unp.uarray(params1[0], error1[0])
g2F = 7.14477345 * 1e-11 / unp.uarray(params2[0], error2[0])
print("Landé-Faktor zu Rb87:", g1F)
print("Landé-Faktor zu Rb85:", g2F)

# Kernspins
gs = 2.0023193048
S = 0.5
L = 0
J = 0.5
gJ = ((gs + 1) * J * (J + 1) + (gs - 1) * (S * (S + 1) - L * (L + 1))) / (2 * J * (J + 1))
I1 = 0.5 * (1/ g1F * gJ - 1)
I2 = 0.5 * (1/ g2F * gJ - 1)
print("Kernspin von Rb87:", I1)
print("Kernspin von Rb85:", I2)

# Isotopenanteile
N1 = unp.uarray(156,2) # Anzahl Pixel des dips 1
N2 = unp.uarray(324,2)
R = N1/N2
print("Anteil von Rb87 in %:", 100*R/(R+1))
print("Anteil von Rb85 in %:", 100*(1 - R/(R+1)))

# quadratischer Zeeman-Effekt
muB = 9.274009994 *1e-24 # J / T
Bmax = 0.257*1e-3 # T
Ehy85 = 4.53 * 1e-24 # J
Ehy87 = 2.01 * 1e-24 # J

print("lin85:",g2F*muB*Bmax, "J")
print("lin87:",g1F*muB*Bmax, "J")
print("quad85:", g2F*g2F*muB*muB*Bmax*Bmax*(-5)/Ehy85, "J")
print("quad87:", g1F*g1F*muB*muB*Bmax*Bmax*(-5)/Ehy87, "J")
print("lin/quad85:", g2F*muB*Bmax / (g2F**2 * muB**2 * Bmax**2 * (-5)/Ehy85))
print("lin/quad87:", g1F*muB*Bmax / (g1F*g1F*muB*muB*Bmax*Bmax*(-3)/Ehy87))


# Tabelle
cap = r"Die aufgenommenen Stromstärken der beiden horizontalen Spulen und die korrespondierenden Magnetfelder bei der ersten Resonanzstelle in Abhängigkeit von der eingestellten Frequenz des Hochfrquenzfeldes. Die zur Sweep-Spule gehörigen Messdaten sind mit S und die zur Horizontal-Feld-Spule mit H gekennzeichnet."
hr = [[r"Frequenz", r"Frequenz", r"Stromstärke", r"Stromstärke", r"Stromstärke", r"Stromstärke", r"Magnetfeld", r"Magnetfeld"], [r"$f\, /\mathrm{kHz}$", r"$\Delta f\, /\mathrm{kHz}$", r"$I_{1,\mathrm{S}}\, /\mathrm{mA}$","$\Delta I_{1,\mathrm{S}}\, /\mathrm{mA}$", r"$I_{1,\mathrm{H}}\, /\mathrm{mA}$", r"$\Delta I_{1,\mathrm{H}}\, /\mathrm{mA}$", r"$B_1 \, / \mathrm{µT}$", r"$\Delta B_1 \, / \mathrm{µT}$"]]
fmt = ["%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"]
with open("tab/tab1.tex", "w") as file:
    file.write(matrix2latex([noms(f)*1e-3, stds(f)*1e-3, noms(Isweep1)*1e3, stds(Isweep1)*1e3, noms(Istat1)*1e3, stds(Istat1)*1e3, noms(B1)*1e6, stds(B1)*1e6], headerRow=hr, alignment="S", caption=cap, label=r"tab:tab1", transpose=False, formatColumn=fmt))

cap = r"Die aufgenommenen Stromstärken der beiden horizontalen Spulen und die korrespondierenden Magnetfelder bei der zweiten Resonanzstelle in Abhängigkeit von der eingestellten Frequenz des Hochfrquenzfeldes. Die zur Sweep-Spule gehörigen Messdaten sind mit S und die zur Horizontal-Feld-Spule mit H gekennzeichnet."
hr = [[r"Frequenz", r"Frequenz", r"Stromstärke", r"Stromstärke", r"Stromstärke", r"Stromstärke", r"Magnetfeld", r"Magnetfeld"], [r"$f\, /\mathrm{kHz}$", r"$\Delta f\, /\mathrm{kHz}$", r"$I_{1,\mathrm{S}}\, /\mathrm{mA}$","$\Delta I_{1,\mathrm{S}}\, /\mathrm{mA}$", r"$I_{1,\mathrm{H}}\, /\mathrm{mA}$", r"$\Delta I_{1,\mathrm{H}}\, /\mathrm{mA}$", r"$B_1 \, / \mathrm{µT}$", r"$\Delta B_1 \, / \mathrm{µT}$"]]
fmt = ["%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f", "%.0f"]
with open("tab/tab2.tex", "w") as file:
    file.write(matrix2latex([noms(f)*1e-3, stds(f)*1e-3, noms(Isweep2)*1e3, stds(Isweep2)*1e3, noms(Istat2)*1e3, stds(Istat2)*1e3, noms(B2)*1e6, stds(B2)*1e6], headerRow=hr, alignment="S", caption=cap, label=r"tab:tab2", transpose=False, formatColumn=fmt))
