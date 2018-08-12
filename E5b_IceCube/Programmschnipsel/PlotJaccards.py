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


# Plot Jaccard-Index NBayes
Jaccard1x, Jaccard1y = np.genfromtxt("Jaccardtable_NBayes.txt", unpack=True)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.plot(Jaccard1x, Jaccard1y, "-", label="Jaccard-Index Naive Bayes")
ax1.set_xlabel("\# Attribute")
ax1.set_ylabel(r"$J$")
#ax1.set_ylim(-12.15, -5.7)
#ax1.set_xlim(-5, 131)
ax1.legend()
ax1.grid()

fig.savefig("plots/Jaccardplot_NBayes.pdf")

# # Plot Jaccard-Index BinTree
# Jaccard2x, Jaccard2y = np.genfromtxt("Jaccardtable_BinTree.txt", unpack=True)
# fig = plt.figure(2)
# ax1 = fig.add_subplot(111)
# ax1.plot(Jaccard2x, Jaccard2y, "-", label="Jaccard-Index Bintree")
# ax1.set_xlabel("\# Attribute")
# ax1.set_ylabel(r"$J$")
# #ax1.set_ylim(-12.15, -5.7)
# #ax1.set_xlim(-5, 131)
# ax1.legend()
# ax1.grid()
#
# fig.savefig("plots/Jaccardplot_BinTree.pdf")


# Plot Jaccard-Index k Neares Neighbors
Jaccard3x, Jaccard3y = np.genfromtxt("Jaccardtable_Neigh2.txt", unpack=True)
fig = plt.figure(3)
ax1 = fig.add_subplot(111)
ax1.plot(Jaccard3x, Jaccard3y, "-", label="Jaccard-Index k Nearest Neighbors")
ax1.set_xlabel("\# Attribute")
ax1.set_ylabel(r"$J$")
#ax1.set_ylim(-12.15, -5.7)
#ax1.set_xlim(-5, 131)
ax1.legend()
ax1.grid()
fig.savefig("plots/Jaccardplot_Neigh2.pdf")
