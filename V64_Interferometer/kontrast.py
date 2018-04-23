import numpy as np
import matplotlib.pyplot as plt

phi, Umin, Umax = np.genfromtxt("kontrast.txt", unpack=True)
k = (Umax -
     Umin) / (Umax + Umin)
print(max(k))
print(phi[np.argmax(k)])
plt.plot(phi, k, ".b", label="Messwerte")
plt.grid()
plt.legend(loc="best")
plt.show()
