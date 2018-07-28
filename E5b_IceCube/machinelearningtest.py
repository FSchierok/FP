# https://www.youtube.com/watch?v=QVekgSxOcQQ
# https://www.youtube.com/watch?v=Rh521YZHlY0
from sklearn import neighbors
from sklearn import svm
from sklearn import neural_network
from matplotlib import pyplot
from matplotlib import colors
from collections import Counter
import numpy as np

X = np.array([[1, 2], [3, 5], [2, 4], [4, 5], [3, 3], [5, 1], [4, 1]])
y = np.array([ 0,      0,      0,      1,      1,      0,      1])

for i in range(8):
    X = np.append(X, X + 0.2 * np.random.randn(len(X), 2), axis = 0)
    y = np.append(y, y)

# Plot
u, v = np.meshgrid(np.linspace(0.0, 6.0, 200),
                   np.linspace(0.0, 6.0, 200))
uv = np.c_[u.ravel(), v.ravel()]
colors1 = colors.ListedColormap(["#0000FF", "#FF0000"])
colors2 = colors.ListedColormap(["#9090FF", "#FF6060"])

# nn = neighbors.NearestNeighbors(1).fit(X)
# distances, indices = nn.kneighbors(uv)
# z = y[indices.reshape(u.shape)]

# nn = neighbors.NearestNeighbors(3).fit(X)
# distances, indices = nn.kneighbors(uv)
# z = np.zeros(len(indices))
# for i in range(len(indices)):
#     c = Counter(y[indices[i, :]])
#     z[i] = c.most_common(1)[0][0]
# z = z.reshape(u.shape)

# svc = svm.LinearSVC()
# svc.fit(X,y)
# z = svc.predict(uv)
# z = z.reshape(u.shape)

neur = neural_network.MLPClassifier([30, 20])
z = neur.fit(X + np.random.randn(len(X), 2), y)
z = neur.predict(uv)
z = z.reshape(u.shape)
print([coef.shape for coef in neur.coefs_])
print(neur.coefs_)
print([interc.shape for interc in neur.intercepts_])
print(neur.intercepts_)

pyplot.contourf(u, v, z, cmap = colors2)
# pyplot.scatter(X[:, 0], X[:, 1], 50, y, cmap = colors1)
pyplot.show()
