import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


'''
Parte 1
'''
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(1, figsize=(8, 6))
plt.clf()

colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
cmap_name = 'rgb'
cm = col.LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')


red_patch = mpatches.Patch(color='red', label='Elemento 1')
blue_patch = mpatches.Patch(color='blue', label='Elemento 2')
green_patch = mpatches.Patch(color='green', label='Elemento 3')
plt.legend(handles=[red_patch, blue_patch, green_patch])

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()
