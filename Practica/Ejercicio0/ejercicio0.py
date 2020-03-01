import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from sklearn import datasets
import matplotlib.patches as mpatches
import math


'''
Parte 1
'''

iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target
labels = iris.target_names
feature = iris.feature_names
cols = ['red','blue','green']
colist=[]
for i in y:
    if i == 0:
        colist.append('red')
    elif i == 1:
        colist.append('blue')
    else:
        colist.append('green')

plt.figure(1, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=colist)
plt.xlabel(feature[2])
plt.ylabel(feature[3])
plt.title('Gráfico 1')

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xticks(())
plt.yticks(())

legend_elements = [mpatches.Patch(color=cols[0], label=labels[0]),
                   mpatches.Patch(color=cols[1], label=labels[1]),
                   mpatches.Patch(color=cols[2], label=labels[2])]
plt.legend(handles=legend_elements)
plt.show()


'''
Parte 2
'''
c = list(zip(X, y))
a, b = zip(*c)
print('Elementos del cojunto (tamaño',len(c),')')
print(a)
print()
print(b)

porcentaje = 0.8
s = len(c)
m = math.floor(len(c)*porcentaje)
idxtraining = np.random.choice(s, size=m, replace=False)
idxtraining.sort()
idxtest = list()

for i in range(s):
    if i not in idxtraining:
        idxtest.append(i)



training = list()
for i in idxtraining:
    training.append(c[i])

test = list()
for i in idxtest:
    test.append(c[i])

print()
a, b = zip(*training)
print('Elementos del cojunto training (tamaño',len(training),', porcentaje',math.ceil(porcentaje*100),'%)')
print(a)
print()
print(b)

print()
a, b = zip(*test)
print('Elementos del cojunto test (tamaño',len(test),', porcentaje',math.ceil((1-porcentaje)*100),'%)')
print(a)
print()
print(b)

input("Pulsa intro para continuar")


'''
Parte 3
'''
