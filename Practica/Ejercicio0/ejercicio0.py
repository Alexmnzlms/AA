import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.lines as line
from sklearn import datasets
import matplotlib.patches as mpatches
import math


'''
Parte 1
'''
print('-----------------------------------------------------------------------')
print('Parte 1')
print('-----------------------------------------------------------------------')
iris = datasets.load_iris()
X = iris.data[:, 2:4]
y = iris.target
labels = iris.target_names
feature = iris.feature_names

print('Datos:')
print(list(X))
print('Objetivo:')
print(y)

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
plt.title('Gráfico 1: Total de valores de Iris')

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
print('-----------------------------------------------------------------------')
print('Parte 2')
print('-----------------------------------------------------------------------')
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

a_training, b_training = zip(*training)
X_training_x = list()
X_training_y = list()
for i in a_training:
    X_training_x.append(i[0])
    X_training_y.append(i[1])

colist_training=[]
for i in b_training:
    if i == 0:
        colist_training.append('red')
    elif i == 1:
        colist_training.append('blue')
    else:
        colist_training.append('green')

print()
print('Elementos del cojunto training (tamaño',len(training),', porcentaje',math.ceil(porcentaje*100),'%)')
print(a_training)
print()
print(b_training)

plt.figure(2, figsize=(8, 6))
plt.clf()

plt.scatter(X_training_x, X_training_y, c=colist_training)
plt.xlabel(feature[2])
plt.ylabel(feature[3])
plt.title('Gráfico 2: Valores del conjunto training')

plt.xticks(())
plt.yticks(())

legend_elements = [mpatches.Patch(color=cols[0], label=labels[0]),
                   mpatches.Patch(color=cols[1], label=labels[1]),
                   mpatches.Patch(color=cols[2], label=labels[2])]
plt.legend(handles=legend_elements)

a_test, b_test = zip(*test)
X_test_x = list()
X_test_y = list()
for i in a_test:
    X_test_x.append(i[0])
    X_test_y.append(i[1])

colist_test=[]
for i in b_test:
    if i == 0:
        colist_test.append('red')
    elif i == 1:
        colist_test.append('blue')
    else:
        colist_test.append('green')

print()
print('Elementos del cojunto test (tamaño',len(test),', porcentaje',math.ceil((1-porcentaje)*100),'%)')
print(a_test)
print()
print(b_test)

plt.figure(3, figsize=(8, 6))
plt.clf()

plt.scatter(X_test_x, X_test_y, c=colist_test)
plt.xlabel(feature[2])
plt.ylabel(feature[3])
plt.title('Gráfico 3: Valores del conjunto test')

plt.xticks(())
plt.yticks(())

legend_elements = [mpatches.Patch(color=cols[0], label=labels[0]),
                   mpatches.Patch(color=cols[1], label=labels[1]),
                   mpatches.Patch(color=cols[2], label=labels[2])]
plt.legend(handles=legend_elements)
plt.show()

'''
Parte 3
'''
print('-----------------------------------------------------------------------')
print('Parte 3')
print('-----------------------------------------------------------------------')
min = 0
max = math.pi*2
n = 100
x = np.linspace(min,max,num=n)
print(len(x),'valores equiespaciados entre', min,'y', max)
print(x)
y = list()
z = list()
w = list()
for i in x:
    y.append(math.sin(i))
    z.append(math.cos(i))
    w.append(math.sin(i) + math.cos(i))
plt.figure(4, figsize=(8, 6))
plt.clf()
plt.plot(x, y, c = 'black', linestyle='dashed')
plt.plot(x, z, c = 'blue', linestyle='dashed')
plt.plot(x, w, c = 'red', linestyle='dashed')

legend_elements = [mpatches.Patch(color='black', label='sin(x)'),
                   mpatches.Patch(color='blue', label='cos(x)'),
                   mpatches.Patch(color='red', label='sin(x)+cos(x)')]
plt.legend(handles=legend_elements)

plt.show()
