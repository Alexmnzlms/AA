import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patches as mpatches

###############################################################################
def obtenerSimetria(x):
    symmetry = 0
    up = x[:32]
    down = x[33:]
    for i in range(31):
        symmetry = symmetry + abs(up[i] - down[i])

    return 1- (symmetry/512)

def obtenerIntensidad(x):
    intensity = 0
    for i in x:
        intensity = intensity + i

    return intensity/1024

def transformacionDataSet(X):
    X_transform = []
    for x in X:
        X_transform.append([obtenerIntensidad(x),obtenerSimetria(x)])

    return X_transform
        

###############################################################################

X_train = np.array(pd.read_csv('optdigits.tra'))
# print('Datos leidos:\n' , X_train)

y_train = X_train[:,64]
X_train= np.delete(X_train,64,axis=1)
X_train = transformacionDataSet(X_train)
print('Conjunto de training:\n' , X_train)
print('Conjunto de etiquetas de training:\n', y_train)


X_test = np.array(pd.read_csv('optdigits.tes'))
# print('Datos leidos:\n' , X_test)

y_test = X_test[:,64]
X_test = np.delete(X_test,64,axis=1)
X_test = transformacionDataSet(X_test)
print('Conjunto de test:\n' , X_test)
print('Conjunto de etiquetas de test:\n', y_test)

clf = lm.SGDClassifier(max_iter=1000, tol=1e-3).fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
y_predict = clf.predict(X_test)

# for i in range(np.size(y_test)):
#    print(y_test[i],y_predict[i])

print(np.shape(clf.coef_))
# print(clf.coef_)

clf = lm.Perceptron().fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
y_predict = clf.predict(X_test)

# for i in range(np.size(y_test)):
#    print(y_test[i],y_predict[i])

print(np.shape(clf.coef_))
# print(clf.coef_)

###############################################################################
def visualizaDatos(n1, n2, X, y):
    X = transformacionDataSet(X)
    labels = ['0','1','2','3','4','5','6','7','8','9']

    Xsplit =[]
    size = np.int(np.size(X)/2 - 1)
    for i in range(size):
        if y[i] == n1 or y[i] == n2:
            Xsplit.append(X[i])

    X = Xsplit
    X =  list(zip(*X))

    print('Datos:', np.shape(X), X[0], X[1])
    print('Objetivo:', np.shape(y), y)

    cols = ['red','blue','green','yellow','black','orange','grey','purple','pink','olive']
    colist=[]
    for i in y:
        if i == n1 or i == n2:
            colist.append(cols[i])

    plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[0], X[1], c=colist)
    plt.xlabel('Intensidad')
    plt.ylabel('Simetria')
    plt.title('Intensidad vs Simetria')
    plt.xticks(())
    plt.yticks(())

    legend_elements = []
    legend_elements.append(mpatches.Patch(color=cols[n1], label=labels[n1]))
    legend_elements.append(mpatches.Patch(color=cols[n2], label=labels[n2]))

    plt.legend(handles=legend_elements)
    plt.show()


# visualizaDatos(2,5,X_train,y_train)
# svisualizaDatos(2,5,X_test,y_test)


