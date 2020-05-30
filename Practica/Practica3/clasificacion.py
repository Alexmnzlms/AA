def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as metrics

###############################################################################
def obtenerSimetria(x):
    symmetry = 0
    X_matriz = []
    fila = []
    fila.append(x[0])
    for i in range(1,np.size(x)):
        if i%8 == 0 or i == np.size(x) - 1:
            X_matriz.append(fila)
            fila = []
        fila.append(x[i])

    X_matriz = np.array(X_matriz)
    X_superior = []
    X_inferior = []
    for i in range(8):
        fila = []
        for j in range(8):
            if i < j:
                fila.append(X_matriz[i][j])
            else:
                fila.append(0)
        X_superior.append(fila)

    for i in range(8):
        fila = []
        for j in range(8):
            if i > j:
                fila.append(X_matriz[i][j])
            else:
                fila.append(0)
        X_inferior.append(fila)

    X_superior = np.array(X_superior)
    X_inferior = np.array(X_inferior)
    X_inferior = X_inferior.T
    X_resta = np.subtract(X_superior,X_inferior)

    for i in range(8):
        for j in range(8):
            symmetry = symmetry + abs(X_resta[i][j])

    symmetry = symmetry / 448

    return (symmetry)

def obtenerIntensidad(x):
    intensity = 0
    for i in x:
        intensity = intensity + i

    return intensity/1024

def transformacionDataSet(X):
    X_transform = []
    for x in X:
        X_transform.append([1,obtenerIntensidad(x),obtenerSimetria(x)])

    return X_transform

def visualizaDatos(n1, n2, X, y, w):
    #X = transformacionDataSet(X)
    labels = ['0','1','2','3','4','5','6','7','8','9']

    Xsplit =[]
    size = np.int(np.size(X) / 3 )
    print('Tamaño:',size,np.shape(X))
    for i in range(size):
        if y[i] == n1 or y[i] == n2:
            Xsplit.append(X[i])

    X = Xsplit
    X =  list(zip(*X))

    Y = list()
    for i in X[1]:
        Y.append( (-w[0] - w[1]*i) / w[2] )

    # print('Datos:', np.shape(X), X[1], X[2])
    # print('Objetivo:', np.shape(y), y)

    cols = ['red','blue','green','yellow','black','orange','grey','purple','pink','olive']
    colist=[]
    for i in y:
        if i == n1 or i == n2:
            colist.append(cols[i])

    plt.figure(1, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[1], X[2], c=colist)
    plt.plot(X[1], Y, c = 'black')
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

def parameters_cv(estimator,parameters, scores, X_train, y_train, X_test, y_test):
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()


        clf = ms.GridSearchCV(
            estimator, parameters, scoring='%s' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(metrics.classification_report(y_true, y_pred))
        print()
        return clf.best_params_
###############################################################################

X_train = np.array(pd.read_csv('datos/optdigits.tra'))
y_train = X_train[:,64]
X_train= np.delete(X_train,64,axis=1)
X_train = transformacionDataSet(X_train)
X_test = np.array(pd.read_csv('datos/optdigits.tes'))
y_test = X_test[:,64]
X_test = np.delete(X_test,64,axis=1)
X_test = transformacionDataSet(X_test)

# Set the parameters by cross-validation
SGD_parameters = [{'loss': ['squared_loss'], 'alpha' : [0.0001, 0.001, 0.01, 1], 'eta0': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}]
Perceptron_parameters = [{'loss': ['perceptron'], 'alpha' : [0.0001, 0.001, 0.01, 1], 'eta0': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}]
RL_parameters = [{'C': [0.01, 0.1],'solver': ['newton-cg','lbfgs'], 'max_iter': [100, 1000], 'multi_class': ['ovr', 'multinomial']}]
scores = ['accuracy']

np.random.seed(0)
parameters_best_sgd = parameters_cv(lm.SGDClassifier(), SGD_parameters, scores, X_train, y_train, X_test, y_test)
parameters_best_perceptron= parameters_cv(lm.SGDClassifier(), Perceptron_parameters, scores, X_train, y_train, X_test, y_test)
parameters_best_lr = parameters_cv(lm.LogisticRegression(), RL_parameters, scores, X_train, y_train, X_test, y_test)

sgd = lm.SGDClassifier(loss = parameters_best_sgd['loss'], alpha=parameters_best_sgd['alpha'], eta0=parameters_best_sgd['eta0'], max_iter=parameters_best_sgd['max_iter']).fit(X_train, y_train)
print('Mejor configuración para el SGD:',parameters_best_sgd)
scores_cv = ms.cross_val_score(sgd, X_train, y_train, scoring ='accuracy', cv=389)
print("E_cv con SGDClassifier, squared_loss: %0.2f" % ( 1 - scores_cv.mean()))

perceptron = lm.SGDClassifier(loss = parameters_best_perceptron['loss'], alpha=parameters_best_perceptron['alpha'], eta0=parameters_best_perceptron['eta0'], max_iter=parameters_best_perceptron['max_iter']).fit(X_train, y_train)
print('Mejor configuración para el perceptron:',parameters_best_perceptron)
scores_cv = ms.cross_val_score(perceptron, X_train, y_train, scoring ='accuracy', cv=389)
print("E_cv con SGDClassifier, perceptron: %0.2f" % ( 1 - scores_cv.mean()))

lr = lm.LogisticRegression(C = parameters_best_lr['C'], solver=parameters_best_lr['solver'], max_iter=parameters_best_lr['max_iter'],multi_class=parameters_best_lr['multi_class']).fit(X_train, y_train)
print('Mejor configuración para la LR:',parameters_best_lr)
scores_cv = ms.cross_val_score(lr, X_train, y_train, scoring ='accuracy', cv=389)
print("E_cv con logisticRegresion: %0.2f" % ( 1 - scores_cv.mean()))


# sgd = lm.SGDClassifier(loss ='squared_loss', alpha=0.001, eta0=0.001, max_iter=100).fit(X_train, y_train)
# print('Mejor configuración para el SGD:')
# scores_cv = ms.cross_val_score(sgd, X_train, y_train, scoring ='accuracy', cv=389)
# print("E_cv con SGDClassifier, squared_loss: %0.2f" % ( 1 - scores_cv.mean()))
#
# perceptron = lm.SGDClassifier(loss ='perceptron', alpha=0.0001, eta0=0.01, max_iter=10000).fit(X_train, y_train)
# print('Mejor configuración para el perceptron:')
# scores_cv = ms.cross_val_score(perceptron, X_train, y_train, scoring ='accuracy', cv=389)
# print("E_cv con SGDClassifier, perceptron: %0.2f" % ( 1 - scores_cv.mean()))

# lr = lm.LogisticRegression(C = 0.1, solver='lbfgs', max_iter=100, multi_class='ovr').fit(X_train, y_train)
# print('Mejor configuración para la LR:')
# scores_cv = ms.cross_val_score(lr, X_train, y_train, scoring ='accuracy', cv=389)
# print("E_cv con logisticRegresion: %0.2f" % ( 1 - scores_cv.mean()))

y_pred = lr.predict(X_test)
print('E_test para regresion logistica:', 1 - metrics.accuracy_score(y_pred, y_test))
