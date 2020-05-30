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
    # Calculamos la simetria de una matriz
    # Convertimos el vector X en matriz X
    symmetry = 0
    X_matriz = []
    fila = []
    fila.append(x[0])
    for i in range(1,np.size(x)):
        if i%8 == 0 or i == np.size(x) - 1:
            X_matriz.append(fila)
            fila = []
        fila.append(x[i])

    #Una vez obtenida la matriz, la separamos en matriz superior e inferior
    # ignorando la diagonal principal
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
    # Guardamos la matriz inferior como la traspuesta
    X_inferior = X_inferior.T

    # Restamos ambas matrices
    X_resta = np.subtract(X_superior,X_inferior)

    # Sumamos los valores absolutos de la matriz resta
    for i in range(8):
        for j in range(8):
            symmetry = symmetry + abs(X_resta[i][j])

    # Normalizamos el valor simetria
    symmetry = symmetry / 448

    return (symmetry)

def obtenerIntensidad(x):
    # Sumamos los valores de intensidad de los pixeles
    intensity = 0
    for i in x:
        intensity = intensity + i

    # Devolvemos el valor normalizado
    return intensity/1024

def transformacionDataSet(X):
    # Reducimos la dimensionalidad del conjunto
    X_transform = []
    for x in X:
        X_transform.append([obtenerIntensidad(x),obtenerSimetria(x)])

    return X_transform

def parameters_cv(estimator,parameters, scores, X_train, y_train):
    for score in scores:
        # Para cada estimador hacemos lo siguiente:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # Definimos un GridSearchCV con los parametros que queremos optimizar
        clf = ms.GridSearchCV(
            estimator, parameters, scoring='%s' % score
        )
        clf.fit(X_train, y_train)

        # Finalmente mostramos los resultados de los distintos ajustes y el mejor
        # ajuste encontrado
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

        return clf.best_params_
###############################################################################

# Leemos el conjunto de datos de training
X_train = np.array(pd.read_csv('datos/optdigits.tra'))
# Extraemos la clase que en este caso es la ultima columna
y_train = X_train[:,64]
# Eliminamos la columna de la clase del conjuto de datos
X_train= np.delete(X_train,64,axis=1)

# X_train = transformacionDataSet(X_train)

# Leemos el conjunto de datos de test
X_test = np.array(pd.read_csv('datos/optdigits.tes'))
# Extraemos la clase que en este caso es la ultima columna
y_test = X_test[:,64]
# Eliminamos la columna de la clase del conjuto de datos
X_test = np.delete(X_test,64,axis=1)

# X_test = transformacionDataSet(X_test)

# Definimos los distintos hiperparametros que queremos optimizar
SGD_parameters = [{'loss': ['squared_loss'], 'alpha' : [0.0001, 0.001, 0.01, 1], 'eta0': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}]
Perceptron_parameters = [{'loss': ['perceptron'], 'alpha' : [0.0001, 0.001, 0.01, 1], 'eta0': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}]
RL_parameters = [{'C': [0.01, 0.1],'solver': ['newton-cg','lbfgs'], 'max_iter': [100, 1000], 'multi_class': ['ovr', 'multinomial']}]

# Establecemos la metrica
scores = ['accuracy']

# Ajustamos la semilla
np.random.seed(0)

# Obtenemos los valores optimos para los hiperparametros
parameters_best_sgd = parameters_cv(lm.SGDClassifier(), SGD_parameters, scores, X_train, y_train)
parameters_best_perceptron= parameters_cv(lm.SGDClassifier(), Perceptron_parameters, scores, X_train, y_train)
parameters_best_lr = parameters_cv(lm.LogisticRegression(), RL_parameters, scores, X_train, y_train)

# Una vez obtenidos los hiperparámetros ajustamos los modelos con los mismos

sgd = lm.SGDClassifier(loss = parameters_best_sgd['loss'], alpha=parameters_best_sgd['alpha'], eta0=parameters_best_sgd['eta0'], max_iter=parameters_best_sgd['max_iter']).fit(X_train, y_train)
print('Mejor configuración para el SGD:',parameters_best_sgd)
# Obtenemos E_cv
scores_cv = ms.cross_val_score(sgd, X_train, y_train, scoring ='accuracy', cv=389)
print("E_cv con SGDClassifier, squared_loss:", ( 1 - scores_cv.mean()))

perceptron = lm.SGDClassifier(loss = parameters_best_perceptron['loss'], alpha=parameters_best_perceptron['alpha'], eta0=parameters_best_perceptron['eta0'], max_iter=parameters_best_perceptron['max_iter']).fit(X_train, y_train)
print('Mejor configuración para el perceptron:',parameters_best_perceptron)
# Obtenemos E_cv
scores_cv = ms.cross_val_score(perceptron, X_train, y_train, scoring ='accuracy', cv=389)
print("E_cv con SGDClassifier, perceptron:", ( 1 - scores_cv.mean()))

lr = lm.LogisticRegression(C = parameters_best_lr['C'], solver=parameters_best_lr['solver'], max_iter=parameters_best_lr['max_iter'],multi_class=parameters_best_lr['multi_class']).fit(X_train, y_train)
print('Mejor configuración para la LR:',parameters_best_lr)
# Obtenemos E_cv
scores_cv = ms.cross_val_score(lr, X_train, y_train, scoring ='accuracy', cv=389)
print("E_cv con logisticRegresion:", ( 1 - scores_cv.mean()))

# Obtenemos E_test para el mejor ajuste obtenido
y_pred = lr.predict(X_test)
print('E_test para regresion logistica:', 1 - metrics.accuracy_score(y_pred, y_test))
