def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as metrics

#################################################################################
# Funcion de limpieza de datos
def limpiar_datos(X):
    for i in range(5):
        X = np.delete(X,0,axis=1)
        #Eliminamos las 5 primeras filas ya que son valores no predictivos

    eliminar = []
    #Despues comprobamos las columnas que tengan más de 100 valores nan y las eliminamos
    for j in range(122):
        contador = 0
        for i in range(1994):
            if np.isnan(X[i,j]):
                contador = contador + 1;
            if contador > 100:
                eliminar.append(j)
                break
    X = np.delete(X,eliminar,axis=1);

    # Despues sustituimos los valores nan que queden por la media de los demás valores
    for i in range(1994):
        for j in range(100):
            if np.isnan(X[i,j]):
                X[i,j] = np.nanmean( np.array(X[:,j], dtype=np.float64) )

    # Finalmente devolvemos X
    return X

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
#################################################################################
# Leemos el conjunto de datos
X = np.array(pd.read_csv('datos/communities.data', na_values='?', header=None))
# Extraemos la clase que en este caso es la ultima columna
y = X[:,127]
# Eliminamos la columna de la clase del conjuto de datos
X = np.delete(X,127,axis=1)
# Aplicamos el preprocesado a los datos
X = limpiar_datos(X)

np.random.seed(78925)

# Obtenemos los conjuntos de training y test en una proporcion del 80-20
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)

# Establecemos la metrica
scores = ['neg_mean_squared_error']

# Definimos los distintos hiperparametros que queremos optimizar
SGD_parameters = [{'alpha' : [0.0001, 0.001, 0.01, 1], 'eta0': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}]

# Obtenemos los valores optimos para los hiperparametros
parameters_best_sgd = parameters_cv(lm.SGDRegressor(), SGD_parameters, scores, X_train, y_train)

# Una vez obtenidos los hiperparámetros ajustamos los modelos con los mismos
sgd = lm.SGDRegressor(alpha=parameters_best_sgd['alpha'], eta0=parameters_best_sgd['eta0'], max_iter=parameters_best_sgd['max_iter']).fit(X_train, y_train)
print('Mejor configuración para el SGD:',parameters_best_sgd)
# Obtenemos E_cv
scores_cv = ms.cross_val_score(sgd, X_train, y_train, scoring ='neg_mean_squared_error', cv=1595)
print("E_cv con SGDRegressor:", (abs(scores_cv.mean())))

# Obtenemos E_test para el mejor ajuste obtenido
y_pred = sgd.predict(X_test)
print('E_test para SGDRegressor:', metrics.mean_squared_error(y_pred, y_test))
