import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.metrics as metrics 

###############################################################################
def obtenerSimetria(x):
    symmetry = 0
    up = x[:32]
    down = x[33:]
    for i in range(31):
        symmetry = symmetry + abs(up[i] - down[i])

    return (symmetry/512)

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

def transformacionDataSetBasica(X):
    X_transform = []
    for x in X:
        X_transform.append(np.concatenate((np.array([1]),x),axis=None))
        
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



X_train = np.array(pd.read_csv('optdigits.tra'))
y_train = X_train[:,64]
X_train= np.delete(X_train,64,axis=1)
X_train = transformacionDataSetBasica(X_train)

X_test = np.array(pd.read_csv('optdigits.tes'))
y_test = X_test[:,64]
X_test = np.delete(X_test,64,axis=1)
X_test = transformacionDataSetBasica(X_test)

# Set the parameters by cross-validation
SGD_parameters = [{'loss': ['squared_loss'], 'alpha' : [0.001, 0.01, 1], 'eta0': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}]
Perceptron_parameters = [{'loss': ['perceptron'], 'alpha' : [0.001, 0.01, 1], 'eta0': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}]
RL_parameters = [{'C': [0.01, 0.1],'solver': ['newton-cg','lbfgs'], 'max_iter': [100, 1000], 'multi_class': ['ovr', 'multinomial']}]
scores = ['f1_macro']

np.random.seed(0)
parameters_best_sgd = parameters_cv(lm.SGDClassifier(), SGD_parameters, scores, X_train, y_train, X_test, y_test)
parameters_best_perceptron= parameters_cv(lm.SGDClassifier(), Perceptron_parameters, scores, X_train, y_train, X_test, y_test)
parameters_best_lr = parameters_cv(lm.LogisticRegression(), RL_parameters, scores, X_train, y_train, X_test, y_test)

lr = lm.LogisticRegression(C = parameters_best_lr['C'], solver=parameters_best_lr['solver'], max_iter=parameters_best_lr['max_iter'],multi_class=parameters_best_lr['multi_class']).fit(X_train, y_train)
print('Mejor configuración para la LR:',parameters_best_lr)
print('E_in con logisticRegresion:', 1 - lr.score(X_train, y_train))
scores = ms.cross_val_score(lr, X_train, y_train, cv=5)
print("Error: %0.2f" % ( 1 - scores.mean()))

sgd = lm.SGDClassifier(loss = parameters_best_sgd['loss'], alpha=parameters_best_sgd['alpha'], eta0=parameters_best_sgd['eta0'], max_iter=parameters_best_sgd['max_iter']).fit(X_train, y_train)
print('Mejor configuración para el SGD:',parameters_best_sgd)
print('E_in con SGDClassifier, squared_loss:', 1 - sgd.score(X_train, y_train))
scores = ms.cross_val_score(lr, X_train, y_train, cv=5)
print("Error: %0.2f" % ( 1 - scores.mean()))

perceptron = lm.SGDClassifier(loss = parameters_best_perceptron['loss'], alpha=parameters_best_perceptron['alpha'], eta0=parameters_best_perceptron['eta0'], max_iter=parameters_best_perceptron['max_iter']).fit(X_train, y_train)
print('Mejor configuración para el perceptron:',parameters_best_perceptron)
print('E_in con SGDClassifier, perceptron:', 1 - perceptron.score(X_train, y_train))
scores = ms.cross_val_score(lr, X_train, y_train, cv=5)
print("Error: %0.2f" % ( 1 - scores.mean()))

