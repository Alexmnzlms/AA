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
def limpiar_datos(X):
    for i in range(5):
        X = np.delete(X,0,axis=1)

    return X

#################################################################################
X = np.array(pd.read_csv('datos/communities.data', na_values='?'))
print(np.shape(X),'\n',X)
X = limpiar_datos(X)
print(np.shape(X),'\n',X)
