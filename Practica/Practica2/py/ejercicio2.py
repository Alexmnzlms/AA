"""
TRABAJO 2
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

np.random.seed(1)

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def ajusta_PLA(datos, label, max_iter, vini):
    #CODIGO DEL ESTUDIANTE
    w = np.copy(vini)
    mejora = True
    iter = 0
    while(mejora and iter < max_iter):
        mejora = False
        iter = iter + 1
        for i in range(len(datos)):
            if(signo(w.T.dot(datos[i])) != label[i]):
                w = w + label[i]*datos[i]
                mejora = True

    return w, iter

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.

    return a, b


#CODIGO DEL ESTUDIANTE

x = simula_unif(100, 2, [-50,50])
a,b = simula_recta([-50,50])
f_values = list()
for i in range(100):
	f_values.append(f(x[i][0],x[i][1],a,b))

f_values = np.array(f_values)

D = list()
for i in x:
	D.append([1.0,i[0],i[1]])
D = np.array(D)


# Random initializations
iterations = []
iterations.append(np.array([0.0,0.0,0.0]))
for i in range(0,10):
    #CODIGO DEL ESTUDIANTE
    iterations.append(np.reshape(simula_unif(3, 1, [0,1]), (1,-1))[0])

iter = []
for v in iterations:
    print('W de partida: ', v)
    w, it = ajusta_PLA(D, f_values, np.Infinity, v)
    print('W obtenida: ', w)
    print('Iteraciones: ', it)
    iter.append(it)

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iter))))

input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b

#CODIGO DEL ESTUDIANTE

print('Añadimos ruido')

positive = list()
negative = list()

for i in f_values:
	if(i == -1):
		negative.append(i)
	else:
		positive.append(i)

index = np.random.choice(len(positive),np.int(0.1*len(positive)), replace=False)
for i in index:
	positive[i] = -1*positive[i]

index = np.random.choice(len(negative), np.int(0.1*len(negative)), replace=False)
for i in index:
	negative[i] = -1*negative[i]

f_values = list()
for i in positive:
	f_values.append(i)

for i in negative:
	f_values.append(i)

f_values = np.array(f_values)

iter = []
for v in iterations:
    print('W de partida: ', v)
    w, it = ajusta_PLA(D, f_values, 10000, v)
    print('W obtenida: ', w)
    print('Iteraciones: ', it)
    iter.append(it)



print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iter))))

input("\n--- Pulsar tecla para continuar ---\n")


'''

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

def sgdRL(?):
    #CODIGO DEL ESTUDIANTE

    return w



#CODIGO DEL ESTUDIANTE

input("\n--- Pulsar tecla para continuar ---\n")



# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).


#CODIGO DEL ESTUDIANTE


input("\n--- Pulsar tecla para continuar ---\n")
'''
