"""
TRABAJO 2
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

np.random.seed(1996)

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

def ajusta_PLA(datos, label, max_iter, vini):
	# datos: conjunto de entrenamiento D
	# label: conjunto de etiquetas asociado a D
	# max_iter: numero máximo de iteraciones
	# vini: valor inicial para w
    # Establecemos vini como valor inicial de w
    w = np.copy(vini)
    mejora = True
    iter = 0
	# Si ha habido mejora y las iteraciones no superan un maximo
    while(mejora and iter < max_iter):
        mejora = False
        iter = iter + 1
		# Para cada xi del conjunto D
        for i in range(len(datos)):
            if(signo(w.T.dot(datos[i])) != label[i]):
				# w_new = w_old + xi*yi
                w = w + label[i]*datos[i]
                mejora = True

	# Devuelve el valor de w y las iteraciones utilizadas
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
for i in range(0,10):
    iterations.append(np.reshape(simula_unif(3, 1, [0,1]), (1,-1))[0])

iterations.append(np.array([0.0,0.0,0.0]))

iter = []
for v in iterations:
    print('W de partida: ', v)
    w, it = ajusta_PLA(D, f_values, np.Infinity, v)
    print('W obtenida: ', w)
    print('Iteraciones: ', it)
    iter.append(it)

print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iter))))

Y = list()
for i in x[: ,0]:
    Y.append( (-w[0] - w[1]*i) / w[2] )

colist=[]
for i in f_values:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

y = x[:,0]*a + b
plt.plot(x[:, 0], y, c = 'grey')
plt.plot(x[:, 0], Y, c = 'black')
plt.scatter(x[:, 0], x[:, 1], c=colist)
plt.title('Ejercicio 2.1. Conjunto de datos del apartado anterior')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-50,50])
legend_elements = [mlines.Line2D([], [], color='grey',markersize=15, label='Recta de clasificación'),
				   mlines.Line2D([], [], color='black',markersize=15, label='Recta obtenida con ajusta_PLA(0,0,0)'),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='-1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='1', markersize=10)]
plt.legend(handles=legend_elements)

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

# Ahora con los datos del ejercicio 1.2.b
# Añadimos ruido
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

colist=[]
for i in f_values:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')


v = np.array([0.0,0.0,0.0])
print('W de partida: ', v)
w, it = ajusta_PLA(D, f_values, 1000, v)
print('W obtenida: ', w)
print('Iteraciones: ', it)

Y = list()
for i in x[: ,0]:
    Y.append( (-w[0] - w[1]*i) / w[2] )

y = x[:,0]*a + b
plt.plot(x[:, 0], y, c = 'grey')
plt.plot(x[:, 0], Y, c = 'black')
plt.scatter(x[:, 0], x[:, 1], c=colist)
plt.title('Ejercicio 2.2. Añadido ruido al conjunto de datos anterior')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-50,50])
legend_elements = [mlines.Line2D([], [], color='grey',markersize=15, label='Recta de clasificación'),
				   mlines.Line2D([], [], color='black',markersize=15, label='Recta obtenida con ajusta_PLA (1000)'),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='-1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='1', markersize=10)]
plt.legend(handles=legend_elements)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

v = np.array([0.0,0.0,0.0])
print('W de partida: ', v)
w, it = ajusta_PLA(D, f_values, 10000, v)
print('W obtenida: ', w)
print('Iteraciones: ', it)

Y = list()
for i in x[: ,0]:
    Y.append( (-w[0] - w[1]*i) / w[2] )

y = x[:,0]*a + b
plt.plot(x[:, 0], y, c = 'grey')
plt.plot(x[:, 0], Y, c = 'black')
plt.scatter(x[:, 0], x[:, 1], c=colist)
plt.title('Ejercicio 2.2. Añadido ruido al conjunto de datos anterior')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([-50,50])
legend_elements = [mlines.Line2D([], [], color='grey',markersize=15, label='Recta de clasificación'),
				   mlines.Line2D([], [], color='black',markersize=15, label='Recta obtenida con ajusta_PLA (10000)'),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='-1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='1', markersize=10)]
plt.legend(handles=legend_elements)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
# Aplicamos SGDRL

def E(x,y,w):

	return (-(y*x) / (1 + np.exp(y*np.dot(w.T,x))))

def sgdRL(x,y,n):
	# x: conjunto de datos
	# y: conjunto de etiquetas
	# n: tasa de aprendizaje
	# Inicializamos w a {0,0,0}
	w = np.zeros(x[0].size)
	# Guardamos una copia de w en w_ant
	w_ant = np.copy(w)
	# Obtenemos los indices en orden aleatorio
	batch = np.random.choice(np.size(x,0), np.size(x,0), replace=False)
	# Obtenemos el valor de w(0)
	# Para cada indice
	for j in batch:
		w = w - n * E(x[j],y[j],w)
	#Mientras ||w_ant - w|| >= 0.01, seguimos calculando w(t)
	while np.linalg.norm(w_ant - w) >= 0.01:
		# Guardamos una copia de w en w_ant
		w_ant = np.copy(w)
		# Obtenemos los indices en orden aleatorio
		batch = np.random.choice(np.size(x,0), np.size(x,0), replace=False)
		# Para cada indice
		for j in batch:
			w = w - n * E(x[j],y[j],w)

	return w

def genera_recta(x):

	points = np.random.choice(np.size(x,0), 2, replace=False)
	x1 = x[points[0]][0]
	x2 = x[points[1]][0]
	y1 = x[points[0]][1]
	y2 = x[points[1]][1]
	# y = a*x + b
	a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
	b = y1 - a*x1       # Calculo del termino independiente.

	return a, b

x = simula_unif(100, 2, [0,2])
a,b = genera_recta(x)
f_values = list()
for i in range(100):
	f_values.append(f(x[i][0],x[i][1],a,b))

f_values = np.array(f_values)

D = list()
for i in x:
	D.append([1.0,i[0],i[1]])
D = np.array(D)


w = sgdRL(D,f_values,0.01)

Y = list()
for i in x[: ,0]:
    Y.append( (-w[0] - w[1]*i) / w[2] )

colist=[]
for i in f_values:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

y = x[:,0]*a + b
plt.plot(x[:, 0], y, c = 'grey')
plt.plot(x[:, 0], Y, c = 'black')
plt.scatter(x[:, 0], x[:, 1], c=colist)
plt.title('Ejercicio 2.3. Conjunto de datos D para regresión logistica')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
plt.ylim([0,2])
legend_elements = [mlines.Line2D([], [], color='grey',markersize=15, label='Recta que clasifica los datos'),
				   mlines.Line2D([], [], color='black',markersize=15, label='Recta obtenida con sgdRL'),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='-1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='1', markersize=10)]
plt.legend(handles=legend_elements)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Obtenemos el error

def ErrLR(x,y,w):
	return np.log(1 + np.exp(-1*y*np.dot(w.T,x)))

num_data = 1000
x = simula_unif(num_data, 2, [0,2])
f_values = list()
for i in range(num_data):
	f_values.append(f(x[i][0],x[i][1],a,b))

f_values = np.array(f_values)

D = list()
for i in x:
	D.append([1.0,i[0],i[1]])
D = np.array(D)

err = []
for i in range(num_data):
	err.append(ErrLR(D[i],f_values[i],w))

err = np.mean(err)
print('E(out) obtenido:',err)



input("\n--- Pulsar tecla para continuar ---\n")
