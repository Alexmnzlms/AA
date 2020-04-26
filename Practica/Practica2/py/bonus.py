#BONUS: Clasificación de Dígitos

"""
BONUS
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math

np.random.seed(1)


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#LINEAR REGRESSION FOR CLASSIFICATION
def signo(x):
	if x >= 0:
		return 1
	return -1

def Err(x,y,w):
	# x: conjunto de datos
	# y: conjunto de etiquetas
	# w: valores de ŵ
	# Contamos los puntos que estan mal clasificados
	cont = 0
	for i in range(len(x)):
		if signo(np.dot(w.T,x[i])) != y[i]:
			cont += 1

	return cont / len(x)


def sgd(x,y,n,iterations):
	w = np.zeros(x[0].size) # Inicializamos w al vector de tantos 0's como
							# caracteristicas tiene x
	c = 0
	epoca = 0
	# Mientras no se supere el numero maximo de iteraciones y no hayamos terminado una epoca
	termianda_epoca = True
	batch = np.array([])
	while c < iterations or not termianda_epoca:
		# Obtenemos la submuestra de X
		if len(batch) == 0:
			termianda_epoca = False
			batch = np.random.choice(np.size(x,0), np.size(x,0), replace=False)

		minibatch = []
		index = []
		for i in range(32):
			if i == len(batch):
				break
			minibatch.append(batch[i])
			index.append(i)

		batch = np.delete(batch,index)

		# Copiamos en w_ant el valor anterior de w
		w_ant = np.copy(w)
		# Para cada wj
		for i in range(np.size(w)):
			sumatoria = 0
			# Calculamos la sumatoria de cada x que pertenece a la submuestra
			for j in minibatch:
				c = c + 1
				sumatoria = sumatoria + x[j][i]*(np.dot(x[j],w.T) - y[j])
			# Actualizamos el valor de wj
			w[i] = w_ant[i] -n * (2.0/np.float(np.size(minibatch))) * sumatoria

		if len(batch) == 0:
			termianda_epoca = True
			epoca +=1
	# Devolvemos w
	return w

w = sgd(x,y,0.01,10000)

print('E(in)', Err(x,y,w))
print('E(test)', Err(x_test,y_test	,w))

Y = list()
for i in x[: ,1]:
    Y.append( (-w[0] - w[1]*i) / w[2] )

fig, ax = plt.subplots()
ax.plot(x[:, 1], Y, c = 'black')
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
ax.set_ylim((-8, 0))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(x[:, 1], Y, c = 'black')
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
ax.set_ylim((-8, 0))
plt.legend()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

#POCKET ALGORITHM

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
    return w

def pocket_PLA(datos, label, w):
	# datos: conjunto de entrenamiento D
	# label: conjunto de etiquetas asociado a D
	# w: valor inicial del vector w
	# Establecemos como Error minimo, el valor de Error(w)
	err_min = Err(datos,label,w)
	w_min = w
	# Hacemos 100 ejecuciones del PLA con 1 iteración
	for i in range(100):
		vini = np.copy(w)
		w = ajusta_PLA(datos, label, 1, vini)
		err_w = Err(datos,label,w)
		# Nos quedamos con el w que produce un error minimo
		if err_w < err_min:
			w_min = w
			err_min = err_w

	# Devuelve el w que produce el error minimo
	return w_min

w = pocket_PLA(x,y,w)


print('E(in)', Err(x,y,w))
print('E(test)', Err(x_test,y_test	,w))

Y = list()
for i in x[: ,1]:
    Y.append( (-w[0] - w[1]*i) / w[2] )


fig, ax = plt.subplots()
ax.plot(x[:, 1], Y, c = 'black')
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
ax.set_ylim((-8, 0))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(x[:, 1], Y, c = 'black')
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
ax.set_ylim((-8, 0))
plt.legend()
plt.show()



input("\n--- Pulsar tecla para continuar ---\n")

#COTA SOBRE EL ERROR

print('Cota con E(in)', Err(x,y,w) + np.sqrt( (1/len(x)) * np.log(2/0.05) ) )
print('Cota con E(test)', Err(x_test,y_test,w) + np.sqrt( (1/len(x)) * np.log(2/0.05) ) )

#CODIGO DEL ESTUDIANTE
