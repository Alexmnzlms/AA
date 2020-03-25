# -*- coding: utf-8 -*-
"""
EJERCICIO 2.
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))

	x = np.array(x, np.float64)
	y = np.array(y, np.float64)

	return x, y

# Funcion para calcular el error
def Err(x,y,w):
	error = []
	elements = np.size(x,0)
	for i in range(elements):
		error.append((np.dot(x[i],w.T) - y[i])**2)

	error = np.array(error)

	return error

# Gradiente Descendente Estocastico
def sgd(x,y,n,iterations):
	w = np.zeros(x[0].size)
	c = 0

	while c < iterations:
		batch = np.random.choice(np.size(x,0), 128, replace=False)
		w_ant = np.copy(w)
		c = c + 1
		for i in range(np.size(w)):
			sumatoria = 0
			for j in batch:
				sumatoria = sumatoria + x[j][i]*(np.dot(x[j],w.T) - y[j])

			w[i] = w_ant[i] -n * (2.0/np.float(np.size(batch))) * sumatoria

	return w


# Pseudoinversa
def pseudoinverse(x,y):
	a = np.dot(x.T,x)
	b = np.linalg.inv(a)
	pseudo = np.dot(b,x.T)
	w = np.dot(pseudo,y)

	return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w1 = sgd(x,y,0.1,1000)
print(w1.shape)
print ('Coordenadas obtenidas: (', w1[0], ', ', w1[1], ', ', w1[2],')')
print ('Bondad del resultado para grad. descendente estocastico:')
print ("Ein (medio): ", np.mean(Err(x,y,w1)))
print ("Eout (medio): ", np.mean(Err(x_test, y_test, w1)))

print('\n')

w2 = pseudoinverse(x,y)
print(w2.shape)
print ('Coordenadas obtenidas: (', w2[0], ', ', w2[1], ', ', w2[2],')')
print ('Bondad del resultado para pseudoinversa:')
print ("Ein (medio): ", np.mean(Err(x,y,w2)))
print ("Eout (medio): ", np.mean(Err(x_test, y_test, w2)))

input("\n--- Pulsar tecla para continuar ---\n")

print(-4*w1[2]+0.3*w1[1]+w1[0])
print(-1*w1[2]+0.15*w1[1]+w1[0])


colist=[]
for i in y:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

a = np.linspace(0,1,num=100)
b = list()
c = list()

for i in a:
    b.append( w1[2]*i + w1[1]*i + w1[0] )
for i in a:
	c.append( w2[2]*i + w2[1]*i + w1[0] )

plt.plot(a, b, c = 'black')
plt.plot(a, c, c = 'grey')
plt.scatter(x[:, 1], x[:, 2], c=colist)
plt.title('Gráfico 1:')
plt.xticks()
plt.yticks()
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')

plt.show()

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

#Seguir haciendo el ejercicio...
