# -*- coding: utf-8 -*-
"""
EJERCICIO 2.
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
	return np.mean((np.dot(x,w.T) - y)**2)

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


print('Calculando gradiente de la funcion...')
w1 = sgd(x,y,0.1,1000)
print(w1.shape)
print ('Coordenadas obtenidas: (', w1[0], ', ', w1[1], ', ', w1[2],')')
print ('Bondad del resultado para grad. descendente estocastico:')
print ('Tasa de aprendizaje: ', 0.1)
print ('Numero de iteraciones: ', 1000	)
print ("Ein (medio): ", Err(x,y,w1))
print ("Eout (medio): ", Err(x_test, y_test, w1))

print('\n')

print('Calculando gradiente de la funcion...')
w2 = pseudoinverse(x,y)
print(w2.shape)
print ('Coordenadas obtenidas: (', w2[0], ', ', w2[1], ', ', w2[2],')')
print ('Bondad del resultado para pseudoinversa:')
print ("Ein (medio): ", Err(x,y,w2))
print ("Eout (medio): ", Err(x_test, y_test, w2))

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
    b.append( (-w1[0] - w1[1]*i) / w1[2] )
for i in a:
	c.append( (-w2[0] - w2[1]*i) / w2[2] )

plt.plot(a, b, c = 'black')
plt.plot(a, c, c = 'grey')
plt.scatter(x[:, 1], x[:, 2], c=colist)
plt.title('Ejercicio 2.1. Grad. descendente estocastico vs Pseudoinversa')
plt.xticks()
plt.yticks()
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
legend_elements = [mpatches.Patch(color='black', label='SGD'),
				   mpatches.Patch(color='grey', label='Pseudoinversa'),
                   mpatches.Patch(color='blue', label='5'),
                   mpatches.Patch(color='red', label='1')]
plt.legend(handles=legend_elements)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('Ejercicio 2.1\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

space = simula_unif(1000,2,1)

print(space)

plt.clf()
plt.scatter(space[:, 0], space[:, 1])
plt.title('Ejercicio 2.2. Mapa de puntos generado')
plt.xticks()
plt.yticks()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 2.2\n')

def F(x1,x2):
	return np.sign((x1-0.2)**2 + x2**2 - 0.6)

f = list()
for i in space:
	f.append(F(i[0],i[1]))

f = np.array(f)

print(f)

index = np.random.choice(1000, 100, replace=False)

for i in index:
	f[i] = -1*f[i]

colist=[]
for i in f:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

plt.clf()
plt.scatter(space[:, 0], space[:, 1], c=colist)
plt.title('Ejercicio 2.2. Mapa de etiquetas')
plt.xticks()
plt.yticks()
plt.xlabel('x1')
plt.ylabel('x2')
legend_elements =  [mpatches.Patch(color='blue', label='1'),
                   mpatches.Patch(color='red', label='-1')]
plt.legend(handles=legend_elements)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 2.3\n')
print('Calculando gradiente de la funcion...')

x = list()
for i in space:
	x.append([1.0,i[0],i[1]])
x = np.array(x)

w = sgd(x,f,0.1,1000)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1], ', ', w[2],')')
print ("Ein (medio): ", Err(x,f,w))

colist=[]
for i in f:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

a = np.linspace(-0.005,0.25,num=100)
b = list()

for i in a:
	b.append( (-w[0] - w[1]*i) / w[2] )


plt.clf()
plt.plot(a, b, c = 'black')
plt.scatter(space[:, 0], space[:, 1], c=colist)
plt.title('Ejercicio 2.3. Ajuste lineal')
plt.xticks()
plt.yticks()
plt.xlabel('x1')
plt.ylabel('x2')
legend_elements = [mpatches.Patch(color='black', label='SGD'),
				   mpatches.Patch(color='blue', label='1'),
                   mpatches.Patch(color='red', label='-1')]
plt.legend(handles=legend_elements)

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 2.5\n')
print('Calculando gradiente de la funcion...')
x = list()
for i in space:
	x.append([1.0,i[0],i[1],i[0]*i[1],i[0]*i[0],i[1]*i[1]])
x = np.array(x)

w = sgd(x,f,0.1,1000)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1], ', ', w[2], ', ', w[3], ', ', w[4], ', ', w[5],')')
print ("Ein (medio): ", Err(x,f,w))

colist=[]
for i in f:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

a = np.linspace(-0.005,0.25,num=100)
b = list()

for i in a:
	b.append( (-w[0] - w[1]*i) / w[2] )


plt.clf()
x = np.linspace(-1.0, 1.05, 100)
y = np.linspace(-1.0, 1.05, 100)
X, Y = np.meshgrid(x,y)
G = w[5]*Y**2 + w[4]*X**2 + w[3]*X*Y + w[2]*Y + w[1]*X + w[0]
plt.contour(X,Y,G,[0])
plt.scatter(space[:, 0], space[:, 1], c=colist)
plt.title('Ejercicio 2.5. Ajuste no lineal')
plt.xticks()
plt.yticks()
plt.xlabel('x1')
plt.ylabel('x2')
legend_elements = [mpatches.Patch(color='black', label='SGD'),
				   mpatches.Patch(color='blue', label='1'),
                   mpatches.Patch(color='red', label='-1')]
plt.legend(handles=legend_elements)

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 2.4\n')
print('Son 1000 iteraciones ( TARDA )\n')

ein = 0
eout = 0
for i in range(1000):
	print(i)
	for j in range(2):
		space = simula_unif(1000,2,1)

		f = list()
		for k in space:
			f.append(F(k[0],k[1]))

		f = np.array(f)

		index = np.random.choice(1000, 100, replace=False)
		for ind in index:
			f[ind] = -1*f[ind]

		x = list()
		for s in space:
			x.append([1.0,s[0],s[1]])
		x = np.array(x)

		if j == 0:
			w = sgd(x,f,0.1,1000)
			ein += Err(x,f,w)
		else:
			eout += Err(x,f,w)

	print('Ein medio: ', ein/(i+1))
	print('Eout medio: ', eout/(i+1))

print('Ein medio: ', ein/1000.0)
print('Eout medio: ', eout/1000.0)
