# -*- coding: utf-8 -*-
"""
EJERCICIO 2.
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

np.random.seed(1)
#################################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1
#################################################################################
#	FUNCIONES
#################################################################################
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
# Parametros:
#   x   -> Vector de datos X con n caracteristicas
#   y   -> Vector de etiquetas Y asociado a X
#   w   -> Pesos del ajuste de la funcion
#   Devuelve la media de (X*Wt - Y)**2
def Err(x,y,w):
	return np.mean((np.dot(x,w.T) - y)**2)

# Gradiente Descendente Estocastico
# Parametros:
#   x   -> Vector de datos X con n caracteristicas
#   y   -> Vector de etiquetas Y asociado a X
#   n   -> Tasa de aprendizaje
#   iterations  -> Numero máximo de iteraciones
#   Devuelve w -> los pesos del ajuste de la funcion
def sgd(x,y,n,iterations):
	w = np.zeros(x[0].size) # Inicializamos w al vector de tantos 0's como
							# caracteristicas tiene x
	c = 0
	# Mientras no se supere el numero maximo de iteraciones
	while c < iterations:
		# Obtenemos la submuestra de X
		batch = np.random.choice(np.size(x,0), 128, replace=False)
		# Copiamos en w_ant el valor anterior de w
		w_ant = np.copy(w)
		c = c + 1
		# Para cada wj
		for i in range(np.size(w)):
			sumatoria = 0
			# Calculamos la sumatoria de cada x que pertenece a la submuestra
			for j in batch:
				sumatoria = sumatoria + x[j][i]*(np.dot(x[j],w.T) - y[j])
			# Actualizamos el valor de wj
			w[i] = w_ant[i] -n * (2.0/np.float(np.size(batch))) * sumatoria
	# Devolvemos w
	return w

# Pseudoinversa
# Parametros:
#   x   -> Vector de datos X con n caracteristicas
#   y   -> Vector de etiquetas Y asociado a X
#   Devuelve w -> los pesos del ajuste de la funcion
def pseudoinverse(x,y):
	# X*Xt
	a = np.dot(x.T,x)
	# (X*Xt)**-1
	b = np.linalg.inv(a)
	# (X*Xt)**-1 * Xt
	pseudo = np.dot(b,x.T)
	# (X*Xt)**-1 * Xt * Y
	w = np.dot(pseudo,y)
	# Devolvemos w
	return w

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

# Funcion que dada dos caracteristica x1 y x2, devuelve 1 o -1
def F(x1,x2):
	return np.sign((x1-0.2)**2 + x2**2 - 0.6)

#################################################################################
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

# Aplicamos el gradiente descendente estocastico para obtener los pesos w
print('Aplicando gradiente descendente estocastico...')
w1 = sgd(x,y,0.1,1000)
# Imprimimos los datos del gradiente, los pesos y los errores in y out
print ('Pesos: (', w1[0], ', ', w1[1], ', ', w1[2],')')
print ('Bondad del resultado para grad. descendente estocastico:')
print ('Tasa de aprendizaje: ', 0.1)
print ('Numero de iteraciones: ', 1000	)
print ("Ein: ", Err(x,y,w1))
print ("Eout: ", Err(x_test, y_test, w1))

print('\n')

# Aplicamos la pseudoinversa para obtener los pesos w
print('Aplicando pseudoinversa...')
w2 = pseudoinverse(x,y)
# Imprimimos los pesos y los errores in y out
print ('Pesos: (', w2[0], ', ', w2[1], ', ', w2[2],')')
print ('Bondad del resultado para pseudoinversa:')
print ("Ein: ", Err(x,y,w2))
print ("Eout: ", Err(x_test, y_test, w2))

#################################################################################
# Obtenemos los colores para cada tipo de dato
colist=[]
for i in y:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

# Obtenemos los puntos de la funcion
a = np.linspace(0,1,num=100)
b = list()
c = list()

# A partir de una caracteristica y los pesos, sacamos otra caracteristica
# w2x2 + w1x1 + w0 = 0
for i in a:
    b.append( (-w1[0] - w1[1]*i) / w1[2] )
for i in a:
	c.append( (-w2[0] - w2[1]*i) / w2[2] )

#################################################################################
# Mostramos el grafico
plt.plot(a, b, c = 'black')
plt.plot(a, c, c = 'grey')
plt.scatter(x[:, 1], x[:, 2], c=colist)
plt.title('Ejercicio 2.1. Grad. descendente estocastico vs Pseudoinversa')
plt.xticks()
plt.yticks()
plt.xlabel('Intensidad promedio')
plt.ylabel('Simetria')
legend_elements = [mlines.Line2D([], [], color='black',markersize=15, label='SGD'),
				   mlines.Line2D([], [], color='grey' ,markersize=15, label='Pseudoinversa'),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='5', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='1', markersize=10)]
plt.legend(handles=legend_elements)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#################################################################################
print('Ejercicio 2.1\n')

# Creamos un espacio uniforme de 1000 puntos
space = simula_unif(1000,2,1)
# Imprimimos el espacio de puntos que hemos generado
print(space)

#################################################################################
# Mostramos el grafico
plt.clf()
plt.scatter(space[:, 0], space[:, 1])
plt.title('Ejercicio 2.2. Mapa de puntos generado')
plt.xticks()
plt.yticks()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#################################################################################
print('Ejercicio 2.2\n')

# Creamos la lista f donde introduciremos las etiquetas asociadas a los puntos
f = list()
for i in space:
	f.append(F(i[0],i[1])) # Aplicamos la funcion F a cada punto del espacio

f = np.array(f) # Convertimos la lista en un array np

# Imprimimos las etiquetas
print(f)

# Elegimos un 10% de las etiquetas
index = np.random.choice(1000, 100, replace=False)

# Multiplicamos por -1 para obtener el 10% de ruido
for i in index:
	f[i] = -1*f[i]

#################################################################################
# Mostramos el grafico

# Obtenemos los colores para cada tipo de dato
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
legend_elements = [mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='-1', markersize=10)]
plt.legend(handles=legend_elements)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#################################################################################
print('Ejercicio 2.3\n')
print('Calculando gradiente de la funcion...')
# AJUSTE LINEAL DE LOS DATOS
# Creamos la lista X donde introduciremos 1 y las cordenadas de los puntos del espacio
# para obtener el vector de caracteristicas (1,x1,x2)
x = list()
for i in space:
	x.append([1.0,i[0],i[1]])
x = np.array(x) # Convertimos la lista x en un array np

# Aplicamos el gradiente descendente estocastico para obtener los pesos de la funcion
# y = w2x2 + w1x1 + w0
w = sgd(x,f,0.1,1000)
# Imprimimos los pesos obtenidos y el error de entrada
print ('Pesos: (', w[0], ', ', w[1], ', ', w[2],')')
print ("Ein: ", Err(x,f,w))

#Obtenemos los valores de x1
a = np.linspace(-0.005,0.25,num=100)
# Conociendo x1 y w, obtenemos valores de x2 para x2w2 + x1w1 + w0 = 0
b = list()
for i in a:
	b.append( (-w[0] - w[1]*i) / w[2] )

#################################################################################
# Mostramos el grafico
plt.clf()
plt.plot(a, b, c = 'black')
plt.scatter(space[:, 0], space[:, 1], c=colist)
plt.title('Ejercicio 2.3. Ajuste lineal')
plt.xticks()
plt.yticks()
plt.xlabel('x1')
plt.ylabel('x2')
legend_elements = [mlines.Line2D([], [], color='black',markersize=15, label='SGD'),
				   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='1', markersize=10),
				   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='-1', markersize=10)]
plt.legend(handles=legend_elements)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#################################################################################
print('Ejercicio 2.5\n')
print('Calculando gradiente de la funcion...')
# AJUSTE NO LINEAL DE LOS DATOS
# Creamos la lista X donde introduciremos 1 y las cordenadas de los puntos del espacio
# para obtener el vector de caracteristicas (1,x1,x2,x1x2,x1x1,x2x2)
x = list()
for i in space:
	x.append([1.0,i[0],i[1],i[0]*i[1],i[0]*i[0],i[1]*i[1]])
x = np.array(x)
# Aplicamos el gradiente descendente estocastico para obtener los pesos de la funcion
# y = w5x2**2 + w4x1**2 + w3x1x2 + w2x2 + w1x1 + w0
w = sgd(x,f,0.1,1000)
# Imprimimos los pesos obtenidos y el error de entrada
print ('Pesos: (', w[0], ', ', w[1], ', ', w[2], ', ', w[3], ', ', w[4], ', ', w[5],')')
print ("Ein (medio): ", Err(x,f,w))

#Obtenemos los valores de x1
a = np.linspace(-0.005,0.25,num=100)

#################################################################################
# Mostramos el grafico
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
legend_elements = [mlines.Line2D([], [], color='black',markersize=15, label='SGD'),
				   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='1', markersize=10),
				   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='-1', markersize=10)]
plt.legend(handles=legend_elements, loc='lower left')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#################################################################################
print('Ejercicio 2.4\n')
print('Son 1000 iteraciones ( TARDA )\n')

ein = 0 # Error de entrada inicial
eout = 0 # Error de salida inicial
for i in range(1000):
	# Para 1000 iteraciones
	print(i) # Imprimimos la iteracion actual
	for j in range(2):
		# Dos veces
		space = simula_unif(1000,2,1) # Generamos un espacio de 1000 puntos
		# Etiquetas asociadas a los puntos del espacio
		f = list()
		for k in space:
			f.append(F(k[0],k[1]))
		f = np.array(f)
		# Introducimos un 10% de ruido
		index = np.random.choice(1000, 100, replace=False)
		for ind in index:
			f[ind] = -1*f[ind]
		# Creamos los vectores de caracteristicas (1,x1,x2)
		x = list()
		for s in space:
			x.append([1.0,s[0],s[1]])
		x = np.array(x)
		# La primera vez calculamos el gradiente descendente estocastico
		# y calculamos el error de entrada actual
		if j == 0:
			w = sgd(x,f,0.1,1000)
			ein += Err(x,f,w)
		else:
		# La segunda vez calculamos el error de salida actual para el ajuste
		# anterior y un nuevo espacio de 1000 puntos
			eout += Err(x,f,w)
	# Imprimimos el error actual
	print('Ein medio: ', ein/(i+1))
	print('Eout medio: ', eout/(i+1))
# Finalmente Imprimimos el error medio obtenido en las 1000 iteraciones
print('Ein medio: ', ein/1000.0)
print('Eout medio: ', eout/1000.0)
