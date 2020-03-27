# -*- coding: utf-8 -*-
"""
EJERCICIO 1.
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.markers as mark
import math
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1 y 2\n')

#################################################################################
#   FUNCIONES
#################################################################################

# Funcion E(u,v)
def E(u,v):
    return np.float( (u*np.exp(v) - 2*v*np.exp(u*-1))**2 )

# Funcion E(u,v) que no devuelve el valor en float64 (Necesario para graficar eje Z)
def E_nf(u,v):
    return (u*np.exp(v) - 2*v*np.exp(u*-1))**2

# Derivada parcial de E con respecto a u
def dEu(u,v):
    return np.float( 2 * (u*np.exp(v) - 2*v*np.exp(-u)) * (np.exp(v) + 2*v*np.exp(-u)) )

# Derivada parcial de E con respecto a v
def dEv(u,v):
    return np.float( 2 * (u*np.exp(v) - 2*v*np.exp(-u)) * (u*np.exp(v) - 2*np.exp(-u)) )

# Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# Funcion F(x,y)
def F(x,y):
    return np.float( (x-2)**2 + 2*(y+2)**2 + (2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))  )

# Funcion F(x,y) que no devuelve el valor en float64 (Necesario para graficar eje Z)
def F_nf(x,y):
    return ( (x-2)**2 + 2*(y+2)**2 + (2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))  )

# Derivada parcial de F con respecto a x
def dFx(x,y):
    return np.float( 2*x + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) - 4 )

# Derivada parcial de F con respecto a y
def dFy(x,y):
    return np.float( 4*y + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + 8 )

# Gradiente de F
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])

# Funcion de gradiente descendente
# Parametros:
#   f   -> Funcion a la que se quiere aplicar el gradiente
#   g   -> Funcion gradiente que devuelve un vector con las derivadas parciales (gradE y gradF)
#   w   -> Punto inicial
#   n   -> Tasa de aprendizaje
#   iterations  -> Numero máximo de iteraciones
#   min -> Valor minimo a alcanzar
#   Devuelve las cordenadas w y el numero de iteraciones
def gradient_descent(f,g,ini,n,iterations,min):
    i = 0
    # Inicializamos w a un vector vacio de 0's
    w = np.array([0,0],np.float64)
    # Guardamos los valores iniciales de u y v
    u = ini[0]
    v = ini[1]
    # Mientras no superemos las iteraciones maximas y no obtengamos un valor menos
    # al minimo:
    while f(u,v) > min and i < iterations:
        grad = g(u,v) # Guardamos en grad el gradiente de la funcion
        # Actualizamos los valores de u y v
        u = u - n*grad[0]
        v = v - n*grad[1]
        i = i + 1

    # Cuando el bucle termina guardamos en w, los ultimos valores de u y v
    w[0] = u
    w[1] = v
    iterations = i

    # Devolvemos w y las iteraciones
    return w, iterations

#################################################################################
# Aplicamos el gradiente a la funcion E
eta = 0.1   # Tasa de aprendizaje
maxIter = 10000000000   # Iteraciones maximas
error2get = 1e-14   # Error minimo
initial_point = np.array([1.0,1.0]) # Punto de inicio
w, it = gradient_descent(E,gradE,initial_point,eta,maxIter,error2get)  # Funcion gradiente

# Imprimimos los resultados
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Tasa de aprendizaje: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# Una vez sabemos el numero de iteraciones, guardamos los valores calculados por
# el gradiente en cada iteracion para poder graficarlos
val_x = list()
val_y = list()
for i in range(it+1):
    w , it = gradient_descent(E,gradE,initial_point,eta,i,error2get)
    val_x.append(w[0])
    val_y.append(w[1])
val_x = np.array(val_x)
val_y = np.array(val_y)

#################################################################################
# Mostramos el grafico
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)

Z = E_nf(X,Y)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet',alpha=0.3)
ax.plot(val_x, val_y, E_nf(val_x, val_y), c = 'red',marker='*')
ax.set(title='Ejercicio 2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
legend_elements = [mlines.Line2D([],[],linewidth=0,marker='*', color='red', label='Punto calculado con gradiente', markersize=10)]
plt.legend(handles=legend_elements,loc='lower right')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#################################################################################
print('Ejercicio 3.1\n')

# Aplicamos el gradiente a la funcion F con tasa de aprendizaje 0.01
eta = 0.01  # Tasa de aprendizaje
maxIter = 50  # Iteraciones maximas
error2get = 0  # Error minimo
initial_point = np.array([1.0,-1.0]) # Punto de inicio
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Tasa de aprendizaje: ', eta)

# Guardamos los valores obtenidos en las iteraciones en dos listas
valores1_x = list()
valores1_y = list()
for i in range(maxIter+1):
    w1, it = gradient_descent(F,gradF,initial_point,eta,i,error2get)  # Funcion gradiente
    valores1_x.append(w1[0])
    valores1_y.append(w1[1])
valores1_x = np.array(valores1_x)
valores1_y = np.array(valores1_y)
print ('Coordenadas obtenidas: (', w1[0], ', ', w1[1],')')

# Aplicamos el gradiente a la funcion F con tasa de aprendizaje 0.01
eta = 0.1  #Tasa de aprendizaje
print ('Punto de inicio: (', initial_point[0], ', ', initial_point[1],')')
print ('Tasa de aprendizaje: ', eta)

# Guardamos los valores obtenidos en las iteraciones en dos listas
valores2_x = list()
valores2_y = list()
for i in range(maxIter+1):
    w2, it = gradient_descent(F,gradF,initial_point,eta,i,error2get)  # Funcion gradiente
    valores2_x.append(w2[0])
    valores2_y.append(w2[1])
valores2_x = np.array(valores2_x)
valores2_y = np.array(valores2_y)
print ('Coordenadas obtenidas: (', w1[0], ', ', w1[1],')')

# Añadimos a una lista los indices que hemos recorrido
a = list()
for i in range(maxIter+1):
    a.append(i)
a = np.array(a)

#################################################################################
# Mostramos el grafico de como descienden los gradientes con 0,1 y 0,01 de tasa de aprendizaje
plt.plot(a, F_nf(valores1_x,valores1_y), c = 'green',marker='o')
plt.plot(a, F_nf(valores2_x,valores2_y), c = 'red',marker='*')
plt.title('Ejercicio 3.1. Descenso del valor de la función por cada iteración')
plt.xticks()
plt.yticks()
plt.xlabel('Iteraciones')
plt.ylabel('F(x,y)')
legend_elements = [mlines.Line2D([], [], color='green',marker='o',markersize=10, label='Tasa de aprendizaje = 0.01'),
				   mlines.Line2D([], [], color='red' ,marker='*',markersize=10, label='Tasa de aprendizaje = 0.1')]
plt.legend(handles=legend_elements)

plt.show()

#################################################################################
input("\n--- Pulsar tecla para continuar ---\n")
# Mostramos el grafico
# La idea era que la recta se viera por encima de la superfice, pero no he conseguido hacerlo
x = np.linspace(1, 3, 50)
y = np.linspace(-3, 0, 50)
X, Y = np.meshgrid(x, y)

Z = F_nf(X,Y)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(valores1_x, valores1_y, F_nf(valores1_x, valores1_y), color = 'green', marker='o', markersize=10)
ax.plot(valores2_x, valores2_y, F_nf(valores2_x, valores2_y), color = 'red', marker='*', markersize=10)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet',alpha=0.3)
ax.set(title='Ejercicio 3.1. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('F(x,y)')
legend_elements = [mlines.Line2D([],[],linewidth=0,marker='*', color='red', label='Punto tasa de aprendizaje = 0.1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='green', label='Punto tasa de aprendizaje = 0.01', markersize=10)]
plt.legend(handles=legend_elements,loc='lower left')
plt.show()

#################################################################################
input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 3.2\n')

eta = 0.1  # Tasa de aprendizaje
maxIter = 1000   # Iteraciones maximas
error2get = 0    # Error minimo

# Guardamos en una lista los puntos iniciales
initial_point = []
initial_point.append([2.1,-2.1])
initial_point.append([3.0,-3.0])
initial_point.append([1.5,1.5])
initial_point.append([1.0,-1.0])

# Convertimos esta lista en dos array np
initial_point = np.array(initial_point)

# Bucle para obtener los gradientes con una tasa de aprendizaje = 0.1
for i in range(4):
    ini = initial_point[i]
    print ('------------------------------------------------')
    print ('Punto de inicio: (', ini[0], ', ', ini[1],')')
    w, it = gradient_descent(F,gradF,ini,eta,maxIter,error2get)
    print ('Tasa de aprendizaje: ', eta)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('Valor de la funcion para w: ', F(w[0],w[1]))
    print ('------------------------------------------------')

eta = 0.01  # Actualizamos la tasa de aprendizaje

# Bucle para obtener los gradientes con una tasa de aprendizaje = 0.01
for i in range(4):
    ini = initial_point[i]
    print ('------------------------------------------------')
    print ('Punto de inicio: (', ini[0], ', ', ini[1],')')
    w, it = gradient_descent(F,gradF,ini,eta,maxIter,error2get)
    print ('Tasa de aprendizaje: ', eta)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('Valor de la funcion para w: ', F(w[0],w[1]))
    print ('------------------------------------------------')
