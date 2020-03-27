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
    return np.float( 4*y + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) - 8 )

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
def gradient_descent(f,g,w,n,iterations,min):
    print('Funcion gradiente para: ', f)
    i = 0
    # Guardamos los valores iniciales de u y v
    u = w[0]
    v = w[1]
    # Mientras no superemos las iteraciones maximas y no obtengamos un valor menos
    # al minimo:
    while f(u,v) > min and i < iterations:
        grad = g(u,v) # Guardamos en grad el gradiente de la funcion
        # Actualizamos los valores de u y v
        u = u - 0.1*grad[0]
        v = v - 0.1*grad[1]
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
print ('Tasa de aprendizaje: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

#################################################################################
# Mostramos el grafico
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)

Z = E_nf(X,Y)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
legend_elements = [mlines.Line2D([],[],linewidth=0,marker='*', color='red', label='Punto calculado con gradiente', markersize=10)]
plt.legend(handles=legend_elements,loc='lower left')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

#################################################################################
print('Ejercicio 3.1\n')

# Aplicamos el gradiente a la funcion F con tasa de aprendizaje 0.01
eta = 0.01  # Tasa de aprendizaje
maxIter = 1000  # Iteraciones maximas
error2get = 1e-14   # Error minimo
initial_point = np.array([1.0,1.0]) # Punto de inicio
w1, it = gradient_descent(F,gradF,initial_point,eta,maxIter,error2get)  # Funcion gradiente

# Imprimimos los resultados
print ('Tasa de aprendizaje: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w1[0], ', ', w1[1],')')

# Aplicamos el gradiente a la funcion F con tasa de aprendizaje 0.01
eta = 0.1  #Tasa de aprendizaje
maxIter = 1000  # Iteraciones maximas
error2get = 1e-14   # Error minimo
initial_point = np.array([1.0,1.0]) # Punto de inicio
w2, it = gradient_descent(F,gradF,initial_point,eta,maxIter,error2get)  # Funcion gradiente

# Imprimimos los resultados
print ('Tasa de aprendizaje: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w2[0], ', ', w2[1],')')

#################################################################################
# Mostramos el grafico
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)

Z = F_nf(X,Y)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet')
min_point1 = np.array([w1[0],w1[1]])
min_point1_ = min_point1[:, np.newaxis]
ax.plot(min_point1_[0], min_point1_[1], F(min_point1_[0], min_point1_[1]), 'go', markersize=10)
min_point2 = np.array([w2[0],w2[1]])
min_point2_ = min_point2[:, np.newaxis]
ax.plot(min_point2_[0], min_point2_[1], F(min_point2_[0], min_point2_[1]), 'r*', markersize=10)
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
error2get = -np.Infinity    # Error minimo

# Guardamos en una lista los puntos iniciales
initial_point = []
initial_point.append([2.1,-2.1])
initial_point.append([3.0,-3.0])
initial_point.append([1.5,1.5])
initial_point.append([1.0,-1.0])

# Convertimos esta lista en dos array np
initial_point = np.array(initial_point)
initial_point1 = np.copy(initial_point)

# Bucle para obtener los gradientes con una tasa de aprendizaje = 0.1
for i in range(4):
    ini = initial_point[i]
    print ('------------------------------------------------')
    print ('Punto de inicio: (', ini[0], ', ', ini[1],')')
    w, it = gradient_descent(F,gradF,ini,eta,maxIter,error2get)
    print ('Tasa de aprendizaje: ', eta)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('------------------------------------------------')

eta = 0.01  # Actualizamos la tasa de aprendizaje

# Bucle para obtener los gradientes con una tasa de aprendizaje = 0.01
for i in range(4):
    ini = initial_point1[i]
    print ('------------------------------------------------')
    print ('Punto de inicio: (', ini[0], ', ', ini[1],')')
    w, it = gradient_descent(F,gradF,ini,eta,maxIter,error2get)
    print ('Tasa de aprendizaje: ', eta)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('------------------------------------------------')
