# -*- coding: utf-8 -*-
"""
EJERCICIO 1.
Nombre Estudiante: Alejandro Manzanares Lemus
"""

import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1 y 2\n')

#################################################################################
#   FUNCIONES EJERCICIO 1 Y GRADIENTE
#################################################################################

#Funcion E(u,v)
def E(u,v):
    return np.float( (u*np.exp(v) - 2*v*np.exp(u*-1))**2 )

#Funcion E(u,v) que no devuelve el valor en float64 (Necesario para Z)
def E_nf(u,v):
    return (u*np.exp(v) - 2*v*np.exp(u*-1))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return np.float( 2 * (u*np.exp(v) - 2*v*np.exp(-u)) * (np.exp(v) + 2*v*np.exp(-u)) )

#Derivada parcial de E con respecto a v
def dEv(u,v):
    return np.float( 2 * (u*np.exp(v) - 2*v*np.exp(-u)) * (u*np.exp(v) - 2*np.exp(-u)) )

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#Funcion de gradiente descendente
#Parametros:
#   f   -> Funcion a la que se quiere aplicar el gradiente
#   g   -> Funcion gradiente que devuelve un vector con las derivadas parciales (gradE y gradF)
#   w   -> Punto inicial
#   n   -> Tasa de aprendizaje
#   iterations  -> Numero máximo de iteraciones
#   min -> Valor minimo a alcanzar
#   Devuelve las cordenadas w y el numero de iteraciones
def gradient_descent(f,g,w,n,iterations,min):
    print('Funcion gradiente para: ', min, ' -> ',iterations)
    i = 0
    u = w[0]
    v = w[1]
    while f(u,v) > min and i < iterations:
        grad = g(u,v)
        u = u - 0.1*grad[0]
        v = v - 0.1*grad[1]
        i = i + 1


    w[0] = u
    w[1] = v
    iterations = i

    return w, iterations

#################################################################################
# Aplicamos el gradiente a la funcion E
eta = 0.1
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(E,gradE,initial_point,eta,maxIter,error2get);

print ('Tasa de aprendizaje: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
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
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print('Ejercicio 3.1\n')

def F(x,y):
    return np.float( (x-2)**2 + 2*(y+2)**2 + (2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))  )

def F_nf(x,y):
    return ( (x-2)**2 + 2*(y+2)**2 + (2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))  )

#Derivada parcial de E con respecto a u
def dFx(x,y):
    return np.float( 2*x + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) - 4 )

#Derivada parcial de E con respecto a v
def dFy(x,y):
    return np.float( 4*y + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) - 8 )

#Gradiente de E
def gradF(x,y):
    return np.array([dFx(x,y), dFy(x,y)])


eta = 0.01
maxIter = 50
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w1, it = gradient_descent(F,gradF,initial_point,eta,maxIter,error2get);

print ('Tasa de aprendizaje: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w1[0], ', ', w1[1],')')

eta = 0.1
maxIter = 50
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w2, it = gradient_descent(F,gradF,initial_point,eta,maxIter,error2get);

print ('Tasa de aprendizaje: ', eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w2[0], ', ', w2[1],')')


# DISPLAY FIGURE
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
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print('Ejercicio 3.2\n')

eta = 0.1
maxIter = 1000
error2get = -np.Infinity

initial_point = []
initial_point.append([2.1,-2.1])
initial_point.append([3.0,-3.0])
initial_point.append([1.5,1.5])
initial_point.append([1.0,-1.0])
initial_point = np.array(initial_point)
initial_point1 = np.copy(initial_point)

for i in range(4):
    ini = initial_point[i]
    print ('------------------------------------------------')
    print ('Punto de inicio: (', ini[0], ', ', ini[1],')')
    w, it = gradient_descent(F,gradF,ini,eta,maxIter,error2get);
    print ('Tasa de aprendizaje: ', eta)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('------------------------------------------------')

eta = 0.01

for i in range(4):
    ini = initial_point1[i]
    print ('------------------------------------------------')
    print ('Punto de inicio: (', ini[0], ', ', ini[1],')')
    w, it = gradient_descent(F,gradF,ini,eta,maxIter,error2get);
    print ('Tasa de aprendizaje: ', eta)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
    print ('------------------------------------------------')
