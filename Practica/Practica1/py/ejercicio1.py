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
print('Ejercicio 1\n')

def E(u,v):
    return np.float((u*math.e**v - 2*v*math.e**(u*-1))**2)

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return np.float(2*(u*math.e**v - 2*v*math.e**(u*-1))*(math.e**v + 2*v*math.e**(u*-1)))

#Derivada parcial de E con respecto a v
def dEv(u,v):
    return np.float(2*(u*math.e**v - 2*v*math.e**(u*-1))*(u*math.e**v - 2*math.e**(u*-1)))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(w,n,iterations,min):
    print('Funcion gradiente para: ', min, ' -> ',iterations)
    i = 0
    u = w[0]
    v = w[1]

    while E(u,v) > min or i < iterations:
        d = gradE(u,v);
        u = u - 0.1*dEu(u,v)
        v = v - 0.1*dEv(u,v)
        i = i + 1
        print('E(',u,',',v,'): ', E(u,v), ',' ,i,'\n')

    print(E(u,v))
    return w, iterations


eta = 0.01
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it = gradient_descent(initial_point,eta,maxIter,error2get);


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...
