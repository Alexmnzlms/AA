# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Alejandro Manzanares Lemus
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import math



# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0
    out = np.zeros((N,dim),np.float64)
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0]))
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)

    return out


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


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

x = simula_unif(50, 2, [-50,50])
#CODIGO DEL ESTUDIANTE
plt.clf()
plt.scatter(x[:, 0], x[:, 1])
plt.title('Ejercicio 1.1. Nube de puntos generada con simula_unif')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = simula_gaus(50, 2, np.array([5,7]))
#CODIGO DEL ESTUDIANTE
plt.clf()
plt.scatter(x[:, 0], x[:, 1])
plt.title('Ejercicio 1.1. Nube de puntos generada con simula_gauss')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

#CODIGO DEL ESTUDIANTE

x = simula_unif(100, 2, [-50,50])

a,b = simula_recta([-50,50])

f_values = list()
for i in range(100):
	f_values.append(f(x[i][0],x[i][1],a,b))

f_values = np.array(f_values)


colist=[]
for i in f_values:
	if i == 1:
	    colist.append('red')
	else:
		colist.append('blue')

y = x[:,0]*a + b

#################################################################################
# Mostramos el grafico
plt.plot(x[:, 0], y, c = 'black')
plt.scatter(x[:, 0], x[:, 1], c=colist)
plt.title('Ejercicio 2.1. Nube de puntos generada')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
legend_elements = [mlines.Line2D([], [], color='black',markersize=15, label='Recta generada por simula_recta'),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='-1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='1', markersize=10)]
plt.legend(handles=legend_elements)

plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#CODIGO DEL ESTUDIANTE

positive = list()
negative = list()

for i in f_values:
	if(i == -1):
		negative.append(i)
	else:
		positive.append(i)

print(positive)
print(negative)

print(len(positive))
print(math.ceil(0.1*len(positive)))

print(len(negative))
print(math.ceil(0.1*len(negative)))

index = np.random.choice(len(positive),np.int(0.1*len(positive)), replace=False)
for i in index:
	positive[i] = -1*positive[i]

index = np.random.choice(len(negative), np.int(0.1*len(negative)), replace=False)
for i in index:
	negative[i] = -1*negative[i]

print(positive)
print(negative)

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

#################################################################################
# Mostramos el grafico
plt.plot(x[:, 0], y, c = 'black')
plt.scatter(x[:, 0], x[:, 1], c=colist)
plt.title('Ejercicio 2.1. Nube de puntos generada con ruido')
plt.xticks()
plt.yticks()
plt.xlabel('x')
plt.ylabel('y')
legend_elements = [mlines.Line2D([], [], color='black',markersize=15, label='Recta generada por simula_recta'),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='blue', label='-1', markersize=10),
                   mlines.Line2D([],[],linewidth=0,marker='o', color='red', label='1', markersize=10)]
plt.legend(handles=legend_elements)

plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01

    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0],
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)

    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2,
                cmap="RdYlBu", edgecolor='white')

    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')

    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]),
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()


#CODIGO DEL ESTUDIANTE

def f1(x):
	return (x[:,0]-10)**2 + (x[:,1]-2)**2 - 400

def f2(x):
	return 0.5*(x[:,0]+10)**2 + (x[:,1]-2)**2 - 400

def f3(x):
	return (x[:,0]-10)**2 - (x[:,1]+2)**2 - 400

def f4(x):
	return x[:,1] - 20*(x[:,0]**2) - 5*x[:,0] + 3

plot_datos_cuad(x,f_values,f1)

plot_datos_cuad(x,f_values,f2)

plot_datos_cuad(x,f_values,f3)

plot_datos_cuad(x,f_values,f4)

input("\n--- Pulsar tecla para continuar ---\n")
