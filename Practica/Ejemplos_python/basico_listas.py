# -*- coding: utf-8 -*-

#Declarar tupla
tupla = (5, 't1', True, 0.5)

#Declarar lista
lista = [5, 't1', True, 0.5]

#Obtener tamaño lista y tubla
l_tupla = len(tupla)
l_lista = len(lista)

#Mostrar por pantalla
print(tupla)
print(lista)

#Acceder elemento
print(tupla[2])
print(lista[2])
lista[2] = 1000

#Añadir elemento
lista.append(False) #Al final
lista.insert(1, 't21') #En la posición

#Eliminar elemento
lista.remove('t1') #Buscando
lista.pop() #Al final
lista.pop(1) #En la posicion 1

#Concatenar
lista2 = ['a', 'b', 'c']
lista_combinada = lista + lista2 #Pega la lista2 al final de lista

#Copiar
lista_copia = lista.copy()

