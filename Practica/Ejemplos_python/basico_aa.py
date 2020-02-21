# -*- coding: utf-8 -*-

#Declaración var y asignación
entero1 = 5 #Int
entero2 = 505 #Int
flotante = 50.5 #Float
boolean_t = True #Boolean
boolean_f = False #Boolean
string1 = 'String1' #String
string2 = 'String2' #String

#Operaciones aritméticas básicas
suma = entero1 + flotante
resta = entero1 - flotante
producto = entero1 * flotante
division = entero1 / flotante
division_entera = entero2 // entero1
resto = entero2 % entero1

#Operaciones lógicas
igual = entero2==(500 + 5)
no_igual = entero1 != suma
mayor = entero2 > entero1 # >=
menor = entero1 < entero2 # <=
and_logico = igual and mayor
or_logico = igual or no_igual

#Cambiar tipos
entero2flotante = float(entero1)
flotante2entero = int(flotante)
astring = str(entero2)
abool = bool(entero1)

#Strings
formatear = 'String con entero %d, flotante %f y string %s' % (entero1, 
                                                               flotante, 
                                                               string1)
concatenar = string1 + str(entero1)

#Mostrar por pantalla
print('Dos de los strings:', string1, string2)
print('String y entero:', concatenar, entero1)

