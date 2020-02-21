# -*- coding: utf-8 -*-


def funcion(a, b):
    c = a+b
    
    return c #Opcional


a = 1
b = 2
c = funcion(a, b)


class Clase():
    def __init__(self, a):
        self.a = a
    
    def llamar(self, b):
        return self.a*b


class Clase2(Clase):
    def __init__(self, a, b):
        super().__init__(a)
        self.b = b
    
    def llamar(self, c):
        return self.a*self.b*c
    
    def __call__(self, c):
        return self.llamar(c)


a = 1
b = 2
c = 3
clase2 = Clase2(a, b)
d = clase2.llamar(c) #O clase2(c)

