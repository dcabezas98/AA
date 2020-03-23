# -*- coding: utf-8 -*-

# Práctica 1, Ejercicio 1
# David Cabezas Berrido

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ---------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 ------------------------------------#

u, v = sp.symbols('u v')

# Función a minimizar
E_sym=sp.Lambda((u,v), sp.simplify((u*sp.exp(v)-2*v*sp.exp(-u))**2))
def E(w):
    return sp.N(E_sym(w[0],w[1]))

# Derivada parcial de E respecto de u
Eu_sym=sp.Lambda((u,v), sp.simplify(sp.diff(E_sym(u,v),u)))
def Eu(w):
	return sp.N(Eu_sym(w[0],w[1]))

# Derivada parcial de E respecto de v
Ev_sym=sp.Lambda((u,v), sp.simplify(sp.diff(E_sym(u,v),v)))
def Ev(w):
	return sp.N(Ev_sym(w[0],w[1]))
	
# Gradiente de E
def gradE(w):
	return np.array((Eu(w), Ev(w)),np.float64)

# Algoritmo del gradiente descendente
def gd(w, lr, grad_fun, fun, epsilon, max_iters = 1000):

	it = 0

	while it < max_iters and fun(w)>epsilon:
		it+=1
		grad=grad_fun(w) # Calcula gradiente
		w=w-lr*grad      # Actualiza el punto
		# print(w, fun(w))

	return w, it

# Punto inicial
w=np.array([1,1],np.float64)
# Número máximo de iteraciones
max_iters=100
# Learning rate
lr=0.1
# Margen de error
epsilon=10**(-14)

w, num_ite = gd(w,lr,gradE,E,epsilon,max_iters)

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar Intro para continuar ---\n")
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar Intro para continuar ---\n")


#----------------------------- Ejercicio 2 --------------------------------#

x, y = sp.symbols('x y')

# Función a minimizar
f_sym=sp.Lambda((x,y), sp.simplify((x-2)**2+2*(y+2)**2+2*sp.sin(2*sp.pi*x)*sp.sin(2*sp.pi*y)))
def f(w):
	return float(f_sym(w[0],w[1]))
	
# Derivada parcial de f respecto de x
fx_sym=sp.Lambda((x,y), sp.simplify(sp.diff(f_sym(x,y),x)))
def fx(w):
	return float(fx_sym(w[0],w[1]))

# Derivada parcial de f respecto de y
fy_sym=sp.Lambda((x,y), sp.simplify(sp.diff(f_sym(x,y),y)))
def fy(w):
	return float(fy_sym(w[0],w[1]))
	
# Gradiente de f
def gradf(w):
	return np.array((fx(w), fy(w)),np.float64)

# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,-1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1

def gd_grafica(w, lr, grad_fun, fun, max_iters = 1000):

	graf = []
	graf.append(fun(w))
	
	for _ in range(max_iters):
		grad=grad_fun(w)
		w=w-lr*grad
		graf.append(fun(w))
		#print(w, fun(w))

	graf = np.array(graf,np.float64)
	
	plt.plot(range(0,max_iters+1), graf, 'bo')
	plt.xlabel('Iteraciones')
	plt.ylabel('f(x,y)')
	plt.show()

# Punto inicial
w=np.array([1,-1],np.float64)
# Número máximo de iteraciones
max_iters=50

print ('Resultados ejercicio 2\n')
print ('\nGrafica con learning rate igual a 0.01')
lr=0.01 # Learning rate
gd_grafica(w,lr,gradf,f,max_iters)
print ('\nGrafica con learning rate igual a 0.1')
lr=0.1 # Learning rate
gd_grafica(w,lr,gradf,f,max_iters)
input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

def gd(w, lr, grad_fun, fun, max_iters = 1000):
	
	for _ in range(max_iters):
		grad=grad_fun(w)
		w=w-lr*grad
		#print(w, fun(w))
	
	return w

lr=0.01 # Learning rate (no se especifica)
# Número máximo de iteraciones (tampoco se especifica)
max_iters=50

print ('Punto de inicio: (2.1, -2.1)\n')
w=gd(np.array((2.1,-2.1),np.float64),lr,gradf,f,max_iters)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')
w=gd(np.array((3.0,-3.0),np.float64),lr,gradf,f,max_iters)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')
w=gd(np.array((1.5,1.5),np.float64),lr,gradf,f,max_iters)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')
w=gd(np.array((1.0,-1.0),np.float64),lr,gradf,f,max_iters)
print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para salir ---\n")