# Práctica 1, Ejercicio 1
# David Cabezas Berrido

import numpy as np
import sympy as sp

u, v = sp.symbols('u v')

# Función de error (a minimizar)
E_sym=sp.Lambda((u,v), sp.simplify((u*sp.exp(v)-2*v*sp.exp(-u))**2))
def E(w):
    return float(E_sym(w[0],w[1]))

# Derivada parcial de E respecto de u
Eu_sym=sp.Lambda((u,v), sp.simplify(sp.diff(E_sym(u,v),u)))
def Eu(w):
	return float(Eu_sym(w[0],w[1]))

# Derivada parcial de E respecto de v
Ev_sym=sp.Lambda((u,v), sp.simplify(sp.diff(E_sym(u,v),v)))
def Ev(w):
	return float(Ev_sym(w[0],w[1]))
	
# Gradiente de E
def gradE(w):
	return np.array((Eu(w), Ev(w)))

def gd(w, lr, grad_fun, fun, epsilon, max_iters = 10000):

	it = 0

	while it < max_iters and fun(w)>epsilon:
		it+=1
		grad=grad_fun(w)
		w=w-lr*grad#/np.linalg.norm(grad)
		#print(w, fun(w))

	return w, it

# Punto inicial
w=np.array([1,1])
# Número máximo de iteraciones
max_iters=5000
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