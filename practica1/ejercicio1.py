# Práctica 1, Ejercicio 1
# David Cabezas Berrido

import numpy as np
import sympy as sp

u, v = sp.symbols('u v')

# Función de error (a minimizar)
E_sym=sp.Lambda((u,v), (u*sp.exp(v)-2*v*sp.exp(-u))**2)
def E(w):
    return E_sym(w[0],w[1])

# Derivada parcial de E respecto de u
Eu_sym=sp.Lambda((u,v), diff(E_sym(u,v),u))
def Eu(w):
	return Eu_sym(w[0],w[1])

# Derivada parcial de E respecto de v
Ev_sym=sp.Lambda((u,v), diff(E_sym(u,v),v))
def Ev(w):
	return Ev_sym(w[0],w[1])
	
# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

def gd(w, lr, grad_fun, fun, epsilon, max_iters = ):		
	return w, it

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')
print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar Intro para continuar ---\n")
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar Intro para continuar ---\n")