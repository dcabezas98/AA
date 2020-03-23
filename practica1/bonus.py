# -*- coding: utf-8 -*-

# Práctica 1, Ejercicio BONUS
# David Cabezas Berrido

# Método de Newton para minimizar funciones

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

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

# Derivadas parciales de segundo orden:
fxx_sym=sp.Lambda((x,y), sp.simplify(sp.diff(fx_sym(x,y),x)))
def fxx(w):
	return float(fxx_sym(w[0],w[1]))

fxy_sym=sp.Lambda((x,y), sp.simplify(sp.diff(fx_sym(x,y),y)))
def fxy(w):
	return float(fxy_sym(w[0],w[1]))

fyy_sym=sp.Lambda((x,y), sp.simplify(sp.diff(fy_sym(x,y),y)))
def fyy(w):
	return float(fyy_sym(w[0],w[1]))

# Matriz Hessiana
def hessf(w):
	a=fxy(w) # f es de clase 2, así que fxy = fyx
	return np.array([[fxx(w),a],[a,fyy(w)]])


# Método de Newton para encontrar un 0 en la derivada
def newton(w, grad_fun, hess_fun, fun, max_iters=500):

	graf = []
	graf.append(fun(w))

	for _ in range(max_iters):

		H1=np.linalg.inv(hess_fun(w))
		w = w - np.dot(H1,grad_fun(w))
		graf.append(fun(w))

	return w, graf


# Gradiente descendiente (para compararlo)
def gd(w, lr, grad_fun, fun, max_iters = 1000):

	graf = []
	graf.append(fun(w))
	
	for _ in range(max_iters):
		grad=grad_fun(w)
		w=w-lr*grad
		graf.append(fun(w))
		#print(w, fun(w))

	graf = np.array(graf,np.float64)

	return w, graf


print('Comparación del método de Newton con el gradiente descendente:\n')

# Número de iteraciones
max_iters=20
# Tasa de aprendizaje para el gradiente descendente
lr=0.01
# Puntos de inicio
condiciones_iniciales=np.array([(2.1, -2.1),(3.0, -3.0),(1.5, 1.5),(1.0, -1.0)])

# Experimentos con resultados (no he puesto pausas porque se pausa sólo hasta que se cierre la gráfica que genera, se puede descomentar la pausa)
for w in condiciones_iniciales:
    wn, grafn = newton(w, gradf, hessf, f, max_iters)
    wg, grafg = gd(w, lr, gradf, f, max_iters)

    # Resultados obtenidos
    print('Punto de inicio:', w)
    print('Solución y valor del método de Newton:')
    print ('(x,y) = (', wn[0], ', ', wn[1],')')
    print ('f(x,y) = ',f(wn))
    print('Solución y valor del gradiente estocástico:')
    print ('(x,y) = (', wg[0], ', ', wg[1],')')
    print ('f(x,y) = ',f(wg))
    print()

    #input("\n--- Pulsar tecla para continuar ---\n")

    # Curva de decrecimiento de la función
    plt.plot(range(0,max_iters+1), grafn, 'bo', alpha=0.6, label='Newton')
    plt.plot(range(0,max_iters+1), grafg, 'ro', alpha=0.4, label='Grad. Desc.')
    plt.xlabel('Iteraciones')
    plt.ylabel('f(x,y)')
    plt.title('Punto de inicio: w = '+str(w))
    plt.legend()
    plt.show()

input("\n--- Pulsar tecla para salir ---\n")