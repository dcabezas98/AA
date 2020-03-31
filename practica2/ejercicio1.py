# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)


def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
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


# -------- EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO -------- #

print('EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO\n')

# 1 Generar nubes de puntos

print('Ejercicio 1')
print('Generación de nubes de puntos\n')

# a) N=50, dim=2, rango=[-50,+50] con simula_unif

print('a) Muestra uniforme en [-50,50]x[-50,50]')

x_a = simula_unif(50,2,(-50,50))

plt.scatter(x_a[:,0],x_a[:,1]) # Pintar puntos
plt.title('Muestra uniforme')
#plt.axis([-50, 50, -50, 50])
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# b) N=50, dim=2, sigma=[5,7] con simula_gaus

print('b) Muestra normal N( (0,0), (sqrt(5), sqrt(7)) )')

x_b = simula_gaus(50, 2, [5,7])

plt.scatter(x_b[:,0],x_b[:,1]) # Pintar puntos
plt.title('Muestra normal')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")



input("\n--- Pulsar tecla para salir ---\n")