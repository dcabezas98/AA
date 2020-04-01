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


# 2 Generar nube de puntos y etiquetarlos con una recta

print('Ejercicio 2\n')
print('Nube de puntos etiquetados con el signo de la distancia a una recta')

x = x_a

a, b = simula_recta([-50,+50])

# Función para etiquetar
def f(x,y):
    return y-a*x-b

label = np.sign(f(x[:,0],x[:,1]))

# a) Dibujar gráfica

x1 = np.array([xi for i, xi in enumerate(x) if label[i]==1])
plt.scatter(x1[:,0],x1[:,1],c='b',label='label=+1')
x2 = np.array([xi for i, xi in enumerate(x) if label[i]==-1])
plt.scatter(x2[:,0],x2[:,1],c='r',label='label=-1') # Muestra etiquetada
r = np.linspace(-50, 50, 50)
plt.plot(r, a*r+b, c='g', label='y='+str(a)[:6]+'x+'+str(b)[:6]) # Recta
plt.legend(loc='lower right')
plt.title('Muestra con etiquetas y recta')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# b) Introducir ruido en el 10% de las etiquetas

noisy=np.random.randint(len(label), size=int(len(label)*0.1))
for i in noisy:
	label[i]=-label[i]

# Dibujo nueva gráfica (con ruido)

print("Muestra con ruido y recta")

x1 = np.array([xi for i, xi in enumerate(x) if label[i]==1])
plt.scatter(x1[:,0],x1[:,1],c='b',label='label=+1')
x2 = np.array([xi for i, xi in enumerate(x) if label[i]==-1])
plt.scatter(x2[:,0],x2[:,1],c='r',label='label=-1') # Muestra etiquetada
r = np.linspace(-50, 50, 50)
plt.plot(r, a*r+b, c='g', label='y='+str(a)[:6]+'x+'+str(b)[:6]) # Recta
plt.legend(loc='lower right')
plt.title('Muestra con etiquetas (con ruido) y recta')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

input("\n--- Pulsar tecla para salir ---\n")