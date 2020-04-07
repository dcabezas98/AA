# -*- coding: utf-8 -*-

# David Cabezas Berrido

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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


# Pinta la muestra con etiquetas por la función parámetro
# Junto con la función: (x,y) con f(x,y)=0
# Permite añadir ruido (para apartado b)
def pintarMuestraEtiquetadaFun(x, ax, fun, fun_str, noise=False):

    label = np.sign(f(x[:,0],x[:,1]))

    noise_str=' '
    if noise: # Para introducir ruido del 10% en cada clase
        noise_str=' con ruido '
        label1=np.array([i for i in range(len(label)) if label[i]==+1]) # Índices de las etiquetas positivas
        noisy1=np.random.randint(len(label1), size=int(len(label1)*0.1)) # Cojo el 10% de ellos
        for i in noisy1:
	        label[label1[i]]=-label[label1[i]] # Cambio el signo de las etiquetas correspondientes

        label2=np.array([i for i in range(len(label)) if label[i]==-1]) # Índices de las etiquetas negativas
        noisy2=np.random.randint(len(label2), size=int(len(label2)*0.1)) # Cojo el 10% de ellos
        for i in noisy2:
	        label[label2[i]]=-label[label2[i]] # Cambio el signo de las etiquetas correspondientes

    x1 = np.array([xi for i, xi in enumerate(x) if label[i]==1])
    sc1=plt.scatter(x1[:,0],x1[:,1],c='r',label='label=+1')
    x2 = np.array([xi for i, xi in enumerate(x) if label[i]==-1])
    sc2=plt.scatter(x2[:,0],x2[:,1],c='b',label='label=-1')

    x_range = np.arange(ax[0], ax[1], 0.05)
    y_range = np.arange(ax[2], ax[3], 0.05)
    X, Y = np.meshgrid(x_range, y_range)
    F=fun(X,Y)
    plt.contour(X,Y,F,[0],colors='g')
    line_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='g', marker = '_')
    
    plt.legend([sc1,sc2,line_proxy], [sc1.get_label(),sc2.get_label(), 'f(x,y)=0'], numpoints = 1)
    plt.title('Muestra con etiquetas'+noise_str+'\ny función etiquetadora: f(x,y)='+fun_str)
    plt.axis(ax)
    plt.show()

# -------- EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO -------- #

print('EJERCICIO SOBRE LA COMPLEJIDAD DE H Y EL RUIDO\n')

# 1 Generar nubes de puntos

print('Ejercicio 1')
print('Generación de nubes de puntos\n')

def pintarMuestra(x, ax, title): # Para pintar las muestras generadas
    plt.scatter(x[:,0],x[:,1])
    plt.axis(ax)
    plt.title(title)
    plt.show()

# a) N=50, dim=2, rango=[-50,+50] con simula_unif

print('a) Muestra uniforme en [-50,50]x[-50,50]')

x_a = simula_unif(50,2,(-50,50))

pintarMuestra(x_a, [-50,50,-50,50], 'Muestra uniforme')

input("\n--- Pulsar tecla para continuar ---\n")

# b) N=50, dim=2, sigma=[5,7] con simula_gaus

print('b) Muestra normal N( (0,0), (sqrt(5), sqrt(7)) )')

x_b = simula_gaus(50, 2, [5,7])

pintarMuestra(x_b, [-50, 50, -50, 50], 'Muestra normal')

input("\n--- Pulsar tecla para continuar ---\n")


# 2 Generar nube de puntos y etiquetarlos con una recta

print('Ejercicio 2\n')
print('Nube de puntos etiquetados con el signo de la distancia a una recta')

x=simula_unif(200, 2, (-50,50))

a, b = simula_recta([-50,+50])
# Función para etiquetar
def f(x,y): # Recta y=ax+b
    return y-a*x-b

# a) Dibujar gráfica

pintarMuestraEtiquetadaFun(x, [-50,50,-50,50], f, 'y'+('{0:+}'.format(-a))[:8]+'x'+('{0:+}'.format(-b))[:8])

input("\n--- Pulsar tecla para continuar ---\n")

# b) Introducir ruido en el 10% de las etiquetas de cada clase

pintarMuestraEtiquetadaFun(x, [-50,50,-50,50], f, 'y'+('{0:+}'.format(-a))[:8]+'x'+('{0:+}'.format(-b))[:8], True) # Dibujo nueva gráfica (con ruido)

# 3

input("\n--- Pulsar tecla para salir ---\n")