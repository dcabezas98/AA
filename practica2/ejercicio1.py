# -*- coding: utf-8 -*-

# David Cabezas Berrido

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib


# Fijamos la semilla, la volveré a fijar en algunas ocasiones
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


# Etiqueta una muestra usando el signo de una función f
# A cada (x,y) le asigna el signo de f(x,y)
# Permite añadir ruido del 10%
def etiquetaMuestraFun(x, fun, noise=False):
    
    label = np.sign(fun(x[:,0],x[:,1])) # Etiqueta la muestra

    if noise: # Para introducir ruido del 10% en cada clase
        label1=np.array([i for i in range(len(label)) if label[i]==+1]) # Índices de las etiquetas positivas
        noisy1=np.random.randint(len(label1), size=int(np.round(len(label1)*0.1))) # Cojo el 10% de ellos
        label2=np.array([i for i in range(len(label)) if label[i]==-1]) # Índices de las etiquetas negativas
        noisy2=np.random.randint(len(label2), size=int(np.round(len(label2)*0.1))) # Cojo el 10% de ellos
        for i in noisy1:
	        label[label1[i]]=-label[label1[i]] # Cambio el signo de las etiquetas positivas
        for i in noisy2:
	        label[label2[i]]=-label[label2[i]] # Cambio el signo de las etiquetas negativas

    return label

# Pinta la muestra etiquetada
# Junto con la función: (x,y) con f(x,y)=0
def pintarMuestraEtiquetadaFun(x, label, ax, fun, title):

    # Pinto la función y las regiones que divide
    cm = ListedColormap(['skyblue','lightsalmon'])

    x_range = np.arange(ax[0], ax[1], 0.05)
    y_range = np.arange(ax[2], ax[3], 0.05)
    X, Y = np.meshgrid(x_range, y_range)
    F=fun(X,Y)
    plt.contour(X,Y,F,[0],colors='g')
    plt.contourf(X,Y,F,0,cmap=cm,alpha=0.4)

    # Pinto muestra etiquetada
    x1 = np.array([xi for i, xi in enumerate(x) if label[i]==1])
    sc1=plt.scatter(x1[:,0],x1[:,1],c='r',label='label=+1',alpha=0.75)
    x2 = np.array([xi for i, xi in enumerate(x) if label[i]==-1])
    sc2=plt.scatter(x2[:,0],x2[:,1],c='b',label='label=-1',alpha=0.75)

    # Para la leyenda
    line_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='g', marker = '_')
    pos_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='lightsalmon', marker = 's')
    neg_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='skyblue', marker = 's')
    
    plt.legend([sc1,sc2,line_proxy, pos_proxy, neg_proxy], [sc1.get_label(),sc2.get_label(), 'f(x,y)=0','f(x,y)>0','f(x,y)<0'], numpoints = 1,framealpha=0.5)
    plt.title(title)
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

# Vuelvo a fijar la semilla puesto que tengo que generar la misma muestra en el la sección 2
np.random.seed(2) 
x=simula_unif(100, 2, (-50,50))

a, b = simula_recta([-50,+50])
# Función para etiquetar
def f(x,y): # Recta y=ax+b
    return y-a*x-b

# a) Dibujar gráfica

label=etiquetaMuestraFun(x, f) # Asigno etiquetas

fun_str='y'+('{0:+}'.format(-a))[:8]+'x'+('{0:+}'.format(-b))[:8]

# Dibujo la muestra junto a la recta
pintarMuestraEtiquetadaFun(x, label, [-50,50,-50,50], f, 'Muestra con etiquetas\ny función etiquetadora: f(x,y)='+fun_str)

input("\n--- Pulsar tecla para continuar ---\n")

# b) Introducir ruido en el 10% de las etiquetas de cada clase
print('Ahora con ruido\n')

label_n=etiquetaMuestraFun(x, f, True) # Asigno etiquetas con ruido

fun_str='y'+('{0:+}'.format(-a))[:8]+'x'+('{0:+}'.format(-b))[:8]

fails=((label_n-label)!=0).sum() # Mido los fallos que comete (rondarán el 10%)
print('Función: f(x,y)='+fun_str)
print('Proporción de puntos mal clasificados:',fails/len(x))

# Dibujo la muestra junto a la recta
pintarMuestraEtiquetadaFun(x, label_n, [-50,50,-50,50], f, 'Muestra con etiquetas con ruido\ny función etiquetadora: f(x,y)='+fun_str)

input("\n--- Pulsar tecla para continuar ---\n")

# 3 Muestra anterior junto con otras funciones

print('Ejercicio 3\n')
print('Muestra anterior junto con otras funciones más complejas\n')

# Funciones que voy a probar
def f1(x,y):
    return (x-10)**2+(y-20)**2-400
def f2(x,y):
    return 0.5*(x+10)**2+(y-20)**2-400
def f3(x,y):
    return 0.5*(x-10)**2-(y+20)**2-400
def f4(x,y):
    return y-20*x**2-5*x+3

fun=[f1,f2,f3,f4]
fun_str=['(x-10)^2+(y-20)^2-400','0.5(x+10)^2+(y-20)^2-400','0.5(x-10)^2-(y+20)^2-400','y-20x^2-5x+3']

for i in range(len(fun)):
    labelf=etiquetaMuestraFun(x,fun[i]) # Mido los fallos de cada una
    fails=((label_n-labelf)!=0).sum()
    print('Función: f(x,y)='+fun_str[i])
    print('Proporción de puntos mal clasificados:',fails/len(x))
    pintarMuestraEtiquetadaFun(x, label_n, [-50,50,-50,50], fun[i], 'Muestra con etiquetas con ruido\ny función: f(x,y)='+fun_str[i])
    input("\n--- Pulsar tecla para continuar ---\n")


input("\n--- Pulsar tecla para salir ---\n")