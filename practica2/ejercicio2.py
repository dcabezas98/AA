# -*- coding: utf-8 -*-

# David Cabezas Berrido

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(2) # Para generar la misma muestra que en el ejercicio 1

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

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

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

# Pinta una muestra etiquetada junto con una recta
def pintarMuestraRecta(x, ax, label, w, title):

    # Pinto muestra etiquetada
    x1 = np.array([xi for i, xi in enumerate(x) if label[i]==1])
    plt.scatter(x1[:,1],x1[:,2],c='r',label='label=+1')
    x2 = np.array([xi for i, xi in enumerate(x) if label[i]==-1])
    plt.scatter(x2[:,1],x2[:,2],c='b',label='label=-1')
    # Pinto recta
    r=np.linspace(ax[0],ax[1],100)
    plt.plot(r, -(w[1]*r+w[0])/w[2], c='g', label='recta y='+('{0:+}'.format(-w[1]/w[2]))[:8]+'x'+('{0:+}'.format(-w[0]/w[2]))[:8])
    
    plt.legend()
    plt.title(title)
    plt.axis(ax)
    plt.show()

# ------------ MODELOS LINEALES ------------ #

print('MODELOS LINEALES\n')

# Ejercicio 1: Algoritmo Perceptron

print('Ejercicio 1\nAlgoritmo Perceptron\n')

# Genero la muestra
x=simula_unif(100, 2, (-50,50))
# Añado x_0=1
x=np.hstack((np.ones((len(x),1)),x)) # Le añado el 1 al vector de características

a, b = simula_recta([-50,+50])
# Función para etiquetar
def f(x,y): # Recta y=ax+b
    return y-a*x-b

# Etiquetas
label = etiquetaMuestraFun(x[:,1:],f)

# Muestra con ruido también la misma de la sección anterior
label_n = etiquetaMuestraFun(x[:,1:],f,True)

# Para comprobar que son correctas
# pintarMuestraRecta(x,[-50,50,-50,50], label, [-b,-a,1],'')
# pintarMuestraRecta(x,[-50,50,-50,50], label_n, [-b,-a,1],'')

# Algoritmo Perceptron
def PLA(datos, label, max_iter, vini):
    
    w = vini
    it = 0
    change=True
    while it < max_iter and change:
        change=False
        it+=1
        for i, xi in enumerate(datos):
            if np.sign(np.dot(w,xi))!=label[i]:
                w=w+label[i]*xi
                change = True
    return w, it

# a) Con datos separables

print('Datos separables:\n')

def  pruebaPLA(x, label):
    vini = np.zeros(x.shape[1]) # Partiendo del vector 0

    print('Partiendo del vector cero')
    it_media = 0
    print('Repitiendo 10 veces: \n ... \n')
    print('Iteraciones necesarias: ', end='')
    for _ in range(10):
        w, it = PLA(x,label,500,vini)
        print(str(it),end=', ')
        it_media+=it

    it_media/=10
    
    print()
    print('Número medio de iteraciones necesarias:', it_media)

    print('Solución obtenida en la última iteración:', w)

    pintarMuestraRecta(x, [-50,50,-50,50], label, w, 'Muestra con etiquetas y solución del PLA partiendo de w=(0,0,0)') # Pinto la solución obtenida

    input("\n--- Pulsar tecla para continuar ---\n")

    print('Partiendo de vectores aleatorios en [0,1]')
    it_media = 0
    print('Repitiendo 10 veces: \n ... \n')
    print('Iteraciones necesarias: ', end='')
    for _ in range(10):
        vini = np.random.uniform(0,1,x.shape[1]) # Partiendo de vectores aleatorios en [0,1]
        w, it = PLA(x,label,500,vini)
        print(str(it),end=', ')
        it_media+=it

    it_media/=10

    print()
    print('Número medio de iteraciones necesarias:', it_media)

pruebaPLA(x,label)

input("\n--- Pulsar tecla para continuar ---\n")


# b) Con datos no separables

print('Datos no separables:\n')

pruebaPLA(x, label_n)

input("\n--- Pulsar tecla para continuar ---\n")


# Ejercicio 2: regresión logística

print('Ejercicio 2\nRegresión Logística\n')

np.random.seed(9)

# Muestra de 100 puntos de [0,2]x[0,2]
x=simula_unif(100,2,[0,2])
# Añado x_0=1
x=np.hstack((np.ones((len(x),1)),x)) # Le añado el 1 al vector de características

# Los etiqueto con una recta aleatoria que pase por [0,2]x[0,2]
a, b = simula_recta([0,2])

def h(x,y): # Función etiquetadora
    return np.sign(y-a*x-b)

# Etiquetas
label = etiquetaMuestraFun(x[:,1:],h)

"""
# Función objetivo: toma valores 0 (para etiqueta = -1) y 1 (para etiqueta = +1), según a que lado de la recta esté el punto (x,y)
def f(x,y):
    return (h(x,y)+1)/2
"""

# Muestra etiquetada y recta que separa las clases
print('Muestra etiquetada y frontera')
pintarMuestraRecta(x,[0,2,0,2],label,np.array([-b,-a,1]),'Muestra etiquetada y frontera entre f(x)=1 y f(x)=0')

input("\n--- Pulsar tecla para continuar ---\n")

# a) Regresión Logística con SGD

# Función sigmoide
def sigma(t):
    return 1/(1+np.exp(-t))

# Error para x_n
def e_n(n, w, x, label):
    return np.log(1+np.exp(-label[n]*np.dot(w,x[n])))

# Gradiente para el error de x_n
def grad_e_n(n, w, x, label):
    return -label[n]*x[n]*sigma(-label[n]*np.dot(w,x[n]))

# Error medio de los x_n
def Error(w,x,label):
    e=0
    for n in range(len(x)):
        e+=e_n(n,w,x,label)
    return e/len(x)

# Regresión Logística con SGD
def rl_sgd(datos, label, lr, max_etapas, wini, diff):
    w=wini
    # Para que nunca se cumpla la condición ||w(t+1)-w(t)||<diff al principio
    w1=wini+np.ones(len(wini))*diff*2
    etapa=0
    while etapa<max_etapas and np.linalg.norm(w-w1) >= diff:
        w1=w # Para guardar el w de la etapa anterior
        # Barajo los datos y las etiquetas al unísono
        r_state=np.random.get_state()
        np.random.shuffle(datos)
        np.random.set_state(r_state)
        np.random.shuffle(label)
        # Itero sobre los N datos
        for n in range(len(datos)): 
            w=w-lr*grad_e_n(n,w,datos,label)
        etapa+=1  
    return w

print('Regresión Logística con Gradiente Descendente Estocástico')

w=np.zeros(x.shape[1]) # Parto de los pesos a 0
lr=0.01
max_etapas=5000 # Para asegurar que pare
diff=0.01

w = rl_sgd(x, label, lr, max_etapas, w, diff)
print('Solución: w=',w)
pintarMuestraRecta(x, [0,2,0,2], label, w, 'Solución obtenida mediante Regresión Logística con SGD')

print('Error en la muestra: Ein=',Error(w,x,label))

input("\n--- Pulsar tecla para continuar ---\n")

# b) Eout

# Muestra de 1500 puntos de [0,2]x[0,2] para test
test_data=simula_unif(1500,2,[0,2])
# Añado x_0=1
test_data=np.hstack((np.ones((len(test_data),1)),test_data)) # Le añado el 1 al vector de características

# Muestra de test
test_label=np.sign(h(test_data[:,1],test_data[:,2]))

print('Estimación del error fuera del conjunto de entrenamiento')
print('Eout=', Error(w,test_data, test_label))

input("\n--- Pulsar tecla para salir ---\n")