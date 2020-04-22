# -*- coding: utf-8 -*-

# David Cabezas Berrido

# Práctica 2: Bonus
# Clasificación de Dígitos

import numpy as np
import matplotlib.pyplot as plt

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 4 o la 8
	for i in range(0,datay.size):
		if datay[i] == 8 or datay[i] == 4:
			if datay[i] == 8:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Pinta una muestra etiquetada junto con una recta
def pintarMuestraRecta(x, label, w, title):

    # Pinto muestra etiquetada
    x1 = np.array([xi for i, xi in enumerate(x) if label[i]==1])
    plt.scatter(x1[:,1],x1[:,2],c='r',label='Dígito 8',alpha=0.6)
    x2 = np.array([xi for i, xi in enumerate(x) if label[i]==-1])
    plt.scatter(x2[:,1],x2[:,2],c='b',label='Dígito 4',alpha=0.4)
    # Pinto recta
    minx = np.min(x[:,1])
    maxx = np.max(x[:,1])
    miny = np.min(x[:,2])
    maxy = np.max(x[:,2])
    r=np.linspace(minx,maxx,100)
    plt.plot(r, -(w[1]*r+w[0])/w[2], c='g', label='recta y='+('{0:+}'.format(-w[1]/w[2]))[:8]+'x'+('{0:+}'.format(-w[0]/w[2]))[:8])

    plt.xlabel('Intensidad promedio de gris')
    plt.ylabel('Simetría respecto al eje vertical')
    plt.legend()
    plt.title(title)
    plt.axis([minx,maxx,miny,maxy])
    plt.show()

# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
	x_pinv=np.linalg.pinv(x)
	w=np.dot(x_pinv,y)
	return w

# Error de clasificación: proporción de elementos mal clasificados
def Eclass(w,x,y):
    err=0
    for i, xi in enumerate(x):
        err+=np.abs(np.sign(np.dot(w,xi))-y[i])/2
    return err/x.shape[0]

# Algoritmo PLA-Pocket
def PLA_Pocket(wini, x, y, etapas):
    w = wini
    pocket = wini
    err_pocket = Eclass(pocket, x, y)
    for _ in range(etapas):
        for i, xi in enumerate(x): # Hago una iteración del PLA
            if np.sign(np.dot(w,xi))!=y[i]:
                w=w+y[i]*xi
        err=Eclass(w,x,y) # Evalúo el error
        if err < err_pocket: # Si es mejor, actualizo
            pocket=w
            err_pocket=err
    return pocket


print('BONUS\nClasificación de Dígitos\n')

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy','datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy','datos/y_test.npy')

# Regresión Lineal (con pseudoinversa)
w = pseudoinverse(x, y)

print('Solución obtenida con Regresión Lineal (mediante pseudoinversa)\n')
print('w=',w)

pintarMuestraRecta(x, y, w, 'Datos de entrenamiento junto con solución obtenida\ncon Regresión Lineal mediante pseudoinversa')
pintarMuestraRecta(x_test, y_test, w, 'Datos de test junto con solución obtenida\ncon Regresión Lineal mediante pseudoinversa')

print('Errores obtenidos:')
print('E_in =',Eclass(w,x,y),'\t(N = '+str(len(x))+' datos)')
print('E_test =',Eclass(w,x_test,y_test),'\t(N = '+str(len(x_test))+' datos)')

input("\n--- Pulsar tecla para continuar ---\n")

# Mejoro la solución con PLA-Pocket
w = PLA_Pocket(w, x, y, 100)

print('Solución mejorada con PLA-Pocket\n')
print('w=',w)

pintarMuestraRecta(x, y, w, 'Datos de entrenamiento junto con solución obtenida\nmediante el algoritmo PLA-Pocket')

pintarMuestraRecta(x_test, y_test, w, 'Datos de test junto con solución obtenida\nmediante el algoritmo PLA-Pocket')

print('Errores obtenidos:')
print('E_in =',Eclass(w,x,y),'\t(N = '+str(len(x))+' datos)')
print('E_test =',Eclass(w,x_test,y_test),'\t(N = '+str(len(x_test))+' datos)')

input("\n--- Pulsar tecla para salir ---\n")