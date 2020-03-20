# -*- coding: utf-8 -*-

# Práctica 1, Ejercicio 1
# David Cabezas Berrido

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D # Para dibujar en 3D

#------------------------------------------------------------------------------#
#------------- Ejercicio sobre regresión lineal ----------------#
#------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 ------------------------------------#

# Función para pintar los planos obtenidos
def Pintar_plano(w, x, y, title):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d') # Ejes
	ax.scatter(x[:,1],x[:,2],y,color='r') # Datos de entrenamiento

	minx = np.min(x[:,1])
	maxx = np.max(x[:,1])
	miny = np.min(x[:,2])
	maxy = np.max(x[:,2])
	xx, yy = np.meshgrid(np.linspace(minx,maxx,10),np.linspace(miny,maxy,10))
	z = np.array(w[0]+xx*w[1]+yy*w[2])

	ax.plot_surface(xx,yy, z, color='b', alpha=0.4) # Plano

	ax.set_xlabel('\nIntensidad promedio de gris')
	ax.set_ylabel('\nSimetría respecto \n al eje vertical')
	ax.set_zlabel('\nEtiqueta \n (-1= dígito 1, 1= dígito 5)')

	# Hago un plot 2D sin nada, para poder poner leyenda
	scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
	plane_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker = 's')
	ax.legend([scatter1_proxy, plane_proxy], ['Train', 'Solución'], numpoints = 1, loc='upper right')

	plt.title(title,loc='left')
	plt.show()

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
	err = 0
	for i, xi in enumerate(x):
		err+=(np.inner(w,xi)-y[i])**2

	return err/x.shape[0]

# Función para calcular el gradiente atendiendo a una parte de la muestra (argumento x,y), sin el 2/M 
def gradErr(x,y,w):
	g=np.zeros(w.shape[0])
	for n, xn in enumerate(x):
		g+=xn*(np.inner(w,xn)-y[n])
	return g

# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):
	
	size=x.shape[0]
	w=np.zeros(x.shape[1])
	indices = np.arange(size)

	for _ in range(max_iters):
		# Primero barajo, para hacer los minibatches de manera aleatoria		
		np.random.shuffle(indices)
		x1 = x[indices[0:tam_minibatch]]
		y1= y[indices[0:tam_minibatch]]
		w=w-lr*gradErr(x1,y1,w)

	return w
	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
	x_pinv=np.linalg.pinv(x)
	w=np.dot(x_pinv,y)
	return w
	
"""
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy','datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy','datos/y_test.npy')


print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')

lr=0.001
max_iters=1000
tam_minibatch=32

# Gradiente descendente estocastico
w = sgd(x,y,lr,max_iters,tam_minibatch)

# Pinto la solución:
Pintar_plano(w,x,y,'Solución obtenida con gradiente descendente estocástico')

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test,w))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa
w = pseudoinverse(x, y)

# Pinto la solución:
Pintar_plano(w,x,y,'Solución obtenida con pseudoinversa')

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")
"""

#------------------------------Ejercicio 2 ------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size] (en caso bidimensional, d=2)
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,[N,d]) # A 2x3 array
	
# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	

print ('Ejercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')
x=simula_unif(1000,2,1)

plt.scatter(x[:,0],x[:,1]) # Pintar puntos
plt.show()

# b) Asignar etiquetas por f(x1,x2)=sign((x1-0.2)²+x2²-0.6)
def f(x1,x2):
	return np.sign((x1-0.2)**2+x2**2-0.6)

y = np.array(f(x[:,0],x[:,1]))

# Introducir ruido:
noisy=np.random.randint(len(y), size=int(len(y)*0.1))
for i in noisy:
	y[i]=-y[i]

# Pintar etiquetas
a = np.array([xi for i, xi in enumerate(x) if y[i]==1])
plt.scatter(a[:,0],a[:,1],c='b',label='y=1')
b = np.array([xi for i, xi in enumerate(x) if y[i]==-1])
plt.scatter(b[:,0],b[:,1],c='r',label='y=-1')
plt.legend(title='Etiqueta:', loc='upper left')
plt.show()

# c) Ajustar por regresión lineal

x=np.array([[1,xi[0],xi[1]] for xi in x])

lr = 0.001
max_iters=1000
tam_minibatch=32

w = sgd(x,y,lr,max_iters,tam_minibatch)

# Pintar la solución obtenida
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # Ejes
ax.scatter(x[:,1],x[:,2],y,color='r') # Datos de entrenamiento

xx, yy = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
z = np.array(w[0]+xx*w[1]+yy*w[2])

ax.plot_surface(xx,yy, z, color='b', alpha=0.4) # Plano

# Hago un plot 2D sin nada, para poder poner leyenda
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
plane_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker = 's')
ax.legend([scatter1_proxy, plane_proxy], ['Train', 'Solución'], numpoints = 1, loc='upper right')

plt.title('Ajuste con características lineales',loc='left')
plt.show()

# d) Ejecutar el experimento 1000 veces

def generate_data():
	x=simula_unif(1000,2,1)
	y = np.array(f(x[:,0],x[:,1]))
	noisy=np.random.randint(len(y), size=int(len(y)*0.1))
	for i in noisy:
		y[i]=-y[i]

	x=np.array([[1,xi[0],xi[1]] for xi in x])

	return x,y

Ein_media = 0
Eout_media = 0

for _ in range(1000):
	# Train
	x,y=generate_data()

	w = sgd(x,y,lr,max_iters,tam_minibatch)

	Ein_media+=Err(x,y,w)

	# Test
	x,y=generate_data()

	Eout_media+=Err(x,y,w)

Ein_media/=1000
Eout_media/=1000

print('Errores Ein y Eout medios tras 1000reps del experimento con características lineales:\n')
print("Ein media: ", Ein_media)
print("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")

# Experimento utilizando características cuadráticas

def generate_data2(): # Nuevos datos, incluye las características cuadráticas

	x, y = generate_data()
	
	x_square=np.array(np.square(x))
	x_product=np.expand_dims(np.prod(np.array(x),axis=1),axis=1)

	x=np.hstack((x,x_product,x_square[:,1:]))

	return x,y

x, y = generate_data2()

w = sgd(x,y,lr,max_iters,tam_minibatch)

# Pintar la solución obtenida
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # Ejes
ax.scatter(x[:,1],x[:,2],y,color='r') # Datos de entrenamiento

xx, yy = np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
z = np.array(w[0]+xx*w[1]+yy*w[2]+np.prod((xx,yy),axis=0)*w[3]+np.square(xx)*w[4]+np.square(yy)*w[5])

ax.plot_surface(xx,yy, z, color='b', alpha=0.4) # Cuádrica

# Hago un plot 2D sin nada, para poder poner leyenda
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='r', marker = 'o')
plane_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c='b', marker = 's')
ax.legend([scatter1_proxy, plane_proxy], ['Train', 'Solución'], numpoints = 1, loc='upper right')

plt.title('Ajuste con características cuadráticas',loc='left')
plt.show()

# Repito 1000 veces el experimento

Ein_media = 0
Eout_media = 0

for _ in range(1000):
	# Train
	x,y=generate_data2()

	w = sgd(x,y,lr,max_iters,tam_minibatch)

	Ein_media+=Err(x,y,w)

	# Test
	x,y=generate_data2()

	Eout_media+=Err(x,y,w)

Ein_media/=1000
Eout_media/=1000

print('Errores Ein y Eout medios tras 1000reps del experimento con características no lineales:\n')
print("Ein media: ", Ein_media)
print("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para salir ---\n")