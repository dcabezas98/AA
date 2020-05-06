# Clasificación de Dígitos
# David Cabezas Berrido

# Variables ajustables:

NSAMPLES=2 # Vara visualizar algunos datos

VISUALIZE2D=True # Para la visualización de los datos en 2D (tarda un poco)

VARTHRESHOLD=0.005 # Umbral de varianza por debajo del cual elimino la característica
POLY=2 # Grado de las características polinomiales (poner 1 o 2)
VARPCA=0.975 # Porcentaje de variabilidad de la distribución que deben explicar las características que no elimine

# Parámetros por defecto del modelo
LMBD=0.007391304347826088 # Penalización para regularización
LR=0.001 # Tasa de aprendizaje del SGD para SoftMax
MINIBATCH_SIZE=1 # Tamaño de minibatch
TOTAL_ITERS=50000 # (TOTAL_ITERS/minibatch_size es el número real de iteraciones)

PARAMSELECT=False # Para el seleccionador de parámetros (tarda varias horas)
LMBD_RANGE=[0.001, 0.05, 24] # Valores para lambda (inicio, fin, valores)
LR_RANGE=[0.001, 0.02, 16] # Valores para lr (inicio, fin, valores)
MINIBATCH_SIZE_RANGE=[1,4,8,16,32] # Posibles valores para el tamaño de minibatch

V_FOLD = 10 # Subdivisiones para Cross-Validation

PRUEBAS = 0  # Número de pruebas detalladas del modelo sobre el conjunto de test

# Rutas a los ficheros de datos
TRAIN='datos/optdigits.tra'
TEST='datos/optdigits.tes'

# Librerías externas

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.lines import Line2D

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection


# Fijo la semilla
np.random.seed(25)


# Función para leer los datos
def readData(data_file):

    x=[]    # Características
    y=[]    # Etiquetas
    
    f=open(data_file)

    for line in f:
        d=list(map(int,line[:-1].split(',')))
        x.append(d[:-1])
        y.append(d[-1])

    x=np.array(x,np.int8)
    y=np.array(y,np.int8)

    return x, y

# Función para almacenar las características como una matriz de 8x8 en lugar de como un vector de 64
def matrixData(x): 

    x_matrix = np.reshape(x,(np.shape(x)[0], 8, 8))

    return x_matrix

# Función para almacenar las etiquetas como 1-hot-vectors
def oneHotLabel(y):

    y=np.reshape(y,(len(y),1))
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False, dtype=np.int8)
    onehot_encoded = onehot_encoder.fit_transform(y)
    
    return onehot_encoded

# Visualizar en 2D
def visualize2D(x,y, alg_name):

    plt.scatter(x[:,0],x[:,1],c=y, cmap='tab10', alpha=0.4)

    cmap=cm.get_cmap('tab10')

    proxys=[]
    labels=[]
    for l in range(10):
        proxys.append(Line2D([0],[0], linestyle='none', c=cmap(l/10), marker='o'))
        labels.append(str(l))

    plt.legend(proxys, labels, numpoints = 1,framealpha=0.5)
    plt.title('Representación de los dígitos en dos dimensiones\n usando el algoritmo '+alg_name)
    plt.show()

# Para visualizar matrices
def visualizeMatrix(m, title='', conf=False):

    plt.matshow(m, cmap='viridis')
    plt.colorbar()
    plt.title(title,pad=20.0)
    if conf:
        plt.ylabel('Verdaderos')
        plt.xlabel('Predicciones')
    plt.show()

# Función sigmoide
def sigm(x):
    return 1/(1+np.exp(-x))

# Función de pérdida logarítmica para clasificación multietiqueta por SoftMax (w es una matriz)
# permite añadir regularización
def E(w, x, y, lmbd = 0):

    N = np.shape(x)[0]
    K = np.shape(y)[1]

    e=0

    for n in range(N):
        for k in range(K):
            t = sigm(np.dot(w[k],x[n]))
            e -= y[n,k]*np.log(t)

    return e/N + lmbd*np.linalg.norm(w)**2/N # E + penalización reg

# Gradiente respecto al vector wj
# permite añadir regularización
def gradE(j,w,x,y, lmbd = 0):

    N = np.shape(x)[0]
    ge = np.zeros(np.shape(x)[1])

    for n in range(N):
        t = sigm(np.dot(w[j],x[n]))
        ge = (t-y[n,j])*x[n]

    return ge + lmbd*w[j] # Gradiente Ein + penalización reg

# Accuracy y matriz de confusión
def classificationScore(y, pred, set_name):

    # Precisión de la clasificación
    accuracy=metrics.accuracy_score(y, pred)
    print('Precisión sobre ' +set_name+':', accuracy)

    # Matriz de confusión:
    print('\nMatriz de confusión sobre ' +set_name)
    conf_mat=metrics.confusion_matrix(y,pred, normalize='all')
    visualizeMatrix(conf_mat, title='Matriz de confusión en el conjunto de '+set_name, conf=True)

    return accuracy


# Clasificador: realiza Regresión Logística multietiqueta
# y predice con SoftMax
class ClassifierLR(BaseEstimator):

    def __init__(self, lr, minibatch_size=MINIBATCH_SIZE, lmbd=0, wini=None):

        # Parámetros
        self.lr=lr
        self.minibatch_size=minibatch_size
        self.lmbd=lmbd # Para regularización
        self.wini=wini # Pesos iniciales

    # Gradiente Descendente Estocástico para entrenar el modelo
    # permite añadir regularización
    def fit(self, x, y):

        # Para el algoritmo necesito las etiquetas codificadas de esta forma
        y=oneHotLabel(y)

        if self.wini is None: # Valor inicial
            w = np.zeros((np.shape(y)[1],np.shape(x)[1]))
        else:
            w = self.wini

        # Iteraciones (cada una sobre un minibatch)
        iters = int(TOTAL_ITERS/self.minibatch_size)

        for _ in range(iters):
        
            # Selecciono minibatch
            minib = np.random.randint(0, len(x), self.minibatch_size) 

            # Calculo gradiente de E respecto a los distintos wj
            grad = np.array([gradE(j, w, x[minib], y[minib],self.lmbd) for j in range(np.shape(y)[1])])

            w = w - self.lr*grad

        self.w_ = w

    # Predicción para una entrada: SoftMax
    def _meaning(self, x):

        p=np.zeros(10)

        for k in range(10):
            p[k]=np.exp(np.dot(self.w_[k],x))

        return np.argmax(p) # Clase con más probabilidad

    # Predicción para un conjunto de entradas
    def predict(self, x):

        predictions = np.zeros(np.shape(x)[0])
        for n in range(np.shape(x)[0]): # Calculo la predicción para cada elemento
            predictions[n] = self._meaning(x[n])

        return predictions 

    # Perdida logarítmica sobre un conjunto de datos
    # incluye regularización
    def logLoss(self, x, y):
        y_1hot=oneHotLabel(y)
        return E(self.w_,x,y_1hot, self.lmbd)

    # Score: accuacy
    def score(self, x, y):
        return metrics.accuracy_score(y, self.predict(x))
      

# Main
if __name__ == "__main__":

    print('\nEjercicio de clasificación:')
    print('Reconocimiento de dígitos manuscritos\n')

    x, y = readData(TRAIN) # Lee datos de entrenamiento
    x_test, y_test = readData(TEST) # Lee datos de test

    x_matrix = matrixData(x)
    x_matrix_test = matrixData(x_test)

    # Visualización de algunas muestras
    if NSAMPLES > 0:
        print('Visualización de algunas muestras')
    for _ in range(NSAMPLES):

        n = np.random.randint(0,len(x_matrix)) # Elijo un elemento

        visualizeMatrix(x_matrix[n], 'Dígito: '+str(y[n]))

        input("\n--- Pulsar tecla para continuar ---\n")


    # Visualización de los datos en 2D
    if VISUALIZE2D:

        print("Representación de los datos en dos dimensiones:\n")

        print("Generando visualización con PCA: ...")

        x_2D=PCA(n_components=2, random_state=1).fit_transform(x)
        visualize2D(x_2D, y, 'PCA')

        input("\n--- Pulsar tecla para continuar ---\n")

        print("Generando visualización con TSNE: ...")
        x_2D = TSNE(n_components=2, init=x_2D).fit_transform(x)
        visualize2D(x_2D, y, 'TSNE')

        input("\n--- Pulsar tecla para continuar ---\n")

    # Preprocesado

    # Matriz de coeficientes de Pearson para ver la correlación entre los datos
    # (necesito eliminar las características con varianza 0 para poder computarlos)
    x_1 = VarianceThreshold().fit_transform(x)
    corr_m = np.corrcoef(np.transpose(x_1))

    print('Matriz de coeficientes de correlación de Pearson antes del preprocesado (eliminadas características con varianza 0)')

    visualizeMatrix(corr_m, 'Matriz de coeficientes de Pearson\nde las características (sin las de varianza 0)')

    input("\n--- Pulsar tecla para continuar ---\n")

    print('Preprocesado:\n')

    # Elimina las características con baja varianza
    selector = VarianceThreshold(VARTHRESHOLD)

    # Introducción de características polinomiales
    poly=preprocessing.PolynomialFeatures(POLY)

    # Escala las características para dejarlas con media 0 y varianza 1
    scaler = preprocessing.StandardScaler()

    # Escoge un subconjunto de características que explican cierto porcentaje de la variabilidad de la  distribución
    pca = PCA(n_components=VARPCA)

    # Guardo el preprocesado en un pipeline para aplicarlo tanto a train como a test
    pipln = Pipeline([('selector',selector),('poly',poly),('scaler',scaler),('PCA',pca)])

    pipln.fit(x) # Lo ajusto con train

    print('Características:', np.shape(x)[1])

    x = pipln.transform(x) # Aplico las transformaciones al conjunto de entrenamiento
    x_test = pipln.transform(x_test) # Aplico las transformaciones a los datos de test

    # Añado x_0=1 para ajustar el término independiente
    x=np.hstack((np.ones((len(x),1)),x))
    x_test=np.hstack((np.ones((len(x_test),1)),x_test))
        
    print('Tras preprocesado:', np.shape(x)[1])
        
    input("\n--- Pulsar tecla para continuar ---\n")

    corr_m = np.corrcoef(np.transpose(x))
    print('Matriz de coeficientes de correlación de Pearson tras el preprocesado')
    visualizeMatrix(corr_m, 'Matriz de coeficientes de Pearson\n de las características tras preprocesado')

    input("\n--- Pulsar tecla para continuar ---\n")
    

    # Clasificación multietiqueta: SoftMax
    print('Regresión Logística Multietiqueta\n')
    clr = ClassifierLR(LR, MINIBATCH_SIZE, LMBD)
    
    # Ajustar hiperparámetros
    if PARAMSELECT:
        print('Ajustando parámetros del modelo ...')
        param_grid={'lr': np.linspace(*LR_RANGE), 'minibatch_size': MINIBATCH_SIZE_RANGE, 'lmbd':np.linspace(*LMBD_RANGE)}
        searcher = model_selection.GridSearchCV(clr, param_grid, n_jobs=-1,verbose=15)
        search = searcher.fit(x,y)

        print('Mejores parámetros:', search.best_params_)
        print('Mejor puntuación:', search.best_score_)
        clr.set_params(**(search.best_params_)) # Ajusto los parámetos

    # Gradiente descendiente estocástico para ajustar el modelo
    print('Entrenando modelo con SGD ...')
    clr.fit(x,y) # Entreno el modelo con todos los datos

    print('Pérdida logarítmica (aumentada) en la muestra:', clr.logLoss(x,y))
    # Predicciones sobre el conjunto de entrenamiento
    pred = clr.predict(x)
    # Medimos la bondad del resultado
    classificationScore(y, pred, 'train')

    input("\n--- Pulsar tecla para continuar ---\n")

    # V-fold Cross-Validation para estimar el error fuera de la muestra
    print('Validación cruzada del modelo ...')
    Ecv=model_selection.cross_val_score(clr,x,y,cv=V_FOLD,n_jobs=-1) 
    print('Ecv =',np.mean(Ecv))
    
    input("\n--- Pulsar tecla para continuar ---\n")

    # Hacemos algunas pruebas sobre el conjunto de test
    clr.fit(x,y) # Entreno el modelo con todos los datos
    if PRUEBAS > 0:
        print('Pruebas sobre test:\n')
    for _ in range(PRUEBAS):

        n = np.random.randint(0,len(x_test)) # Elijo un elemento
        
        pred=clr._meaning(x_test[n]) # Predigo el valor

        visualizeMatrix(x_matrix_test[n], 'Dígito: '+str(y_test[n])+'\nPredicción: '+str(pred)) # Visualizo la entrada

        input("\n--- Pulsar tecla para continuar ---\n")

    # Predicciones sobre el test
    print('Pérdida logarítmica (aumentada) en test:', clr.logLoss(x_test, y_test))
    pred_test = clr.predict(x_test)
    # Medimos la bondad del resultado
    classificationScore(y_test, pred_test, 'test')
    
    input("\n--- Pulsar tecla para salir ---\n")
