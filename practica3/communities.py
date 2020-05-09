# Predicción de crímenes violentos per cápita
# David Cabezas Berrido

# Variables ajustables:

TEST_SIZE=0.25 # Proporción de datos usados para test

VISUALIZE2D=True # Para la visualización de los datos en 2D (tarda un poco)

DROPNA=0.5 # Proporción mínima de valores no perdidos para mantener un atributo

VARTHRESHOLD=0.005 # Umbral de varianza por debajo del cual elimino la característica
POLY=2 # Grado de las características polinomiales (poner 1 o 2)
VARPCA=0.99 # Porcentaje de variabilidad de la distribución que deben explicar las características que no elimine

LMBD=78.78787878787878 # Penalización para regularización

PARAMSELECT=False # Para el seleccionador de parámetros (tarda bastante)
LMBD_RANGE=[70, 85, 100] #[0.00001, 500, 10000] # Valores para lambda (inicio, fin, valores)

V_FOLD = 10 # Subdivisiones para Cross-Validation

PRUEBAS = 4  # Número de pruebas detalladas del modelo sobre el conjunto de test

# Rutas a los ficheros de datos
DATA='datos/communities.data'

# Librerías externas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# Fijo la semilla
np.random.seed(25)


# Función para leer los datos
def readData(data_file):
    
    data=pd.read_csv(DATA,sep=',', na_values='?',header=None) # Leo el archivo
    x=data.iloc[:,5:-1] # Elimino las características no predictivas (5 primeras)
    y=data.iloc[:,-1] # Atributo objetivo (a predecir)

    return x, y

# Visualizar en 2D
def visualize2D(x,y, alg_name):

    plt.scatter(x[:,0],x[:,1],c=y, cmap='viridis', alpha=0.4)

    plt.colorbar()

    plt.title('Representación de los datos en 2D y su índice de criminalidad\n usando el algoritmo '+alg_name)
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

# Estimador para regresión lineal
class LinearRegressor(BaseEstimator):

    def __init__(self, lmbd=0):

        self.lmbd=lmbd # Para regularización

    # Ajusta los pesos del modelo mediante el algoritmo de pseudoinversa
    # (tiene en cuenta la regularización)
    def fit(self, x, y):
        m=np.dot(x.T,x) + self.lmbd*np.identity(x.shape[1])
        m=np.linalg.inv(m)
        l=np.dot(x.T,y)
        self.w_ = np.dot(m,l)

    # Predicción para una entrada
    def _meaning(self, x):
        return np.dot(self.w_,x)

    # Predicción para un conjunto de entradas
    def predict(self, x):
        predictions = np.zeros(np.shape(x)[0])
        for n in range(np.shape(x)[0]): # Calculo la predicción para cada elemento
            predictions[n] = self._meaning(x[n])
        return predictions 

    # Error Cuadrático Medio, métrica que usaré
    def MSE(self, x, y):
        return metrics.mean_squared_error(y,self.predict(x))

    # Score: Consideraré como mejor modelo aquel con el menor ECM
    # Para validación
    def score(self, x, y): # El signo menos es porque GridSearchCV elige el de más score
        return - self.MSE(x,y)

    # Pérdida: Error Cuadrático Medio aumentado
    def Eaug(self, x, y):
        return self.MSE(x,y) + self.lmbd*np.linalg.norm(self.w_)**2

# Main
if __name__ == "__main__":

    print('\nEjercicio de regresión:')
    print('Predicción del número de crímenes violentos per capita\n')

    x, y = readData(DATA) # Lee datos

    # Split: separamos los conjuntos de training y test
    x, x_test, y, y_test = model_selection.train_test_split(x, y, train_size=1-TEST_SIZE, test_size=TEST_SIZE, random_state=31)

    # Preprocesado
    print('Características:', np.shape(x)[1])

    # Elimina los atributos en los que abundan los valores perdidos
    if DROPNA:
        x.dropna(1,thresh=len(x)*DROPNA,inplace=True)
        x_test=x_test.reindex(x.columns,axis=1)
        print('Tras eliminar las que tienen demasiados valores perdidos:', np.shape(x)[1])

    print('\nRelleno valores perdidos con KNN')

    # Estimamos valores perdidos por KNN
    imputer = KNNImputer(missing_values=np.NaN,weights='distance').fit(x)
    x=imputer.transform(x)
    x_test=imputer.transform(x_test)

    # Visualización de los datos en 2D
    if VISUALIZE2D:

        print("\nRepresentación de los datos en dos dimensiones:\n")

        print("Generando visualización con PCA: ...")

        x_2D=PCA(n_components=2, random_state=1).fit_transform(x)
        visualize2D(x_2D, y, 'PCA')

        input("\n--- Pulsar tecla para continuar ---\n")

        print("Generando visualización con TSNE: ...")
        x_2D = TSNE(n_components=2, init=x_2D).fit_transform(x)
        visualize2D(x_2D, y, 'TSNE')

        input("\n--- Pulsar tecla para continuar ---\n")    

    # Matriz de coeficientes de Pearson para ver la correlación entre los datos
    # (necesito eliminar las características con varianza 0 para poder computarlos)
    x_1 = VarianceThreshold().fit_transform(x)
    corr_m = np.corrcoef(np.transpose(x_1))

    print('\nMatriz de coeficientes de correlación de Pearson antes del preprocesado (eliminadas características con varianza 0)')

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
    pipln = Pipeline([('imputer',imputer),('selector',selector),('poly',poly),('scaler',scaler),('PCA',pca)])

    pipln.fit(x) # Lo ajusto con train

    x = pipln.transform(x) # Aplico las transformaciones al conjunto de entrenamiento
    x_test = pipln.transform(x_test) # Aplico las transformaciones a los datos de test
    
    print('Características tras preprocesado:', np.shape(x)[1])
    
    input("\n--- Pulsar tecla para continuar ---\n")

    corr_m = np.corrcoef(np.transpose(x))
    print('Matriz de coeficientes de correlación de Pearson tras el preprocesado')
    visualizeMatrix(corr_m, 'Matriz de coeficientes de Pearson\n de las características tras preprocesado')

    input("\n--- Pulsar tecla para continuar ---\n")

    # Regresión Lineal
    print('Regresión Lineal')
    regressor = LinearRegressor(lmbd=LMBD)

    # Añado x_0=1 para ajustar el término independiente
    x=np.hstack((np.ones((len(x),1)),x))
    x_test=np.hstack((np.ones((len(x_test),1)),x_test))

    # Ajustar hiperparámetros: en este caso sólo lambda
    if PARAMSELECT:
        print('Ajustando parámetros del modelo ...')
        param_grid={'lmbd':np.linspace(*LMBD_RANGE)}
        searcher = model_selection.GridSearchCV(regressor, param_grid, n_jobs=-1,verbose=15)
        search = searcher.fit(x,y)

        print('Mejores parámetros:', search.best_params_)
        print('Mejor puntuación:', search.best_score_)
        regressor.set_params(**(search.best_params_)) # Ajusto los parámetos

        input("\n--- Pulsar tecla para continuar ---\n")

    print('Ajustando modelo mediante pseudoinversa ...')
    regressor.fit(x,y) # Entreno el modelo con todos los datos

    # Eficacia sobre la muestra
    print('ECM aumentado sobre la muestra:', regressor.Eaug(x,y))
    mse=regressor.MSE(x,y)
    print('ECM sobre la muestra:', mse)
    print('R²=',1-mse/np.var(y))

    input("\n--- Pulsar tecla para continuar ---\n")

    # V-fold Cross-Validation para estimar el error fuera de la muestra
    print('Validación cruzada del modelo ...')
    Ecv=model_selection.cross_val_score(regressor,x,y,cv=V_FOLD,n_jobs=-1) 
    print('Ecv =',-np.mean(Ecv))

    input("\n--- Pulsar tecla para continuar ---\n")

    # Pruebas sobre el test
    regressor.fit(x,y) # Entreno el modelo con todos los datos
    if PRUEBAS > 0:
        print('Pruebas sobre test:\n')
    for _ in range(PRUEBAS):

        print('Selecciono un elemento aleatorio de test')
        n = np.random.randint(0,len(x_test)) # Elijo un elemento
        
        pred=regressor._meaning(x_test[n]) # Predigo el valor
        print('Objetivo:',y_test.iloc[n])
        print('Predicción:', pred)
        print('Diferencia:', pred-y_test.iloc[n])

        input("\n--- Pulsar tecla para continuar ---\n")

    # Predicciones sobre el test
    print('ECM aumentado sobre el test:', regressor.Eaug(x_test,y_test))
    mse=regressor.MSE(x_test,y_test)
    print('ECM sobre el test:', mse)
    print('R²=',1-mse/np.var(y_test))

    input("\n--- Pulsar tecla para continuar ---\n")

    # Comparación con KNN
    print('Comparación con KNN')
    knn= KNeighborsRegressor()
    knn.fit(x,y)
    pred=knn.predict(x_test)
    print('ECM sobre test:', metrics.mean_squared_error(y_test,pred))
    print('KNN R²=',knn.score(x_test,y_test))

    input("\n--- Pulsar tecla para continuar ---\n")

    # Comparación con Random Forest
    print('Comparación con Random Forest')
    rf = RandomForestRegressor(n_jobs=-1)
    rf.fit(x,y)
    pred=rf.predict(x_test)
    print('ECM sobre test:', metrics.mean_squared_error(y_test,pred))
    print('KNN R²=',rf.score(x_test,y_test))

    input("\n--- Pulsar tecla para salir ---\n")
