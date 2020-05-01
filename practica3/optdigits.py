# Clasificación de Dígitos
# David Cabezas Berrido

# Parámetros ajustables:

VISUALIZAR2D=False # Para la visualización de los datos en 2D (tarda un poco)
PREPROCESSING=True # Preprocesado de los datos
VARTHRESHOLD=0.005 # Umbral de varianza por debajo del cual elimino la característica

TRAIN='datos/optdigits.tra'
TEST='datos/optdigits.tes'

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.lines import Line2D

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold

np.random.seed(18)

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

    label_encoder = preprocessing.LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y)
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False, dtype=np.int8)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
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


if __name__ == "__main__":

    x, y = readData(TRAIN) # Lee datos de entrenamiento

    ####### Para hacer pruebas (con todos los datos tarda bastante)
    #x=x[:200]
    #y=y[:200]
    #######

    # Visualización de los datos en 2D
    if(VISUALIZAR2D):

        print("Representación de los dígitos en dos dimensiones:\n")

        print("Generando visualización con PCA: ...")

        x_2D=PCA(n_components=2, random_state=1).fit_transform(x)
        visualize2D(x_2D, y, 'PCA')

        input("\n--- Pulsar tecla para continuar ---\n")

        print("Generando visualización con TSNE: ...")
        x_2D = TSNE(n_components=2, init=x_2D).fit_transform(x)
        visualize2D(x_2D, y, 'TSNE')

        input("\n--- Pulsar tecla para continuar ---\n")


    # Preprocesado
    if(PREPROCESSING):
        print('Preprocesado:\n')
        print('Características:',x.shape[1])
        # Elimina las características con baja varianza
        selector = VarianceThreshold(VARTHRESHOLD)
        x = selector.fit_transform(x)
        print('Tras eliminar las de baja varianza:',x.shape[1])