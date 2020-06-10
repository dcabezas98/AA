# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# main.py

import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from preprocessing import preprocessing, polynomial
from model import modelPerformance, modelAccuracy
from random_forest import grafNestimators, grafAlpha
from mlp import grafNneur
from logistic_regression import grafLRAlpha


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

# For 2D visualization of DATA
VISUALIZE2D=False

# To show that there are no features with null variance
VARTHRESH=False

# Generate graphs for hyperparameter selections (take several hours)
PARAMSELECT=True

# Save greyscale data to disc
PNG_TO_NP=False

# Save preprocessed data to disc
SAVE_PRE=False

# Load preprocecessed data directly from disc
LOAD_PRE=True

# Paths
CHARACTERS='datos/characters.txt'

TRAIN_GRAY='datos/DevanagariGrayscale/train.npz'
TEST_GRAY='datos/DevanagariGrayscale/test.npz'

TRAIN_PRE='datos/DevanagariPreprocessed/train.npz'
TEST_PRE='datos/DevanagariPreprocessed/test.npz'

# Hyperparameters for each model:

# Random Forest:
RF_N_ESTIMATORS=287 # Number of estimators for baggin
RF_ALPHA=0 # Cost-Complexity Parameter

# Multi-Layer Perceptron:
MLP_NNEURS=59

# Names of classes
with open(CHARACTERS,'r') as f:
    characters = f.read().split('\n')[:-1]

# Matrix visualization
def visualizeMatrix(m, title='', conf=False):
    plt.matshow(m, cmap='viridis')
    plt.colorbar()
    plt.title(title,pad=20.0)
    if conf:
        plt.ylabel('Verdaderos')
        plt.xlabel('Predicciones')
    plt.show()
    
# Class name from integer label
def className(n,c=characters):
    assert 1<=n<=46
    return c[n-1]
 
# Load greyscale vector
def loadGrey(filename):
    X=np.load(filename)
    data = X['arr_0']
    label= X['arr_1']
    X.close()
    return data, label

# 2D data visualization
def plot2D(x,y, alg_name,classes):
    plt.scatter(x[:,0],x[:,1],c=y, cmap='tab20', alpha=0.5)
    plt.title('Representación de los '+classes+' en dos dimensiones\n usando el algoritmo '+alg_name)
    plt.show()

# 2D projection and visualization of data
def visualize2D(x,y,classes=''):
    x2=PCA(n_components=2).fit_transform(x) # PCA projection
    plot2D(x2,y,'PCA',classes)
    x2=TSNE(n_components=2,init=x2).fit_transform(x) # TSNE projection
    plot2D(x2,y,'TSNE',classes)
    

# Main
if __name__ == "__main__":

    print('CLASIFICACIÓN DE SÍMBOLOS DEVANAGARI\n')

    if PNG_TO_NP: # Load PNG images and save greyscale format to disc
        exec(open('./png_to_np.py').read())

    if LOAD_PRE:
        print('Cargando datos preprocesados')
        train, train_label = loadGrey(TRAIN_PRE)
        test, test_label = loadGrey(TEST_PRE)
    else:
        print('Cargando datos')
        train, train_label = loadGrey(TRAIN_GRAY)
        test, test_label = loadGrey(TEST_GRAY)
        print('Preprocesando datos')
        train = preprocessing(train)
        test = preprocessing(test)

    if SAVE_PRE:
        print('Guardando datos preprocesados')
        np.savez_compressed(TRAIN_PRE, train, train_label)
        np.savez_compressed(TEST_PRE, test, test_label)

    if VISUALIZE2D: # Generate 2D visualization
        x, x1, y, y1 = train_test_split(train, train_label,stratify=train_label, train_size=0.4)

        x2=[x[i] for i in range(len(x)) if 1<=y[i]<=18] # Classes to plot
        y2=[y[i] for i in range(len(y)) if 1<=y[i]<=18]
        visualize2D(x2,y2,'caracteres del 1 al 18')

        x2=[x[i] for i in range(len(x)) if 19<=y[i]<=36] # Classes to plot
        y2=[y[i] for i in range(len(y)) if 19<=y[i]<=36]
        visualize2D(x2,y2,'caracteres del 19 al 36')

        x2=[x[i] for i in range(len(x)) if 37<=y[i]<=46] # Classes to plot
        y2=[y[i] for i in range(len(y)) if 37<=y[i]<=46]
        visualize2D(x2,y2,'dígitos del 0 al 9')
    
    if VARTHRESH: # To show that there are no useless (variance 0) features
        print('Características tras preprocesado:', train.shape[1])
        varthresh = VarianceThreshold()
        train=varthresh.fit_transform(train)
        print('Características tras eliminar las de varianza nula:', train.shape[1])

    # For hyperparameters selection
    if PARAMSELECT:
        # Random Forest
        #grafNestimators(train, train_label)
        #grafAlpha(train, train_label)
        # MLP
        #grafNneur(train, train_label)
        # LogisticRegression
        train2=polynomial(train,4)
        train2=StandardScaler().fit_transform(train2)
        grafLRAlpha(train2, train_label)


    # Split validation set from train data
    # train, val, train_label, val_label = train_test_split(train, train_label, stratify=train_label, train_size=0.7, test_size=0.3)

    print('\nRandom Forest:\n')

    '''
    rf=RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, ccp_alpha=a, n_jobs=4)
    rf.fit(train, train_label)
    print('Random Forest Accuracy: ', rf.score())
    '''

    exit()
    print('\nMLP:\n')
    mlp=MLPClassifier(hidden_layer_sizes=(100,100),activation='tanh',max_iter=400,early_stopping=False)
    mlp.fit(train,train_label)
    print('MLP:',mlp.score(val,val_label))
    
    print('\nRegresión Logística:\n')

    print(train.shape)

    lr=LogisticRegression(max_iter=1000)
    lr.fit(train,train_label)
    print('LR:',lr.score(val,val_label))

    sgd=SGDClassifier(loss='log')
    sgd.fit(train,train_label)
    print('SGD:',sgd.score(val,val_label))

    print('\nAdaBoost:\n')
    exit()
    ab=AdaBoostClassifier(n_estimators=200, learning_rate=0.1)
    ab.fit(train,train_label)
    print('AB:',ab.score(val,val_label))    
