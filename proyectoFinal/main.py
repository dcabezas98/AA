# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# main.py

import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocessing, polynomial
from random_forest import grafNestimators, grafAlpha
from mlp import grafNneur
from logistic_regression import grafLRAlpha

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

# Validation score
VALIDATION=True

# For 2D visualization of DATA
VISUALIZE2D=False

# To show that there are no features with null variance
VARTHRESH=False

# Generate graphs for hyperparameter selections (takes several hours)
PARAMSELECT=False

# Save greyscale data to disc
PNG_TO_NP=False

# Save preprocessed data to disc
SAVE_PRE=False

# Load preprocecessed data directly from disc
LOAD_PRE=False

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

# Logistic Regression
LR_ALPHA=4e-5

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
    
# Accuracy and confusion matrix
def modelPerformance(estimator, train, train_label, test, test_label):
    estimator.fit(train, train_label)
    pred = estimator.predict(test)
    acc = accuracy_score(test_label, pred)
    conf_mat=confusion_matrix(test_label, pred, normalize='all')
    return acc, conf_mat

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
        print('Número de características:',train.shape[1])
        print('Preprocesando datos')
        train = preprocessing(train)
        test = preprocessing(test)
    
    print('Número de características:',train.shape[1])

    if SAVE_PRE:
        print('Guardando datos preprocesados')
        np.savez_compressed(TRAIN_PRE, train, train_label)
        np.savez_compressed(TEST_PRE, test, test_label)

    # New polynomial features for lineal model
    trainLin=polynomial(train,4)
    stdScaler=StandardScaler().fit(trainLin)
    trainLin=stdScaler.transform(trainLin)
    testLin=polynomial(test,4)
    testLin=stdScaler.transform(trainLin)
    print('Características para el modelo lineal:', trainLin.shape[1])
    input("\n--- Pulsar tecla para continuar ---\n")

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

        input("\n--- Pulsar tecla para continuar ---\n")
    
    if VARTHRESH: # To show that there are no useless (variance 0) features
        print('Características tras preprocesado:', train.shape[1])
        varthresh = VarianceThreshold()
        train=varthresh.fit_transform(train)
        print('Características tras eliminar las de varianza nula:', train.shape[1])
        input("\n--- Pulsar tecla para continuar ---\n")

    # For hyperparameters selection
    if PARAMSELECT:
        # Random Forest
        grafNestimators(train, train_label)
        grafAlpha(train, train_label)
        # MLP
        grafNneur(train, train_label)
        # Logistic Regression
        grafLRAlpha(trainLin, train_label)

    # Models
    rf=RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, ccp_alpha=RF_ALPHA, n_jobs=4) # Random Forest
    mlp=MLPClassifier(hidden_layer_sizes=(MLP_NNEURS,MLP_NNEURS),activation='tanh',max_iter=800,early_stopping=True) # MLP
    lr=SGDClassifier(loss='log', alpha=LR_ALPHA, n_jobs=4, max_iter=2000) # Logistic Regression

    # Validation scores
    if VALIDATION:
        # Split validation set from train data
        tra, val, tra_label, val_label = train_test_split(train, train_label, stratify=train_label, train_size=0.8, test_size=0.2)

        # Split validation set from train data for linear model
        traLin, valLin, traLin_label, valLin_label = train_test_split(trainLin, train_label, stratify=train_label, train_size=0.8, test_size=0.2)

        print('Estimaciones por validación')
        '''
        print('\nRandom Forest:')
        rf.fit(tra, tra_label)
        print('Train Accuracy:',rf.score(tra,tra_label))
        print('Validation Accuracy:', rf.score(val,val_label))
        '''
        
        #input("\n--- Pulsar tecla para continuar ---\n")

        print('\nMulti-Layer Perceptron (MLP):')
        mlp.fit(tra,tra_label)
        print('Train Accuracy:',mlp.score(tra,tra_label))
        print('Validation Accuracy:', mlp.score(val,val_label))
        
        #input("\n--- Pulsar tecla para continuar ---\n")
        '''
        print('\nRegresión Logística:')
        lr.fit(traLin,traLin_label)
        print('Train Accuracy:',lr.score(traLin,traLin_label))
        print('Validation Accuracy:', lr.score(valLin,valLin_label))
        '''

        #input("\n--- Pulsar tecla para continuar ---\n")

    exit()

    print('Desempeño sobre Test')
    '''
    print('\nRandom Forest:')
    acc, conf_mat = modelPerformance(rf,train,train_label,test,test_label)
    print('Accuracy:',acc)
    #visualizeMatrix(conf_mat,'Random Forest:\nMatriz de confusión sobre Test',conf=True)
    '''
    #input("\n--- Pulsar tecla para continuar ---\n")

    print('\nMulti-Layer Perceptron (MLP):')
    acc, conf_mat = modelPerformance(mlp,train,train_label,test,test_label)
    print('Accuracy:',acc)
    #visualizeMatrix(conf_mat,'MLP:\nMatriz de confusión sobre Test',conf=True)

    #input("\n--- Pulsar tecla para continuar ---\n")
    '''
    print('\nRegresión Logística:')
    acc, conf_mat = modelPerformance(lr,train,train_label,test,test_label)
    print('Accuracy:',acc)
    #visualizeMatrix(conf_mat,'Regresión Logística:\nMatriz de confusión sobre Test',conf=True)
    '''
    #input("\n--- Pulsar tecla para continuar ---\n")