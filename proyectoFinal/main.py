# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# main.py

import numpy as np
import matplotlib.pyplot as plt

from itertools import product

from preprocessing import preprocessing
from model import modelPerformance, modelAccuracy

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

# Random Forest Parameters:
RF_N_ESTIMATORS=[100,200] # Number of estimators for baggin
RF_ALPHA=[0,0.05,0.1] # Cost-Complexity Parameter
#RF_SAMPLES=[0.5,0.75,None] # Subsample size for bootstrap

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

    # Split validation set from train data
    train, val, train_label, val_label = train_test_split(train, train_label, stratify=train_label, train_size=0.7, test_size=0.3)

    print(train.shape)
    print(val.shape)
    
    print('\nRANDOM FOREST:\n')
    # Validation score for hyperparameters in grid
"""
    for n, a, s in product(RF_N_ESTIMATORS,RF_ALPHA,RF_SAMPLES):
        rf=RandomForestClassifier(n_estimators=n, ccp_alpha=a, max_samples=s, criterion='entropy', n_jobs=4)
        acc_val, acc_train = modelAccuracy(rf, train, train_label, val, val_label)
        print('n_estimators =', n)
        print('alpha =',a)
        print('bootstrap samples =',s)
        print('Accuracy sobre train:',acc_train)
        print('Accuracy sobre validación:',acc_val)
        print()
"""
