# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# mlp.py
# Generate graphs to decide better hyperparameters for MLP

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# MLP Parameters:
N_NEUR= 59 # Number of neurons in hidden layers
FIRST = 58 #55 # 50
LAST = 62 #70 # 100
INC = 1 #5 # 10

# Compares accuracy of different numbers of neurons
def grafNneur(data, label):
    output=[]
    for n in range(FIRST, LAST + 1, INC):
        print(n)
        mlp=MLPClassifier(hidden_layer_sizes= (N_NEUR, N_NEUR), activation='tanh', max_iter = 800)
        score=np.mean(cross_val_score(mlp,data,label,cv=2))
        output.append(score)
        print(output[-1])
    plt.plot(range(FIRST,LAST + 1,INC),output, c='b')
    plt.title('Accuracy media frente a nº de neuronas')
    plt.xlabel('Nº de neuronas')
    plt.ylabel('Accuracy')

    plt.show()
