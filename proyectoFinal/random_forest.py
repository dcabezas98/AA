# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# random_forest.py
# Generate graphs to decide better hyperparameters for Random Forest

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Random Forest Parameters:
RF_N_ESTIMATORS= 277 # Number of estimators for baggin
FIRST_N = 285 #275 #200 #50
LAST_N = 295 #300
INC_N = 2 #5 #25 #50

RF_ALPHA = 0 # Cost-Complexity Parameter
FIRST_A = 0
LAST_A = 0.00001 # 0.1
NUM_A = 5

# Compares accuracy of different numbers of estimators
def grafNestimators(data, label):
    output=[]
    for n in range(FIRST_N, LAST_N + 1, INC_N):
        print(n)
        rf=RandomForestClassifier(n_estimators=n, n_jobs=4)
        score=np.mean(cross_val_score(rf,data,label,cv=3))
        output.append(score)
        print(output[-1])
    plt.plot(range(FIRST_N,LAST_N + 1,INC_N),output, c='g')
    plt.title('Accuracy media frente a nº de estimadores')
    plt.xlabel('Nº de estimadores')
    plt.ylabel('Accuracy')

    plt.show()
    
# Compares accuracy of different values of alpha
def grafAlpha(train, train_label, val, val_label):
    output=[]
    for a in np.linspace(FIRST_A, LAST_A, NUM_A):
        print(a)
        rf=RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, ccp_alpha=a, n_jobs=4)
        output.append(np.mean(cross_val_score(rf,data,label,cv=3)))

    print(output)

    plt.plot(np.linspace(FIRST_A,LAST_A,NUM_A),output, c='g')
    plt.title('Accuracy media frente a alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.show()
