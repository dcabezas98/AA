# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# logistic_regression.py
# Generate graphs to decide better hyperparameters for LR

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

# Logistic Regression Parameters:
ALPHA = 4e-5 # Alpha for regularization
FIRST = 0.000025 #0.00002 #1e-5 #5e-5
LAST = 0.000045 #0.00007 #0.000353 #0.005
NUM = 5 #6 #10 #11

# Compares accuracy of different values of alpha
def grafLRAlpha(data, label):
    output=[]
    for a in np.linspace(FIRST, LAST, NUM):
        print(a)
        lr=SGDClassifier(loss='log', alpha=a,n_jobs=4,max_iter=2000)
        score=np.mean(cross_val_score(lr,data,label,cv=3))
        output.append(score)
        print(output[-1])
    plt.plot(np.linspace(FIRST,LAST, NUM),output, c='r')
    plt.title('Accuracy media frente a alpha (LR)')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')

    plt.show()
