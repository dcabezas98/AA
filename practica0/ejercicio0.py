
import sklearn as sk
from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Parte 1:

iris = sk.datasets.load_iris() # Leer dataset

x = iris.data # Características: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
y = iris.target # Clases: ['setosa' 'versicolor' 'virginica']

x2 = x[:,-2:] # Dos últimas características

for j in range(len(iris.target_names)):
    a, b = [], []
    for i in range(len(x2)):
        if y[i]==j:
            a+=[x2[i][0]]
            b+=[x2[i][1]]
    color = list(['tab:red', 'tab:green', 'tab:blue'])[j]
    plt.scatter(a, b, c=color, label=iris.target_names[j])

plt.xlabel(iris.feature_names[-2])
plt.ylabel(iris.feature_names[-1])
plt.title("Petal measures")
plt.legend(title="Classes")
plt.show()

# Parte 2:

# stratify=y permite preservar la proporción de elementos de cada clase
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(x2, y, stratify=y, train_size=0.8, test_size=0.2)

print(Counter(y_train))
print(Counter(y_test))

# Parte 3:

x = np.linspace(0,2*np.pi,100)

plt.plot(x, np.sin(x), c='k', ls='--', label='sin(x)')
plt.plot(x, np.cos(x), c='b', ls='--', label='cos(x)')
plt.plot(x, np.sin(x)+np.cos(x), c='r', ls='--', label='sin(x)+cos(x)')
plt.legend()
plt.show()