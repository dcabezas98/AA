
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt
from random import shuffle

# Parte 1:

iris = sk.datasets.load_iris() # Leer dataset

x = iris.data # Características: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
y = iris.target # Clases: ['setosa' 'versicolor' 'virginica']

x2 = [f[-2:] for f in x] # Dos últimas características

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

data = list(zip(x2, y))

training=[]
test=[]

for j in range(len(iris.target_names)):
    a = [d for d in data if d[-1]==j]
    shuffle(a)
    training+=a[:int(len(a)*0.8)]
    test+=a[int(len(a)*0.8):]

print(test)
print(len(training))