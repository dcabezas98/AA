import numpy as np
import matplotlib.pyplot as plt

from preprocessing import preprocessing

from sklearn.ensemble import RandomForestClassifier

# Paths
CHARACTERS='datos/characters.txt'

TRAIN_GRAY='datos/DevanagariGrayscale/train.npz'
TEST_GRAY='datos/DevanagariGrayscale/test.npz'

# Matrix visualization
def visualizeMatrix(m, title='', conf=False):
    plt.matshow(m, cmap='viridis')
    plt.colorbar()
    plt.title(title,pad=20.0)
    if conf:
        plt.ylabel('Verdaderos')
        plt.xlabel('Predicciones')
    plt.show()
    
# Names of classes
with open(CHARACTERS,'r') as f:
    characters = f.read().split('\n')[:-1]

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

    train, train_label = loadGrey(TRAIN_GRAY)
    test, test_label = loadGrey(TEST_GRAY)

    train = preprocessing(train)
    test = preprocessing(test)

    print(train.shape)
    print(test.shape)

    rf=RandomForestClassifier(n_jobs=4)
    rf.fit(train,train_label)
    print('Accuracy RF:', rf.score(test,test_label))
    print('Accuracy RF TRAIN:', rf.score(train,train_label))
