
import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
CHARACTERS='datos/characters.txt'

TRAIN_IMG_DIR='datos/DevanagariHandwrittenCharacterDataset/Train/'
TEST_IMG_DIR='datos/DevanagariHandwrittenCharacterDataset/Test/'

TRAIN_GREY='datos/DevanagariGreyscale/train.npz'
TEST_GREY='datos/DevanagariGreyscale/test.npz'

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
 
# Load data from .png format
def loadPng(folder, characters=characters):
    data=[]
    label=[]
    l=0
    for c in characters:
        l+=1 # Classes from 1 to 46
        print('Cargando: '+c)
        path_to_folder=folder+c+'/*.png'
        for img_path in glob.glob(path_to_folder):
            image = plt.imread(img_path)[2:-2,2:-2] # Cut frame
            #visualizeMatrix(image,title=c)
            data.append(image)
            label.append(l)
            
    return np.array(data,np.float32), np.array(label,np.int8)

# Save greyscale vector
def saveGrey(filename, data, label):
    np.savez_compressed(filename, data, label)
    """ No requieren concatenar
    https://numpy.org/devdocs/reference/generated/numpy.savez.html#numpy.savez
    https://numpy.org/devdocs/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
    """

# Load greyscale vector
def loadGrey(filename):
    X=np.load(filename)
    data = X['arr_0']
    label= X['arr_1']
    X.close()
    return data, label

"""
# Load train data from images
train_mat, train_label = loadPng(TRAIN_IMG_DIR, characters)
# Matrix to vector       
train=np.reshape(train_mat,(train_mat.shape[0],784))
# Save train as greyscale
saveGrey(TRAIN_GREY, train, train_label)
"""

# Load test data from images
test_mat, test_label = loadPng(TEST_IMG_DIR, characters[:5])
# Matrix to vector       
test=np.reshape(test_mat,(test_mat.shape[0],784))
# Save test as greyscale
saveGrey(TEST_GREY, test, test_label)


x, y = loadGrey(TEST_GREY)



# Vector to matrix
x_mat=np.reshape(x,(x.shape[0],28,28))

for _ in range(5):
    n = np.random.randint(0,len(x_mat))
    visualizeMatrix(x_mat[n], title=className(y[n]))
    input('\n--- Pulsar tecla para continuar ---\n')

