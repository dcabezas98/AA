# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# png_to_np.py
# Lee los datos en formato .png y los escribe como arrays de numpy (sin marco)

import glob
import numpy as np
import matplotlib.pyplot as plt

# Paths
CHARACTERS='datos/characters.txt'

TRAIN_IMG_DIR='datos/DevanagariHandwrittenCharacterDataset/Train/'
TEST_IMG_DIR='datos/DevanagariHandwrittenCharacterDataset/Test/'

TRAIN_GRAY='datos/DevanagariGrayscale/train.npz'
TEST_GRAY='datos/DevanagariGrayscale/test.npz'
    
# Names of classes
with open(CHARACTERS,'r') as f:
    characters = f.read().split('\n')[:-1]

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
            data.append(image)
            label.append(l)
            
    return np.array(data,np.float32), np.array(label,np.int8)

# Save greyscale vector
def saveGrey(filename, data, label):
    np.savez_compressed(filename, data, label)

print('Cargando datos de entrenamiento')
# Load train data from images
train_mat, train_label = loadPng(TRAIN_IMG_DIR, characters)
# Matrix to vector
train=np.reshape(train_mat,(train_mat.shape[0],784))
# Save train as greyscale
saveGrey(TRAIN_GRAY, train, train_label)
print('Datos de entrenamiento guardados en grayscale')

print('Cargando datos de test')
# Load test data from images
test_mat, test_label = loadPng(TEST_IMG_DIR, characters)
# Matrix to vector       
test=np.reshape(test_mat,(test_mat.shape[0],784))
# Save test as greyscale
saveGrey(TEST_GRAY, test, test_label)
print('Datos de test guardados en grayscale')
