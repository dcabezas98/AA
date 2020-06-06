
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, block_reduce
from skimage.transform import resize

from joblib import Parallel, delayed

TEST_GRAY='datos/DevanagariHandwrittenCharacterDataset/Test/'
# Paths
CHARACTERS='datos/characters.txt'

WIDTH=24

# Names of classes
with open(CHARACTERS,'r') as f:
    characters = f.read().split('\n')[:-1]

def entero(l):
        return list(map(int,l))

# Center character by crop and resize image
def centerAndResize(img):
    
    # Ignote low intensity pixel to obtain components
    thresh = threshold_otsu(img)
    bw = closing(img > min(thresh*2,0.95), square(3))
    bw2=np.array(list(map(entero,bw)))
    visualizeMatrix(bw2,'Closing del thresholding')
    label_image = label(bw) # Separate into connected regions

    # Compute box that contains all components
    mminr=28; mminc=28; mmaxr=0; mmaxc=0
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        mminr=min(minr,mminr)
        mminc=min(minc,mminc)
        mmaxr=max(maxr,mmaxr)
        mmaxc=max(maxc,mmaxc)

    visualizeMatrix(img[mminr:mmaxr,mminc:mmaxc],'Recortado a '+str(mmaxr-mminr)+' x '+str(mmaxc-mminc))
    # Resize to unified size
    return resize(img[mminr:mmaxr,mminc:mmaxc],(WIDTH,WIDTH), anti_aliasing=True)

# Preprocessing for single image
def preprocess(img):
    img = np.reshape(img,(28,28))
    img = centerAndResize(img)
    if BLOCK_REDUCE:
        img = block_reduce(img,(2,2),np.mean)
    img = np.reshape(img,img.shape[0]*img.shape[1])
    return img


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
 
# Load data from .png format
def loadPng(folder, characters=characters):
    data=[]
    for c in characters:
        print('Cargando: '+c)
        path_to_folder=folder+c+'/75428.png'
        image = plt.imread(path_to_folder)[2:-2,2:-2] # Cut frame
        data.append(image)
            
    return np.array(image,np.float32)

test = loadPng(TEST_GRAY, characters[4:5])

img = test
l=characters[4]

print(img.shape)

visualizeMatrix(img,'Imagen de '+l)

img=centerAndResize(img)

visualizeMatrix(img, 'Centrado y reescalado a 24 x 24')

img = block_reduce(img,(2,2),np.mean)

visualizeMatrix(img, 'Downsampling a 12 x 12')
