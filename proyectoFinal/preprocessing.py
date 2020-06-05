# Aprendizaje Automático: Proyecto Final
# Clasificación de símbolos Devanagari
# Patricia Córdoba Hidalgo
# David Cabezas Berrido

# preprocessing.py
# Preprocesado de imágenes: centrado y eliminación de dimensionalidad
# promediando por bloques

import numpy as np

from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops, block_reduce
from skimage.transform import resize

from joblib import Parallel, delayed

# Tuneable parameters
WIDTH=24 # WIDTH to resize
BLOCK_REDUCE=True # Wether or not to perform block reduce

# Center character by crop and resize image
def centerAndResize(img):

    # Ignote low intensity pixel to obtain components
    thresh = threshold_otsu(img)
    bw = closing(img > min(thresh*2,0.95), square(3))
    label_image = label(bw) # Separate into connected regions

    # Compute box that contains all components
    mminr=28; mminc=28; mmaxr=0; mmaxc=0
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        mminr=min(minr,mminr)
        mminc=min(minc,mminc)
        mmaxr=max(maxr,mmaxr)
        mmaxc=max(maxc,mmaxc)

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
    
# Aplay all preprocessing to data
def preprocessing(data):
    out = Parallel(n_jobs=4)(map(delayed(preprocess),data))            
    return np.array(out,np.float32)
