
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.transform import resize

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.ndimage import zoom



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


def entero(l):
        return list(map(int,l))

# Center character by crop and resize image
def centerAndResize(img):
    visualizeMatrix(img,'ORIGINAL')
    thresh = threshold_otsu(img)
    bw = closing(img > min(thresh*2,0.95), square(3))

    bw2=np.array(list(map(entero,bw)))
    visualizeMatrix(bw2, 'BINARY')

    label_image = label(bw)

    mminr=28; mminc=28; mmaxr=0; mmaxc=0
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        mminr=min(minr,mminr)
        mminc=min(minc,mminc)
        mmaxr=max(maxr,mmaxr)
        mmaxc=max(maxc,mmaxc)

    visualizeMatrix(img[mminr:mmaxr,mminc:mmaxc],'CROP')

    return resize(img[mminr:mmaxr,mminc:mmaxc],(24,24))



#####################################

test_mat, test_label = loadPng(TEST_IMG_DIR, characters[:5])

#samples = np.random.randint(0,len(test_mat),10)
samples=[94,159]
for n in samples:
    image=test_mat[n]
    print(n)
    image=centerAndResize(image)
    visualizeMatrix(image,className(test_label[n]))    
        
exit()



"""
# Load train data from images
train_mat, train_label = loadPng(TRAIN_IMG_DIR, characters)
# Matrix to vector       
train=np.reshape(train_mat,(train_mat.shape[0],784))
# Save train as greyscale
saveGrey(TRAIN_GREY, train, train_label)
"""
"""
# Load test data from images
test_mat, test_label = loadPng(TEST_IMG_DIR, characters[:5])
# Matrix to vector       
test=np.reshape(test_mat,(test_mat.shape[0],784))
# Save test as greyscale
saveGrey(TEST_GREY, test, test_label)
"""

x, y = loadGrey(TRAIN_GREY)
x_test, y_test = loadGrey(TEST_GREY)

rf = RandomForestClassifier(n_jobs=4)
knn = KNeighborsClassifier(n_neighbors=3,n_jobs=4)

#rf.fit(x,y)
#knn.fit(x,y)

elements=np.random.randint(0,x_test.shape[0],500)
#print('rf train', rf.score(x,y))
#print('knn train', knn.score(x,y))
#print('rf test', rf.score(x_test,y_test))
#print('knn test', knn.score(x_test[elements],y_test[elements]))
"""
x_m=np.reshape(x,(x.shape[0],28,28))
x_m_test=np.reshape(x_test,(x_test.shape[0],28,28))

def small(m):
    return zoom(m,0.5)

x1_m = np.array(list(map(small,x_m)))
x1_m_test = np.array(list(map(small,x_m_test)))

print('x1_m', x1_m.shape)
print('x1_m_test', x1_m_test.shape)

x1=np.reshape(x1_m,(x1_m.shape[0],196))
x1_test=np.reshape(x1_m_test,(x1_m_test.shape[0],196))
"""

pca= PCA(0.99)
pca.fit(x)
x1=pca.transform(x)
x1_test=pca.transform(x_test)
print(x.shape)
print(x1.shape)
rf.fit(x1,y)
knn.fit(x1,y)

#print('rf train', rf.score(x1,y1))
#print('knn train', knn.score(x1,y1))
print('rf test', rf.score(x1_test,y_test))
print('knn test', knn.score(x1_test[elements],y_test[elements]))


exit()
# Vector to matrix
x_mat=np.reshape(x,(x.shape[0],28,28))

for _ in range(5):
    n = np.random.randint(0,len(x_mat))
    visualizeMatrix(x_mat[n], title=className(y[n]))
    input('\n--- Pulsar tecla para continuar ---\n')

