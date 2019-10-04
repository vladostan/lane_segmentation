def extract_largest_blobs(mask, area_threshold, num_blobs=1):

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=4)
                
    st = stats[:,-1][stats[:,-1] > area_threshold] #Ban areas smaller than threshold
                    
    if nb_components == 1 or len(st) < 2:
        return None, None, None

    if (num_blobs <= len(st)-1):
        n = num_blobs+1
    else:
        n = len(st)
    
    blob_index = np.argsort(stats[:,-1])[-n:-1]
                
    return output, blob_index[::-1], centroids[blob_index[::-1]]

def centroids(centroids_a, centroids_b):
    
    if len(centroids_b) < 2:
        return 1
    
    cb_sorted = centroids_b[:,0].copy()
    cb_sorted.sort()

    if (cb_sorted[0] < centroids_a[0,0] < cb_sorted[1]):
        return 2
    else:
        return 1

def fill_holes(mask):

    mask_floodfill = mask.astype('uint8').copy()
    h, w = mask.shape[:2]
    cv2.floodFill(mask_floodfill, np.zeros((h+2, w+2), np.uint8), (0,0), 255)

    out = mask | cv2.bitwise_not(mask_floodfill)
    
    return out.astype(np.bool)

def merge(mask_a, mask_b):
    
    out = np.zeros((256,640), dtype=np.uint8)

    out[mask_b] = 2
    out[mask_a] = 1
    
    return out

def postProcess(mask_a, mask_b, area_threshold):
        
    output_a, blob_index_a, centroids_a = extract_largest_blobs(mask_a, area_threshold)
    output_b, blob_index_b, centroids_b = extract_largest_blobs(mask_b, area_threshold, num_blobs=2)
        
    anon = centroids_a is None
    bnon = centroids_b is None
    
    if anon:
        mask_a = np.zeros_like(mask_a)
    
    if bnon:
        mask_b = np.zeros_like(mask_b)
    
    if anon and bnon:
        return merge(mask_a, mask_b)
    
    if not anon:
        mask_a = output_a == blob_index_a[0]
        
    if not bnon:
        mask_b = output_b == blob_index_b[0]
        
    if not anon and not bnon:
        keep = centroids(centroids_a, centroids_b) # Num of centroids_b to keep
        if keep > 1:
            mask_b += output_b == blob_index_b[1]
    
    if not anon:
        mask_a = fill_holes(mask_a)
        
    if not bnon:
        mask_b = fill_holes(mask_b)
            
    return merge(mask_a, mask_b)

# -*- coding: utf-8 -*-

# In[1]:
import os
import numpy as np
from PIL import Image
import time
from glob import glob
import cv2

# In[]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

# In[2]:
def get_image(path):
    
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    
    return img

# In[4]:
from models.Unet import unet
from models.Unet_short import unet_short
#from models.DeepLabv3plus import Deeplabv3
from models.deeplabv31p import Deeplabv31p
from segmentation_models import Unet, FPN, Linknet, PSPNet
from segmentation_models.backbones import get_preprocessing

num_classes = 3
input_shape = (256,640,3)

#model = unet(input_size = input_shape, n_classes=num_classes)
#model = unet_short(input_size = input_shape, n_classes=num_classes)
#model = Deeplabv3(input_shape=input_shape, classes=num_classes)  
#model = Deeplabv3(input_shape=input_shape, classes=num_classes, backbone='xception')  
#model = Deeplabv31p(input_shape = input_shape, classes = num_classes)

#backbone = 'densenet121'
backbone = 'resnet18'

#model = Unet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
#model = FPN(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
#model = PSPNet(backbone_name='resnet34', input_shape=input_shape, classes=num_classes, activation='softmax')

#from keras import optimizers
#optimizer = optimizers.Adam(lr = 1e-4)
#metrics = ['accuracy']
#loss = 'categorical_crossentropy'
#model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.summary()

# In[7]:
PATH = os.path.abspath('data')

SOURCE_IMAGES = [os.path.join(PATH, "images/innopolis")]

images = []
labels = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    labels.extend(glob(os.path.join(si.replace("images/","labels/"), "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

images.sort()
labels.sort()

print(len(images))
print(len(labels))

# In[]
from sklearn.model_selection import train_test_split

test_size = 0.99
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)

#images_test = images_test[:50]
print(len(images_test))

# In[56]:
model.predict(np.zeros((1,256,640,3)))

preprocessing_fn = get_preprocessing(backbone)

start_time = time.time()

for i, im in enumerate(images_test):
    
    x = get_image(im)
    x = np.expand_dims(x, axis = 0)
    x = preprocessing_fn(x)
    model.predict(x)
    #model.predict(np.zeros((1,256,640,3)))
    print(i)
            
#        mask_np = output.squeeze().cpu().numpy()
    
print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(len(images_test)/(time.time() - start_time)))

# In[]:
#import matplotlib.pyplot as plt
#import numpy as np
#
#img = plt.imread("/home/kenny/Desktop/lanes-segmentation/image3.jpg")
#y = model.predict(np.expand_dims(img[:256,:256,], axis=0))
#out = y.squeeze().reshape(256,256,21)
#mask = np.argmax(out, axis=-1)
#plt.imshow(mask)

