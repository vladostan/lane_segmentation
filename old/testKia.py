#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from glob import glob
import numpy as np
import math
from PIL import Image

# In[2]:
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

PATH = os.path.abspath('.')

#name = "2018-11-30-12-58-17_sunny"
name = "kia_mppi_2018-09-06-15-53-35_до_МЧС"

SOURCE_IMAGES = [os.path.join(PATH, "data/images/"+name)]

images = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

images.sort()
print(np.size(images))

# In[19]:
def crop(x, crop_h=320, crop_w=1152, h_offset = 0):
    
    h,w = x.shape[:2]
    h_new = h-(crop_h+h_offset), h-h_offset
    w_new = int(math.floor((w-crop_w)/2.)), int(math.floor(w-(w-crop_w)/2.))
    
    return x[h_new[0]:h_new[1], w_new[0]:w_new[1], :]

# In[20]:
def get_image(path, aspect_ratio = 1):
        
    image = Image.open(path)
    image = image.resize((1152,int(1152/aspect_ratio)))
    
    return(np.asarray(image))

# In[22]:
#images = images[:10]

# In[23]:
from models.Unet import unet

model = unet(input_size = (320,1152,3), n_classes=2)

#print("Model summary:")
#model.summary()

# In[ ]:cd
from keras import optimizers

#model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)
loss = 'binary_crossentropy'
metrics = ['accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, loss, metrics))

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

# model.load_weights('weights/egolane/2018-11-23 11-22-58.hdf5')
model.load_weights("weights/road/2018-11-23 13-41-05.hdf5")

# In[24]:
y_pred = np.zeros([len(images), 320, 1152, 2], dtype=np.float32)

for i,im in enumerate(images):
        
    img = get_image(im, aspect_ratio = 1.25)
    img = crop(img,h_offset=130)
    img = np.expand_dims(img, axis=0)

    y_pred[i] = model.predict(img, batch_size=1, verbose=1)

# In[26]:
np.save("results/"+name, y_pred[:,:,:,1])