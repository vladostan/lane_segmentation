#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np


# In[4]:


PATH = os.path.abspath('data')

# SOURCE_IMAGES = [os.path.join(PATH, "images/um")]
SOURCE_IMAGES = [os.path.join(PATH, "images/um"), 
                 os.path.join(PATH, "images/umm"), 
                 os.path.join(PATH, "images/uu")]

images = []
labels = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    labels.extend(glob(os.path.join(si.replace("images/","labels/road/"), "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

images.sort()
labels.sort()


# In[6]:


print(np.size(images))
print(np.size(labels))


# In[7]:


def get_image(path):
    
    image = plt.imread(path)
    
    return(np.asarray(image[:320,:1152]))


# In[8]:


def get_label(path):

    label = plt.imread(path, 0)
    
    return(np.asarray(label[:320,:1152]))


# In[9]:


def rgbto2(lbl):
    w,h = lbl.shape[:2]
    out = np.zeros([w,h],dtype=np.int8)
    for i in range(w):
        for j in range(h):
            if(lbl[i,j,2] == 255):
                out[i,j] = 1
    return out


# In[10]:


from sklearn.model_selection import train_test_split

test_size = 0.20
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)


# In[11]:


x_test = []

for i in images_test:
    x_test.append(get_image(i))

x_test = np.asarray(x_test)


# In[12]:


x_test.shape


# In[48]:


from models.Unet import unet

model = unet(input_size = (320,1152,3), n_classes=2)

print("Model summary:")
# model.summary()

# In[ ]:
from keras import optimizers

#model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)
loss = 'binary_crossentropy'
metrics = ['accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, loss, metrics))

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

model.load_weights('weights/egolane/2018-11-19 08-11-09.hdf5')
# model.load_weights('weights/egolane/2018-11-19 12-22-39.hdf5')
# model.load_weights('weights/road/2018-11-20 08-32-15.hdf5')


# In[49]:


y_pred = model.predict(x_test, batch_size=1, verbose=1)

np.save("results/egolane_on_road_ds", y_pred)
