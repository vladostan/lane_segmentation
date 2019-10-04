# -*- coding: utf-8 -*-

# In[1]:
import os
import math
import numpy as np
from PIL import Image
import time
from glob import glob
from caffe2.python import workspace

# In[3]:
#Crop bottom center
def crop(x, crop_h, crop_w):
    h,w = x.shape[:2]
    h_new = h-crop_h, h
    w_new = int(math.floor((w-crop_w)/2)), int(math.floor(w-(w-crop_w)/2))
    return x[h_new[0]:h_new[1], w_new[0]:w_new[1], :]
    
# In[2]:
def normalize(x):
    return np.float32(x/255.)

# In[3]:
def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

# In[7]:
PATH = os.path.abspath('.')

SOURCE_IMAGES = [os.path.join(PATH, "../data/images/um")]

SOURCE_IMAGES = [os.path.join(PATH, "../data/images/um"), 
                 os.path.join(PATH, "../data/images/umm"), 
                 os.path.join(PATH, "../data/images/uu")]

images = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

# In[4]:
init_net = "old_init_net.pb"
predict_net = "old_predict_net.pb"

# run caffe2 inference
with open('old_init_net.pb', 'rb') as f:
    init_net = f.read()
with open('old_predict_net.pb', 'rb') as f:
    predict_net = f.read()

predictor = workspace.Predictor(init_net, predict_net)

# In[56]:
crop_h = 320 
crop_w = 1152

crop_h = 256 
crop_w = 256

images = images[:10]

# In[56]:
start_time = time.time()

for im in images:
    
    img = Image.open(im)
    img = np.array(img)
    img = crop(img, crop_h=crop_h, crop_w=crop_w)
    img = hwc_to_chw(img)
    img = normalize(img)
    img = np.expand_dims(img, axis = 0)
    
    output = predictor.run([img])[0]
    
    print("Output for image {} predicted".format(im))    

print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(len(images)/(time.time() - start_time)))

del(predictor)
