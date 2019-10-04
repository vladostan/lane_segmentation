# -*- coding: utf-8 -*-

# In[1]:
import os
import math
import numpy as np
from PIL import Image
import time
from glob import glob
from caffe2.python.onnx.backend import Caffe2Backend
import onnx
import torch

# In[3]:
#Crop bottom center
def crop(x, crop_h, crop_w):
    h,w = x.shape[:2]
    h_new = h-crop_h, h
    w_new = int(math.floor((w-crop_w)/2.)), int(math.floor(w-(w-crop_w)/2.))
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

#SOURCE_IMAGES = [os.path.join(PATH, "../data/images/um"), 
#                 os.path.join(PATH, "../data/images/umm"), 
#                 os.path.join(PATH, "../data/images/uu")]

images = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

# In[4]:
model = onnx.load("oldunet.onnx")
#prepared_backend = Caffe2Backend.prepare(model)
prepared_backend = Caffe2Backend.prepare(model, device='CUDA:0')

# In[56]:
crop_h = 320 
crop_w = 1152

#crop_h = 256 
#crop_w = 256

#images = images[:3]

# In[56]:
start_time = time.time()

for im in images:
    
    img = Image.open(im)
    img = np.array(img)
    
    img = crop(img, crop_h=crop_h, crop_w=crop_w)

    img = hwc_to_chw(img)
    img = normalize(img)
    img = np.expand_dims(img, axis = 0)
    img = torch.from_numpy(img)
    
    #img = torch.rand(1,3,320,1152)

    W = {model.graph.input[0].name: img.cpu().data.numpy()}
    c2_out = prepared_backend.run(W)[0]
        
    print("Output for image {} predicted".format(im))    

print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(len(images)/(time.time() - start_time)))

del(prepared_backend)
