# -*- coding: utf-8 -*-

# In[1]:
import os
import math
import numpy as np
from PIL import Image
import time
from glob import glob
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

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

workspace.ResetWorkspace();
#device_opts = core.DeviceOption(caffe2_pb2.CPU, 0)
device_opts = core.DeviceOption(caffe2_pb2.CUDA, 0)

init_def = caffe2_pb2.NetDef()
with open(init_net, 'rb') as f:
    init_def.ParseFromString(f.read())
    init_def.device_option.CopyFrom(device_opts)
    workspace.RunNetOnce(init_def.SerializeToString())

net_def = caffe2_pb2.NetDef()
with open(predict_net, 'rb') as f:
    net_def.ParseFromString(f.read())
    net_def.device_option.CopyFrom(device_opts)
    workspace.CreateNet(net_def.SerializeToString())

name = net_def.name
out_name = net_def.external_output[-1];
in_name = net_def.external_input[0]

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
    
    workspace.FeedBlob(in_name, img, device_opts)
    workspace.RunNet(name, 1)
    results = workspace.FetchBlob(out_name)
        
    print("Output for image {} predicted".format(im))    

print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(len(images)/(time.time() - start_time)))

del(predictor)
