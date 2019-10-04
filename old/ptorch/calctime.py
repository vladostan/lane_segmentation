# -*- coding: utf-8 -*-

# In[1]:
import os
import sys

import numpy as np
import torch

from PIL import Image

from unet import UNet
from network import U_Net

import time
from glob import glob

# In[]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

# In[2]:
def get_image(path):
    
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    
    return img

# In[2]:
def normalize(x):
    return np.float32(x/255.)

# In[3]:
def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

# In[4]:
net = UNet(n_channels=3, n_classes=1)
#net = U_Net(img_ch=3, output_ch=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of learnable parameters in the model: {}".format(count_parameters(net)))

if torch.cuda.is_available():
    gpu = True
else:
    gpu = False
        
if gpu:
#    weights = "checkpoints/oldUnet.pth"
#    net.load_state_dict(torch.load(weights))
    net.cuda() 

else:
#    net.load_state_dict(torch.load(weights, map_location='cpu'))
    net.cpu()

net.eval()

# In[7]:
PATH = os.path.abspath('../data')

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

test_size = 0.2
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)

#images_test = images_test[:50]
print(len(images_test))

# In[56]:
with torch.no_grad():
    
    start_time = time.time()
    
    for i, im in enumerate(images_test):
        
#        img = get_image(im)
#        img = hwc_to_chw(img)
#        img = normalize(img)
        img = np.zeros([3, 256, 640], dtype = np.float32)
        
        X = torch.from_numpy(img).unsqueeze(0)

        if gpu:
            X = X.cuda()

        output = net(X)
        print(i)
                
#        mask_np = output.squeeze().cpu().numpy()
        
print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(len(images_test)/(time.time() - start_time)))
