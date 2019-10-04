#!/usr/bin/env python
# coding: utf-8

# In[1]:
import random
import sys
import os
from optparse import OptionParser
import numpy as np
from PIL import Image

# In[2]:
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
# In[3]:
from eval import eval_net
from network import U_Net

# In[4]:
def get_ids(directory):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(directory))

# In[255]:
def load_img(ids, directory, suffix):
    for iid in ids:
        img = np.array(Image.open(directory + iid + suffix))
        yield img[:320,:1152]

# In[256]:
def load_mask(ids, directory, suffix):
    for iid in ids:
        msk = np.array(Image.open(directory + iid + suffix)).astype('float32')
        yield msk[:320,:1152]

# In[257]:
def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

# In[258]:
def normalize(x):
    return np.float32(x/255.)

# In[260]:
def get_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    imgs = load_img(ids, dir_img, '.png')

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = load_mask(ids, dir_mask, '.png')

    return zip(imgs_normalized, masks)

# In[263]:
def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

# In[264]:
def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

# In[19]:
epochs = 100
batch_size = 1
lr = 0.0001
val_percent = 0.25
save_cp = True
gpu = False
load = False #Load file model

# In[18]:
#net = UNet(n_channels=3, n_classes=1)
net = U_Net(img_ch=3, output_ch=1)

if gpu: 
    net.cuda()
    
if load:
    weights = "MODEL.pth"
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(weights))
    else:
        net.load_state_dict(torch.load(weights, map_location='cpu'))
    print('Model loaded from {}'.format(weights))

# In[20]:
dir_img = '../data/images/um/'
dir_mask = '../data/labels/egolane/um/'
dir_checkpoint = 'checkpoints/'

ids = get_ids(dir_img)
iddataset = split_train_val(ids, val_percent)
iddataset['train']

N_train = len(iddataset['train'])
N_val = len(iddataset['val'])

# In[35]:
print('''
Starting training:
Epochs: {}
Batch size: {}
Learning rate: {}
Training size: {}
Validation size: {}
Checkpoints: {}
CUDA: {}
'''.format(epochs, batch_size, lr, len(iddataset['train']),
       len(iddataset['val']), str(save_cp), str(gpu)))

# In[41]:
optimizer = optim.Adam(net.parameters(), lr=lr)

# In[42]:
criterion = nn.BCELoss()

# In[ ]:
for epoch in range(epochs):
    print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
    net.train()

    # reset the generators
    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

    epoch_loss = 0

    for i, b in enumerate(batch(train, batch_size)):
        imgs = np.array([j[0] for j in b])
        true_masks = np.array([j[1] for j in b])

        imgs = torch.from_numpy(imgs)
        true_masks = torch.from_numpy(true_masks)

        if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

        masks_pred = net(imgs)
        masks_probs_flat = masks_pred.view(-1)

        true_masks_flat = true_masks.view(-1)

        loss = criterion(masks_probs_flat, true_masks_flat)
        epoch_loss += loss.item()

        print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / float(N_train), loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

    if 1:
        val_dice = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))

    if save_cp:
        torch.save(net.state_dict(),
                   dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved !'.format(epoch + 1))
        
