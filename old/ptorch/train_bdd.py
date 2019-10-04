#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys
import os
import numpy as np
from PIL import Image
import datetime

# In[]

# Get the date and time
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

# Print stdout to file
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/driveable/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

#sys.stdout = open('logs/{}'.format(loggername), 'w')

print('Date and time: {}\n'.format(loggername))

# In[2]:
import torch
import torch.nn as nn
from torch import optim

# In[4]:
def get_ids(directory):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(directory))

# In[255]:
#def load_img(ids, directory, suffix):
#    for iid in ids:
#        
#        img = Image.open(directory + iid + suffix)
#        img = img.resize((640,360))
#        img = np.array(img)
#        yield img
        
def load_img2(iid, directory, suffix):
        
    img = Image.open(directory + iid + suffix)
    img = img.resize((640,360))
#    img = img.resize((320,180))

    img = np.array(img)
        
    return img

# In[256]:
#def load_mask(ids, directory, suffix):
#    for iid in ids:
#        
#        msk = Image.open(directory + iid + "_drivable_id" + suffix)
#        msk = msk.resize((640,360))
#        msk = np.array(msk).astype('float32')
#        yield msk
        
def load_mask2(iid, directory, suffix):
        
    msk = Image.open(directory + iid + "_drivable_id" + suffix)
    msk = msk.resize((640,360))
#    msk = msk.resize((320,180))

    msk = np.array(msk).astype('float32')
    return msk

# In[257]:
def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

# In[258]:
def normalize(x):
    return np.float32(x/255.)

# In[260]:
#def get_imgs_and_masks(ids, dir_img, dir_mask):
#    """Return all the couples (img, mask)"""
#
#    imgs = load_img(ids, dir_img, '.jpg')
#
#    # need to transform from HWC to CHW
#    imgs_switched = map(hwc_to_chw, imgs)
#    imgs_normalized = map(normalize, imgs_switched)
#
#    masks = load_mask(ids, dir_mask, '.png')
#
#    return zip(imgs_normalized, masks)

# In[263]:
#def batch(iterable, batch_size):
#    """Yields lists by batch"""
#    b = []
#    for i, t in enumerate(iterable):
#        b.append(t)
#        if (i + 1) % batch_size == 0:
#            yield b
#            b = []
#
#    if len(b) > 0:
#        yield b

# In[19]:
epochs = 100
batch_size = 1
lr = 0.00005
save_cp = True
gpu = True
load = True

# In[18]:
from unet import UNet_multiclass

net = UNet_multiclass(n_channels=3, n_classes=3)

if gpu: 
    net.cuda()
    
if load:
    weights = "checkpoints/driveable/BDD_2_CP30.pth"
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(weights))
    else:
        net.load_state_dict(torch.load(weights, map_location='cpu'))
    print('Model loaded from {}'.format(weights))

# In[20]:
path = "/home/kenny/Desktop/bdd"
path = "/home/datasets/bdd/bdd100k"

dir_img_train = path + '/images/100k/train/'
dir_img_val = path + '/images/100k/val/'

dir_mask_train = path + '/drivable_maps/labels/train/'
dir_mask_val = path + '/drivable_maps/labels/val/'

dir_checkpoint = 'checkpoints/driveable/'

ids_train = get_ids(dir_img_train)
ids_val = get_ids(dir_img_val)

iddataset = dict()

iddataset['train'] = list(ids_train)
iddataset['val'] = list(ids_val)

#iddataset['train'] = iddataset['train'][:40]
#iddataset['val'] = iddataset['val'][:40]

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
CUDA: {}\n
'''.format(epochs, batch_size, lr, len(iddataset['train']),
       len(iddataset['val']), str(save_cp), str(gpu)))

# In[41]:
optimizer = optim.Adam(net.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True, threshold=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True, threshold=1e-3)
criterion = nn.CrossEntropyLoss()

print("Optimizer: {}\n".format(optimizer))
print("ReduceLROnPlateau: mode:{} factor:{} patience:{} verbose:{} threshold:{}\n".format(scheduler.mode, scheduler.factor, scheduler.patience, scheduler.verbose, scheduler.threshold))
print("Loss: {}\n".format(criterion))

# In[ ]:
#########################
#########################
#########################
train_losses = []
val_losses = []

best_train_loss = float('inf')
best_val_loss = float('inf')

best_train_epoch = 0
best_val_epoch = 0

for epoch in range(epochs):
    print('Starting epoch {}/{}.'.format(epoch, epochs))

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        
        print(phase)
        
        running_loss = 0.0

        if phase == 'train':
            net.train(True)  # Set model to training mode
            for i in range(N_train):
                                
                img = load_img2(iddataset['train'][i], dir_img_train, '.jpg')
                img = hwc_to_chw(img)
                img = normalize(img)
                img = np.expand_dims(img, axis = 0)                

                msk = load_mask2(iddataset['train'][i], dir_mask_train, '.png')
                msk = np.expand_dims(msk, axis = 0)
	
                img = torch.from_numpy(img)
                true_mask = torch.from_numpy(msk)
        
                if gpu:
                    img = img.cuda()
                    true_mask = true_mask.cuda()
        
                mask_pred = net(img)
                #mask_probs_flat = mask_pred.view(-1)
        
                #true_mask_flat = true_mask.view(-1)
        
                #loss = criterion(mask_probs_flat, true_mask_flat)
                loss = criterion(mask_pred, true_mask.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
#                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / float(N_train), loss.item()))
                
                running_loss += loss.item()
                
            epoch_loss = running_loss/N_train
            train_losses.append(epoch_loss)
            
            if train_losses[epoch] < train_losses[epoch-1] and epoch > 0:
                print("Epoch finished! Train loss improved from {} to {}".format(train_losses[epoch-1], train_losses[epoch]))
            elif train_losses[epoch] > train_losses[epoch-1] and epoch > 0:
                print("Epoch finished! Train loss became worse from {} to {}".format(train_losses[epoch-1], train_losses[epoch]))
            else:
                print('Epoch finished! {} loss: {}'.format(phase, epoch_loss))
                
            if train_losses[epoch] < best_train_loss:
                best_train_loss = train_losses[epoch]
                best_train_epoch = epoch
            print("Best train loss: {} at epoch: {}".format(best_train_loss, best_train_epoch))
                
            scheduler.step(epoch_loss)
            #print("Learning rate: {}".format(optimizer.defaults['lr']))
                
        else:
            net.train(False)  # Set model to evaluate mode
            for i in range(N_val):
                                
                img = load_img2(iddataset['val'][i], dir_img_val, '.jpg')
                img = hwc_to_chw(img)
                img = normalize(img)
                img = np.expand_dims(img, axis = 0)                
                
                msk = load_mask2(iddataset['val'][i], dir_mask_val, '.png')
                msk = np.expand_dims(msk, axis = 0)

                img = torch.from_numpy(img)
                true_mask = torch.from_numpy(msk)
        
                if gpu:
                    img = img.cuda()
                    true_mask = true_mask.cuda()
        
                mask_pred = net(img)
                #mask_probs_flat = mask_pred.view(-1)
        
                #true_mask_flat = true_mask.view(-1)
        
                #loss = criterion(mask_probs_flat, true_mask_flat)
                loss = criterion(mask_pred, true_mask.long())                

                optimizer.zero_grad()
#                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / float(N_val), loss.item()))
                
                running_loss += loss.item()
                
            epoch_loss = running_loss/N_val
            val_losses.append(epoch_loss)
            
            if val_losses[epoch] < val_losses[epoch-1] and epoch > 0:
                print("Epoch finished! Val loss improved from {} to {}".format(val_losses[epoch-1], val_losses[epoch]))
            elif val_losses[epoch] > val_losses[epoch-1] and epoch > 0:
                print("Epoch finished! Val loss became worse from {} to {}".format(val_losses[epoch-1], val_losses[epoch]))
            else:
                print('Epoch finished! {} Loss: {}'.format(phase, epoch_loss))
                
            if val_losses[epoch] < best_val_loss:
                best_val_loss = val_losses[epoch]
                best_val_epoch = epoch
            print("Best val loss: {} at epoch: {}".format(best_val_loss, best_val_epoch))
                    
        
    if save_cp:
        torch.save(net.state_dict(),
                   dir_checkpoint + 'BDD_3_CP{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved !'.format(epoch + 1))
        
print('Training finished\n')
print('Date and time: {}\n'.format(str(datetime.datetime.now()).split(".")[0].replace(":","-")))
#for epoch in range(epochs):
#    print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
#    net.train()
#
#    # reset the generators
#    train = get_imgs_and_masks(iddataset['train'], dir_img_train, dir_mask_train)
#    val = get_imgs_and_masks(iddataset['val'], dir_img_val, dir_mask_val)
#
#    epoch_loss = 0
#
#    for i, b in enumerate(batch(train, batch_size)):
#        imgs = np.array([j[0] for j in b])
#        true_masks = np.array([j[1] for j in b])
#
#        imgs = torch.from_numpy(imgs)
#        true_masks = torch.from_numpy(true_masks)
#
#        if gpu:
#            imgs = imgs.cuda()
#            true_masks = true_masks.cuda()
#
#        masks_pred = net(imgs)
#        masks_probs_flat = masks_pred.view(-1)
#
#        true_masks_flat = true_masks.view(-1)
#
#        loss = criterion(masks_probs_flat, true_masks_flat)
#        epoch_loss += loss.item()
#
#        print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / float(N_train), loss.item()))
#
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#    print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
#
#    if 1:
#        val_dice = eval_net(net, val, gpu)
#        print('Validation Dice Coeff: {}'.format(val_dice))
#
#    if save_cp:
#        torch.save(net.state_dict(),
#                   dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
#        print('Checkpoint {} saved !'.format(epoch + 1))
#        
#    scheduler.step(val_dice)
    
##########################
##########################
##########################
#for epoch in range(epochs):
#    print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
#    
#    # reset the generators
#    train = get_imgs_and_masks(iddataset['train'], dir_img_train, dir_mask_train)
#    val = get_imgs_and_masks(iddataset['val'], dir_img_val, dir_mask_val)
#    
#    # Each epoch has a training and validation phase
#    for phase in ['train', 'val']:
#        
#        print(phase)
#        
#        running_loss = 0.0
#
#        if phase == 'train':
#            net.train(True)  # Set model to training mode
#            for i, b in enumerate(batch(train, batch_size)):
#                print(i)
#                imgs = np.array([j[0] for j in b])
#                true_masks = np.array([j[1] for j in b])
#        
#                imgs = torch.from_numpy(imgs)
#                true_masks = torch.from_numpy(true_masks)
#        
#                if gpu:
#                    imgs = imgs.cuda()
#                    true_masks = true_masks.cuda()
#        
#                masks_pred = net(imgs)
#                masks_probs_flat = masks_pred.view(-1)
#        
#                true_masks_flat = true_masks.view(-1)
#        
#                loss = criterion(masks_probs_flat, true_masks_flat)
#                optimizer.zero_grad()
#                loss.backward()
#                optimizer.step()
#                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / float(N_train), loss.item()))
#                
#                running_loss += loss.item()
#            epoch_loss = running_loss/N_train
#            print('Epoch finished! {} Loss: {}'.format(phase, epoch_loss / i))
#            scheduler.step(epoch_loss)
#                
#        else:
#            net.train(False)  # Set model to evaluate mode
#            for i, b in enumerate(batch(val, batch_size)):
#                imgs = np.array([j[0] for j in b])
#                true_masks = np.array([j[1] for j in b])
#        
#                imgs = torch.from_numpy(imgs)
#                true_masks = torch.from_numpy(true_masks)
#        
#                if gpu:
#                    imgs = imgs.cuda()
#                    true_masks = true_masks.cuda()
#        
#                masks_pred = net(imgs)
#                masks_probs_flat = masks_pred.view(-1)
#        
#                true_masks_flat = true_masks.view(-1)
#        
#                loss = criterion(masks_probs_flat, true_masks_flat)
#                optimizer.zero_grad()
#                print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / float(N_val), loss.item()))
#                
#                running_loss += loss.item()
#            epoch_loss = running_loss/N_val
#            print('Epoch finished! {} Loss: {}'.format(phase, epoch_loss / i))
#    
#        
#    if save_cp:
#        torch.save(net.state_dict(),
#                   dir_checkpoint + 'BDDCP{}.pth'.format(epoch + 1))
#        print('Checkpoint {} saved !'.format(epoch + 1))
