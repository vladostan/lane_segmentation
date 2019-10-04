# -*- coding: utf-8 -*-

# In[1]:
import os
import numpy as np
from PIL import Image
import time
from glob import glob

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

# In[2]:
def get_image(path):
    
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    
    return img

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

def custom_generator(images_path, labels_path, preprocessing_fn = None, doaug = False, batch_size = 1, validation = False):
    
    i = 0
    
    while True:
        
        if validation or not doaug:
	        x_batch = np.zeros((batch_size, 256, 640, 3))
	        y_batch = np.zeros((batch_size, 256, 640))
        else:
            x_batch = np.zeros((2*batch_size, 256, 640, 3))
            y_batch = np.zeros((2*batch_size, 256, 640))
        
        for b in range(batch_size):
            
            if i == len(images_path):
                i = 0
                
            x = get_image(images_path[i])
            y = get_image(labels_path[i])
            
            if validation or not doaug:
                x_batch[b] = x
                y_batch[b] = y
            else:
                x2 = augment(x)
                x_batch[2*b] = x
                x_batch[2*b+1] = x2
                y_batch[2*b] = y
                y_batch[2*b+1] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
    
        yield (x_batch, y_batch)

# In[4]:
from segmentation_models import Unet, FPN, Linknet, PSPNet

num_classes = 3
input_shape = (256,640,3)

backbone = 'resnet18'

model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')

model.summary()

# In[]
from sklearn.model_selection import train_test_split

test_size = 0
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)
images_train = images_train[:10]

print(len(images_train))

# In[ ]:
from segmentation_models.backbones import get_preprocessing

batch_size = 1

preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(images_path = images_train, 
                             labels_path = labels_train, 
                             preprocessing_fn = preprocessing_fn, 
                             doaug = False,
                             batch_size = batch_size)

# In[ ]:
from keras import optimizers
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

losses = ['categorical_crossentropy']
metrics = ['categorical_accuracy']

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, losses, metrics))

model.compile(optimizer = optimizer, loss = losses, metrics = metrics)

# In[ ]:
from keras import callbacks
from callbacks import TelegramCallback

#model_checkpoint = callbacks.ModelCheckpoint('weights/innopolis/{}.hdf5'.format(loggername), monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)
#tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 4, verbose = 1, min_lr = 1e-7)
early_stopper = callbacks.EarlyStopping(monitor='loss', patience = 10, verbose = 1)

config = {
    'token': '720029625:AAGG5aS46wOliEIs0HmUFgg8koN_ScI3AIY',   # paste your bot token
    'telegram_id': 218977821,                                   # paste your telegram_id
}
tg_callback = TelegramCallback(config)

clbacks = [reduce_lr, early_stopper, tg_callback]

if log:
    csv_logger = callbacks.CSVLogger('logs/innopolis/{}.log'.format(loggername))
    clbacks.append(csv_logger)

print("Callbacks: {}\n".format(clbacks))

# In[ ]:
steps_per_epoch = len(images_train)//batch_size
epochs = 10
verbose = 2

print("Steps per epoch: {}".format(steps_per_epoch))

print("Starting training...\n")
history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = verbose,
    callbacks = clbacks
)
print("Finished training\n")
            