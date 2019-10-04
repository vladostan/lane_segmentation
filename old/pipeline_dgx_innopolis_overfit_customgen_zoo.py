# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime

# In[]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]
log = True

# Get the date and time
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

# Print stdout to file
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/innopolis/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

if log:
    sys.stdout = Logger()

#sys.stdout = open('logs/{}'.format(loggername), 'w')

print('Date and time: {}\n'.format(loggername))

# READ IMAGES AND MASKS
# In[2]:
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
from PIL import Image

def get_image(path):
    
    img = Image.open(path)
    img = img.resize((640,256))
    img = np.array(img)
    
    return img

# In[]
# Visualise
visualize = False

if visualize:
    i = 17
    x = get_image(images[i])
    y = get_image(labels[i])
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].imshow(x)
    axes[1].imshow(y, cmap = 'gray')
    fig.tight_layout()

print("Image dtype:{}".format(get_image(images[0]).dtype))
print("Label dtype:{}\n".format(get_image(labels[0]).dtype))

# # Prepare for training
# In[ ]:
from sklearn.model_selection import train_test_split

test_size = 0
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)

print(len(images_train))
print(len(labels_train))
print(len(images_test))
print(len(labels_test))

# In[]: Class weights count
num_classes = 3
cw = {'background':0, 'direct':0, 'alternative':0}
class_weight_counting = False

def classcount(lbl, num_classes=num_classes):
    w,h = lbl.shape
    n = np.zeros(num_classes-1, dtype=np.int32)
    for i in range(w):
        for j in range(h):
            for c in range(num_classes-1):
                if(lbl[i,j] == c):
                    n[c] += 1
    return n

if class_weight_counting:
    for cntr,i in enumerate(labels_train):
        n = classcount(get_image(i))
        cw['background'] += n[0]
        cw['direct'] += n[1]
        cw['alternative'] += 256*640-n[0]-n[1]
        print(cw)
        print(cntr)
        print(sum(cw.values()) == (cntr+1)*256*640)
    
# In[]: Class weighting
#class_weights = np.median(np.asarray(list(cw.values()))/sum(cw.values()))/(np.asarray(list(cw.values()))/sum(cw.values()))
class_weights = np.array([0.16541089, 1.,         1.42351978]) # 0 split with random_state = 1
print("Class weights: {}\n".format(class_weights))

# In[]: AUGMENTATIONS
doaug = False

if doaug:
    from albumentations import (
        OneOf,
        Blur,
        RandomGamma,
        HueSaturationValue,
        RGBShift,
        RandomBrightness,
        RandomContrast,
        MedianBlur,
        CLAHE
    )
    
    aug = OneOf([
            Blur(blur_limit=5, p=1.),
            RandomGamma(gamma_limit=(50, 150), p=1.),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
            RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
            RandomBrightness(limit=.25, p=1.),
            RandomContrast(limit=.25, p=1.),
            MedianBlur(blur_limit=5, p=1.),
            CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
            ], p=1.)
    
    def augment(image, aug=aug):
    
        augmented = aug(image=image)
        image_augmented = augmented['image']
        
        return image_augmented

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
           
# In[ ]:
from segmentation_models.backbones import get_preprocessing

batch_size = 1

#images_train = images_train[:50]

backbone = 'resnet18'

preprocessing_fn = get_preprocessing(backbone)

train_gen = custom_generator(images_path = images_train, 
                             labels_path = labels_train, 
                             preprocessing_fn = preprocessing_fn, 
                             doaug = doaug,
                             batch_size = batch_size)

if visualize:
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(15, 10)
    axes[0,0].imshow(train_gen[0][0])
    axes[0,1].imshow(train_gen[1][0,:,:,0], cmap='gray')
    axes[1,0].imshow(train_gen[1][0,:,:,1], cmap='gray')
    axes[1,1].imshow(train_gen[1][0,:,:,2], cmap='gray')
    fig.tight_layout()

# In[ ]:
# # Define model
# # U-Net
from segmentation_models import Unet, FPN, Linknet, PSPNet

#model = Unet(backbone_name=backbone, input_shape=(256, 640, 3), classes=3, activation='softmax')
#model = FPN(backbone_name=backbone, input_shape=(256, 640, 3), classes=3, activation='softmax')
model = Linknet(backbone_name=backbone, input_shape=(256, 640, 3), classes=3, activation='softmax')
#model = PSPNet(backbone_name='resnet34', input_shape=(256, 640, 3), classes=3, activation='softmax')

print("Model summary:")
model.summary()

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

#model.compile(optimizer = Adam(lr=learning_rate, epsilon=1e-8, decay=1e-6), sample_weight_mode = "temporal",
#              loss = losses, metrics = metrics)
#model.compile(optimizer = Adam(lr=learning_rate, epsilon=1e-8, decay=1e-6), loss = losses, metrics = metrics)
model.compile(optimizer = optimizer, loss = losses, metrics = metrics)


# In[ ]:
from keras import callbacks

model_checkpoint = callbacks.ModelCheckpoint('weights/innopolis/{}.hdf5'.format(loggername), monitor = 'loss', verbose = 1, save_best_only = True, save_weights_only = True)
#tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor = 0.5, patience = 4, verbose = 1, min_lr = 1e-7)
early_stopper = callbacks.EarlyStopping(monitor='loss', patience = 10, verbose = 1)
clbacks = [model_checkpoint, reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/innopolis/{}.log'.format(loggername))
    clbacks.append(csv_logger)

print("Callbacks: {}\n".format(clbacks))


# In[ ]:
steps_per_epoch = len(images_train)//batch_size
validation_steps = len(images_test)//batch_size
epochs = 1000
verbose = 1

print("Steps per epoch: {}".format(steps_per_epoch))
print("Validation steps: {}\n".format(validation_steps))

print("Starting training...\n")
history = model.fit_generator(
    generator = train_gen,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = verbose,
    callbacks = clbacks,
    class_weight = class_weights
)
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))

# In[]:
#y = model.predict(np.expand_dims(get_image(images_train[4]), axis=0))
#out = y.squeeze()
#mask = np.argmax(out, axis=-1)
#plt.imshow(mask)
