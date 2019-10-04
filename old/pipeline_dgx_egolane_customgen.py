# coding: utf-8

# In[1]:
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime

# In[]

# Get the date and time
now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

# Print stdout to file
import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/egolane/{}.txt'.format(loggername), 'w')

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

# READ IMAGES AND MASKS
# In[2]:
PATH = os.path.abspath('data')

SOURCE_IMAGES = [os.path.join(PATH, "images/um")]

images = []
labels = []

for si in SOURCE_IMAGES:
    images.extend(glob(os.path.join(si, "*.png")))
    labels.extend(glob(os.path.join(si.replace("images/","labels/egolane/"), "*.png")))
    
print("Datasets used: {}\n".format(SOURCE_IMAGES))

images.sort()
labels.sort()

print(len(images))
print(len(labels))

# In[]
def get_image(path):
    
    image = plt.imread(path, 0)
    
    return(np.asarray(image[:320,:1152]))
    
print("Image dtype:{}\n".format(get_image(images[0]).dtype))

def get_label(path):

    label = plt.imread(path, 0)
    
    return(np.asarray(label[:320,:1152]))
    
print(get_image(images[83]).shape)
print(get_image(labels[84]).shape)

# In[]
def rgbto2(lbl):
    w,h = lbl.shape[:2]
    out = np.zeros([w,h],dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            if(lbl[i,j,2] == 255):
                out[i,j] = 1
    return out

# # Prepare for training
# In[ ]:

from sklearn.model_selection import train_test_split

test_size = 0.25
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)

print(len(images_train))
print(len(labels_train))
print(len(images_test))
print(len(labels_test))

# In[]: Class weighting
cw = {'background':0, 'egolane':0}

def classcount(lbl):
    w,h = lbl.shape[:2]
    n = 0
    for i in range(w):
        for j in range(h):
            if(lbl[i,j,2] == 255):
                n += 1
    return n

for i in labels_train:
    n = classcount(get_label(i))
    cw['background'] += (368640 - n) #320*1152
    cw['egolane'] += n
    
class_weights = np.median(np.asarray(list(cw.values()))/sum(cw.values()))/(np.asarray(list(cw.values()))/sum(cw.values()))
class_weights_dict = dict(enumerate(class_weights))

# In[]: AUGMENTATIONS
from albumentations import (
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma
)

def augment(image, mask):
    
    aug = OneOf([
            CLAHE(p=1., clip_limit=4.),
            RandomContrast(p=1., limit=0.25),
            RandomGamma(p=1., gamma_limit=(50,200))
            ], p=1)

    augmented = aug(image=image, mask=mask)

    image_heavy = augmented['image']
    mask_heavy = augmented['mask']
    
    return image_heavy, mask_heavy

# In[]: CUSTOM GENERATORS
from keras.utils import to_categorical

def custom_generator(images_path, labels_path, batch_size = 1, validation = False):
    
    while True:
        ids = np.random.randint(0, len(images_path), batch_size)
        
        image_batch = np.take(images_path, ids)
        label_batch = np.take(labels_path, ids)
        
        batch_input = np.zeros([batch_size, 320, 1152, 3], dtype='uint8')
        batch_output = np.zeros([batch_size, 320, 1152], dtype='uint8')

        # READ Images and masks:
        for i in range(len(image_batch)):
            inp = get_image(image_batch[i])
            batch_input[i] = inp
            outp = rgbto2(get_label(label_batch[i]))
            batch_output[i] = outp
            
        if validation:
            out_x = np.float32(batch_input/255.)
            out_y = to_categorical(batch_output, num_classes=2)
        else:
            out_x = np.zeros_like(batch_input)
            out_y = np.zeros_like(batch_output)
        
            # AUGMENT
            for i in range(len(batch_input)):
                out_x[i], out_y[i] = augment(batch_input[i], batch_output[i])
                
            out_x = np.float32(out_x/255.)
            out_y = to_categorical(out_y, num_classes=2)

        out_y = out_y.astype('int8')
        
        yield(out_x, out_y)      
#        return(out_x, out_y)
        
# In[ ]:
batch_size = 1

train_gen = custom_generator(images_path=images_train, labels_path=labels_train, batch_size=1)
val_gen = custom_generator(images_path=images_test, labels_path=labels_test, batch_size=1, validation = True)

visualize = False

if visualize:
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.set_size_inches(15, 10)
    axes[0].imshow(train_gen[0][0])
    axes[1].imshow(train_gen[1][0,:,:,0], cmap='gray')
    axes[2].imshow(train_gen[1][0,:,:,1], cmap='gray')
    fig.tight_layout()

#print(train_gen[0].shape)
#print(train_gen[1].shape)
#print(val_gen[0].shape)
#print(val_gen[1].shape)

# In[ ]:
# # Define model
# # U-Net
from models.Unet import unet

model = unet(input_size = (320,1152,3), n_classes=2)

print("Model summary:")
model.summary()

# In[ ]:
from keras import optimizers
from keras import backend as K
from metrics import dice_loss, dice

learning_rate = 5e-5
optimizer = optimizers.Adam(lr = learning_rate)
#metrics = ['accuracy']
#loss = 'binary_crossentropy'
metrics = [dice]
loss = [dice_loss]

print("Optimizer: {}, learning rate: {}, loss: {}, metrics: {}\n".format(optimizer, learning_rate, loss, metrics))

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
#model.load_weights('weights/egolane/2018-11-19 12-22-39.hdf5')
model.load_weights('weights/egolane/2018-11-23 10-56-16.hdf5')

# In[ ]:
import tensorflow as tf

def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#K.set_session(get_tf_session())

# In[ ]:
from keras import callbacks

model_checkpoint = callbacks.ModelCheckpoint('weights/egolane/{}.hdf5'.format(loggername), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
#tensor_board = callbacks.TensorBoard(log_dir='./tblogs')
reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                              patience=2, verbose = 1, min_lr=1e-6)
csv_logger = callbacks.CSVLogger('logs/egolane/{}.log'.format(loggername))
early_stopper = callbacks.EarlyStopping(monitor='loss', min_delta = 0.0001, patience = 4, verbose = 1)

callbacks = [model_checkpoint, reduce_lr, csv_logger, early_stopper]

print("Callbacks: {}\n".format(callbacks))

# In[ ]:
steps_per_epoch = len(images_train)//batch_size*2
validation_steps = len(images_test)//batch_size
epochs = 100
verbose = 2

print("Steps per epoch: {}".format(steps_per_epoch))
print("Validation steps: {}\n".format(validation_steps))

print("Starting training...\n")
history = model.fit_generator(
    train_gen,
    validation_data = val_gen,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    class_weight = class_weights,
    epochs = epochs,
    verbose = verbose,
    callbacks = callbacks
)
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))
