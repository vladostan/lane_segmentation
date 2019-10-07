# coding: utf-8

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# In[]: Imports
import json
import pickle
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import datetime
import sys
from keras.utils import to_categorical
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Linknet
from keras import optimizers, callbacks
from losses import dice_coef_multiclass_loss
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

# In[]: Parameters
log = True
verbose = 2
aug = True

use_class_weights = True
if use_class_weights:
    class_weights_train = np.array([0.21809305, 1., 1.64836766])

visualize = False

num_classes = 3

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'

load_weights = True
if load_weights:
    weights = "2019-10-04 13-50-27"

random_state = 28

batch_size_init = 32
batch_factor = 2
batch_size = batch_size_init//batch_factor

val_size = 0.2
test_size = 0.

monitor = 'val_' if val_size > 0 else ''

# In[]: Logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open('logs/{}.txt'.format(loggername), 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass    

if log:
    now = datetime.datetime.now()
    loggername = str(now).split(".")[0]
    loggername = loggername.replace(":","-")
    sys.stdout = Logger()

print('Date and time: {}\n'.format(loggername))
print("LOG: {}\nAUG: {}\nUSE CLASS WEIGHTS: {}\nNUM CLASSES: {}\nRESIZE: {}\nINPUT SHAPE: {}\nBACKBONE: {}\nRANDOM STATE: {}\nBATCH SIZE: {}".format(log, aug, use_class_weights, num_classes, resize, input_shape, backbone, random_state, batch_size))
print("TRAIN:VAL:TEST SPLIT = {}:{}:{}\n".format(1-val_size-test_size, val_size, test_size))

# In[]: Dataset
dataset_dir = "../../../colddata/datasets/supervisely/kamaz/kisi/"
subdirs = ["2019-04-24", "2019-05-08", "2019-05-15", "2019-05-20", "2019-05-22"]

obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print("DATASETS USED: {}".format(subdirs))
print("TOTAL FILES COUNT: {}\n".format(len(ann_files)))

# In[]:
def get_image(path, label = False, resize = False):
    img = Image.open(path)
    if resize:
        img = img.resize(input_shape[:2][::-1])
    img = np.array(img) 
    if label:
        return img[..., 0]
    return img
    
img_path = ann_files[0].replace('/ann/', '/img/').split('.json')[0]
label_path = ann_files[0].replace('/ann/', '/masks_machine/').split('.json')[0]

print("Images dtype: {}".format(get_image(img_path).dtype))
print("Labels dtype: {}\n".format(get_image(label_path, label = True).dtype))
print("Images shape: {}".format(get_image(img_path, resize = True if resize else False).shape))
print("Labels shape: {}\n".format(get_image(label_path, label = True, resize = True if resize else False).shape))

# In[]: Visualise
if visualize:
    i = 28
    x = get_image(img_path, resize = True if resize else False)
    y = get_image(label_path, label = True, resize = True if resize else False)
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].imshow(x)
    axes[1].imshow(y)
    fig.tight_layout()

# In[]: Prepare for training
ann_files_train, ann_files_valtest = train_test_split(ann_files, test_size=val_size+test_size, random_state=random_state)
ann_files_val, ann_files_test = train_test_split(ann_files_valtest, test_size=test_size/(test_size+val_size+1e-8)-1e-8, random_state=random_state)
del(ann_files_valtest)

print("Training files count: {}".format(len(ann_files_train)))
print("Validation files count: {}".format(len(ann_files_val)))
print("Testing files count: {}\n".format(len(ann_files_test)))

if log:
    with open('pickles/{}.pickle'.format(loggername), 'wb') as f:
        pickle.dump(ann_files_train, f)
        pickle.dump(ann_files_val, f)
        pickle.dump(ann_files_test, f)
        
# In[]: Class weight counting
def cw_count(ann_files, mode):
    print("Class weight calculation for {} started".format(mode))
    cw_seg = np.zeros(num_classes, dtype=np.int64)

    for af in tqdm(ann_files):
        label_path = af.replace('/ann/', '/masks_machine/').split('.json')[0]
        l = get_image(label_path, label = True, resize = True if resize else False)
        
        for i in range(num_classes):
            cw_seg[i] += np.count_nonzero(l==i)
        
    if sum(cw_seg) == len(ann_files)*input_shape[0]*input_shape[1]:
        print("Class weights for {} calculated successfully:".format(mode))
        class_weights = np.median(cw_seg/sum(cw_seg))/(cw_seg/sum(cw_seg))
        for cntr,i in enumerate(class_weights):
            print("Class {} = {}".format(cntr, i))
    else:
        print("Class weights calculation for {} failed".format(mode))
        
    return class_weights
        
if use_class_weights:
    try:
        class_weights_train
    except NameError:
        class_weights_train = cw_count(ann_files_train, "training")
        if val_size > 0:
            class_weights_val = cw_count(ann_files_val, "validation")
        if test_size > 0:
            class_weights_test = cw_count(ann_files_test, "testing")

# In[]:
def augment(image):
    
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

    augmented = aug(image=image)
    image_augmented = augmented['image']
    
    return image_augmented
    
# In[]: 
def train_generator(files, preprocessing_fn = None, aug = False, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_factor*batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y_batch = np.zeros((batch_factor*batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
            y = get_image(ann_files[i].replace('/ann/', '/masks_machine/').split('.json')[0], label=True, resize = True if resize else False)
            
            x_batch[batch_factor*b] = x
            y_batch[batch_factor*b] = y
            
            if aug == 1:
                x_batch[batch_factor*b+1] = augment(x)
                y_batch[batch_factor*b+1] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
            
        yield (x_batch, y_batch)
        
def val_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        y_batch = np.zeros((batch_size, input_shape[0], input_shape[1]))
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
            y = get_image(ann_files[i].replace('/ann/', '/masks_machine/').split('.json')[0], label=True, resize = True if resize else False)
            
            x_batch[b] = x
            y_batch[b] = y
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
        y_batch = to_categorical(y_batch, num_classes=num_classes)
        y_batch = y_batch.astype('int64')
            
        yield (x_batch, y_batch)
    
# In[]:
preprocessing_fn = get_preprocessing(backbone)

train_gen = train_generator(files = ann_files_train, 
                             preprocessing_fn = preprocessing_fn, 
                             aug = aug,
                             batch_size = batch_size)

if val_size > 0:
    val_gen = val_generator(files = ann_files_val, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size_init)

# In[]: Bottleneck
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')

if load_weights:
    print("\nLoading model weights: {}".format(weights))
    model.load_weights('weights/' + weights + '.hdf5')
    
print("\nModel summary:")
model.summary()

# In[]: 
loss = [dice_coef_multiclass_loss]
metrics = ['categorical_accuracy']

learning_rate = 6.25e-06
optimizer = optimizers.Adam(lr = learning_rate)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("\nOptimizer: {}\nLearning rate: {}\nLoss: {}\nMetrics: {}\n".format(optimizer, learning_rate, loss, metrics))

# In[]:
reduce_lr = callbacks.ReduceLROnPlateau(monitor = monitor+'loss', factor = 0.5, patience = 5, verbose = 1, min_lr = 1e-8)
early_stopper = callbacks.EarlyStopping(monitor = monitor+'loss', patience = 10, verbose = 1)

clbacks = [reduce_lr, early_stopper]

if log:
    csv_logger = callbacks.CSVLogger('logs/{}.log'.format(loggername))
    model_checkpoint = callbacks.ModelCheckpoint('weights/{}.hdf5'.format(loggername), monitor = monitor+'loss', verbose = 1, save_best_only = True, save_weights_only = True)
    clbacks.append(csv_logger)
    clbacks.append(model_checkpoint)

print("Callbacks used:")
for c in clbacks:
    print("{}".format(c))

# In[]: 
steps_per_epoch = len(ann_files_train)//batch_size
epochs = 1000

validation_data = None
validation_steps = None
class_weight = None

if use_class_weights:
    class_weight = class_weights_train

print("Steps per epoch: {}".format(steps_per_epoch))
if val_size > 0:
    validation_data = val_gen
    validation_steps = len(ann_files_val)//batch_size_init
    print("Validation steps: {}\n".format(validation_steps))

print("Starting training...\n")
history = model.fit_generator(
        generator = train_gen,
        steps_per_epoch = steps_per_epoch,
        epochs = epochs,
        verbose = verbose,
        callbacks = clbacks,
        validation_data = val_gen,
        validation_steps = validation_steps,
        class_weight = class_weight
)
print("Finished training\n")

now = datetime.datetime.now()
loggername = str(now).split(".")[0]
loggername = loggername.replace(":","-")
print('Date and time: {}\n'.format(loggername))