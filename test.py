# -*- coding: utf-8 -*-

# In[]: Set GPU
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# In[]: Imports
import pickle
import json
from keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pylab as plt
from glob import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from segmentation_models.backbones import get_preprocessing
from segmentation_models import Linknet
from losses import dice_coef_multiclass_loss
from keras import optimizers
import cv2

# In[]: Parameters
visualize = True
save_results = False

num_classes = 3

resize = True
input_shape = (256, 640, 3) if resize else (512, 1280, 3)

backbone = 'resnet18'

random_state = 28
batch_size = 1

verbose = 1

weights = "2019-10-04 13-50-27"

# In[]:
dataset_dir = "../../../colddata/datasets/supervisely/kamaz/kisi/"
subdirs = ["2019-04-24", "2019-05-08", "2019-05-15", "2019-05-20", "2019-05-22"]

obj_class_to_machine_color = dataset_dir + "obj_class_to_machine_color.json"

with open(obj_class_to_machine_color) as json_file:
    object_color = json.load(json_file)

ann_files = []
for subdir in subdirs:
    ann_files += [f for f in glob(dataset_dir + subdir + '/ann/' + '*.json', recursive=True)]
    
print("DATASETS USED: {}".format(subdirs))
print("TOTAL IMAGES COUNT: {}\n".format(len(ann_files)))
    
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

# In[]: Prepare for training
#val_size = 0.
#test_size = 0.9999
#
#print("Train:Val:Test split = {}:{}:{}\n".format(1-val_size-test_size, val_size, test_size))
#
#ann_files_train, ann_files_valtest = train_test_split(ann_files, test_size=val_size+test_size, random_state=random_state)
#ann_files_val, ann_files_test = train_test_split(ann_files_valtest, test_size=test_size/(test_size+val_size+1e-8)-1e-8, random_state=random_state)
#del(ann_files_valtest)
#
#print("Training files count: {}".format(len(ann_files_train)))
#print("Validation files count: {}".format(len(ann_files_val)))
#print("Testing files count: {}\n".format(len(ann_files_test)))

with open('pickles/{}.pickle'.format(weights), 'rb') as f:
    ann_files_train = pickle.load(f)
    ann_files_val = pickle.load(f)
    ann_files_test = pickle.load(f)
    
# In[]: 
def predict_generator(files, preprocessing_fn = None, batch_size = 1):
    
    i = 0
    
    while True:
        
        x_batch = np.zeros((batch_size, input_shape[0], input_shape[1], 3), dtype=np.uint8)
        
        for b in range(batch_size):
            
            if i == len(files):
                i = 0
                
            x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
            
            x_batch[b] = x
                
            i += 1
            
        x_batch = preprocessing_fn(x_batch)
            
        yield x_batch
        
        
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

predict_gen = predict_generator(files = ann_files_test, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

eval_gen = val_generator(files = ann_files_test, 
                             preprocessing_fn = preprocessing_fn, 
                             batch_size = batch_size)

# In[]: Bottleneck
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
model.load_weights('weights/' + weights + '.hdf5')

# In[]: 
loss = [dice_coef_multiclass_loss]
metrics = ['categorical_accuracy']

learning_rate = 1e-4
optimizer = optimizers.Adam(lr = learning_rate)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print("\nOptimizer: {}\nLearning rate: {}\nLoss: {}\nMetrics: {}\n".format(optimizer, learning_rate, loss, metrics))

# In[]:
i = 228
x = get_image(ann_files[i].replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
x = preprocessing_fn(x)
y_pred = model.predict(np.expand_dims(x, axis=0))
y_true = get_image(ann_files[i].replace('/ann/', '/masks_machine/').split('.json')[0], label=True, resize = True if resize else False)

if visualize:
    plt.imshow(np.squeeze(np.argmax(y_pred, axis=-1)))

# In[]:
#steps = len(ann_files_test)//batch_size
#
#history = model.evaluate_generator(
#        generator = eval_gen,
#        steps = steps,
#        verbose = verbose
#        )
#
#print(history)

#with open('evaluate.pickle', 'wb') as f:
#    pickle.dump(history, f)
    
# In[]:
#y_pred = model.predict_generator(
#        generator = predict_gen,
#        steps = steps,
#        verbose = verbose
#        )
#
#with open('predict.pickle', 'wb') as f:
#    pickle.dump(y_pred, f)

# In[]:
from metrics import tpfpfn, mAccuracy, mPrecision, mRecall, mIU, mF1, mTNR, mNPV, mFPR, mFDR, mFNR, mBACC

if save_results:
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (5,30)
    fontScale              = 1
    fontColor              = (255,0,0)
    lineType               = 2
    
mAccuracy_ = 0
mPrecision_ = 0
mRecall_ = 0
mIU_ = 0
mF1_ = 0
mTNR_ = 0
mNPV_ = 0
mFPR_ = 0
mFDR_ = 0
mFNR_ = 0
mBACC_ = 0

dlina = len(ann_files_test)
    
for aft in tqdm(ann_files_test):
    
    x = get_image(aft.replace('/ann/', '/img/').split('.json')[0], resize = True if resize else False)
    x_vis = x.copy()
    x = preprocessing_fn(x)
    y_pred = model.predict(np.expand_dims(x,axis=0))
      
    y_true = get_image(aft.replace('/ann/', '/masks_machine/').split('.json')[0], label=True, resize = True if resize else False)
    
    if save_results:
        vis_pred = cv2.addWeighted(x_vis,1,cv2.applyColorMap(255//2*np.squeeze(np.argmax(y_pred, axis=-1)).astype(np.uint8),cv2.COLORMAP_OCEAN),1,0)
        cv2.putText(vis_pred, 'Prediction', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        
        vis_true = cv2.addWeighted(x_vis,1,cv2.applyColorMap(255//2*y_true.astype(np.uint8),cv2.COLORMAP_OCEAN),1,0)
        cv2.putText(vis_true, 'Ground Truth', bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
                 
        if not os.path.exists("results/{}".format(weights)):
            os.mkdir("results/{}".format(weights))
            
        cv2.imwrite("results/{}/{}.png".format(weights, aft.split('/')[-1].split('.')[0]), cv2.cvtColor(np.vstack((vis_pred, vis_true)), cv2.COLOR_BGR2RGB))
    
    y_true = y_true.astype('int64')  
    y_pred = np.squeeze(np.argmax(y_pred, axis=-1)).astype('int64')

    mAccuracy_ += mAccuracy(y_pred, y_true)/dlina
    mPrecision_ += mPrecision(y_pred, y_true)/dlina
    mRecall_ += mRecall(y_pred, y_true)/dlina
    mIU_ += mIU(y_pred, y_true)/dlina
    mF1_ += mF1(y_pred, y_true)/dlina
    mTNR_ += mTNR(y_pred, y_true)/dlina
    mNPV_ += mNPV(y_pred, y_true)/dlina
    mFPR_ += mFPR(y_pred, y_true)/dlina
    mFDR_ += mFDR(y_pred, y_true)/dlina
    mFNR_ += mFNR(y_pred, y_true)/dlina
    mBACC_ += mBACC(y_pred, y_true)/dlina
    
print("accuracy: {}".format(mAccuracy_))
print("precision: {}".format(mPrecision_))
print("recall: {}".format(mRecall_))
print("iu: {}".format(mIU_))
print("f1: {}".format(mF1_))
print("TNR: {}".format(mTNR_))
print("NPV: {}".format(mNPV_))
print("FPR: {}".format(mFPR_))
print("FDR: {}".format(mFDR_))
print("FNR: {}".format(mFNR_))
print("BACC: {}".format(mBACC_))