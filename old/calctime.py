# -*- coding: utf-8 -*-

# In[1]:
import os
import numpy as np
from PIL import Image
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

# In[4]:
from models.Unet import unet
from models.Unet_short import unet_short
#from models.DeepLabv3plus import Deeplabv3
from models.deeplabv31p import Deeplabv31p
from segmentation_models import Unet, FPN, Linknet, PSPNet
from segmentation_models.backbones import get_preprocessing

num_classes = 3
#input_shape = (240,624,3)
#input_shape = (336,624,3)
input_shape = (256,640,3)

#model = unet(input_size = input_shape, n_classes=num_classes)
#model = unet_short(input_size = input_shape, n_classes=num_classes)
#model = Deeplabv3(input_shape=input_shape, classes=num_classes)  
#model = Deeplabv3(input_shape=input_shape, classes=num_classes, backbone='xception')  
#model = Deeplabv31p(input_shape = input_shape, classes = num_classes)

backbone = 'resnet18'
#backbone = 'mobilenetv2'
#backbone = 'seresnext50'
#backbone = 'seresnet18'

#backbone = 'seresnext101'

#backbone = 'densenet201'
#backbone = 'densenet121'
#backbone = 'seresnet50'
#backbone = 'seresnet152'
#backbone = 'seresnet101'
#backbone = 'resnext50'
#backbone = 'senet154'
#backbone = 'inceptionv3'
#backbone = 'inceptionresnetv2'
#backbone = 'resnext101'

#model = Unet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
#model = FPN(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
#model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')
model = Linknet(backbone_name=backbone, input_shape=input_shape, classes=num_classes, activation='softmax')

#from keras import optimizers
#optimizer = optimizers.Adam(lr = 1e-4)
#metrics = ['accuracy']
#loss = 'categorical_crossentropy'
#model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.summary()

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

# In[]
from sklearn.model_selection import train_test_split

test_size = 0.9
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=test_size, random_state=1)

#images_test = images_test[:50]
print(len(images_test))

# In[56]:
model.predict(np.zeros((1,input_shape[0],input_shape[1],3)))

preprocessing_fn = get_preprocessing(backbone)

start_time = time.time()

for i, im in enumerate(images_test):
    
#    x = get_image(im)
#    x = np.expand_dims(x, axis = 0)
#    x = np.float32(x/255.)
#    model.predict(x)
    model.predict(np.zeros((1,input_shape[0],input_shape[1],3)))
    print(i)
            
#        mask_np = output.squeeze().cpu().numpy()
    
print("--- {} seconds ---".format(time.time() - start_time))
print("--- {} fps ---".format(len(images_test)/(time.time() - start_time)))

# In[]:
#import matplotlib.pyplot as plt
#import numpy as np
#
#img = plt.imread("/home/kenny/Desktop/lanes-segmentation/image3.jpg")
#y = model.predict(np.expand_dims(img[:256,:256,], axis=0))
#out = y.squeeze().reshape(256,256,21)
#mask = np.argmax(out, axis=-1)
#plt.imshow(mask)

