#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import numpy as np
import sys
from scipy import misc
from cv_bridge import CvBridge, CvBridgeError
import PIL.Image as PIL
from segmentation_models import Linknet
from segmentation_models.backbones import get_preprocessing

import webcolors

def colorize(mask):
    hex_colors = ['#000000', '#FF0000', '#0000FF']

    rgb_colors = []

    for hex_color in hex_colors:
        rgb_colors.append(webcolors.hex_to_rgb(hex_color))
        
    colors = np.array(rgb_colors)
        
    colorMask = np.zeros([mask.shape[0], mask.shape[1], 3], dtype=np.uint8)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            colorMask[r,c,] = colors[mask[r,c]]

    return colorMask

frames_ratio = 3
frames_counter = 0

#from models.Unet_short import unet_short
#model = unet_short(input_size = (256, 640, 3), n_classes=3)

model = Linknet(backbone_name='resnet18', input_shape=(256, 640, 3), classes=3, activation='softmax')
model.load_weights("weights/innopolis/linknet-resnet18_overfit/2019-01-21 11-14-12.hdf5")

preprocessing_fn = get_preprocessing('resnet18')

model._make_predict_function()

def callback(data):
    global path
    start = time.time()
    global model
    global preprocessing_fn
    rospy.loginfo(rospy.get_caller_id() + "I heard")
    
    img = PIL.open(data.data)
    img = img.resize((640,256))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocessing_fn(img)
    
    print(img.shape)

    pred = model.predict(img)

    name = str(time.time())
    
#    print(pred.shape)
    
    pred = pred.squeeze()
    pred = np.argmax(pred, axis=-1)  
    pred = pred.astype(np.uint8)
    pred = colorize(pred)
    
#    print(pred.shape)

    misc.imsave(name + ".png", pred)

    pub = rospy.Publisher('mask_publisher', Image)
#    cv_image = np.asarray(cv2.imread(name + ".png"))
    bridge = CvBridge()
    image_message = bridge.cv2_to_imgmsg(pred, "bgr8")

    print("predicting finished")
    pub.publish(image_message)
    print(time.time() - start)

def listener():
     # In ROS, nodes are uniquely named. If two nodes with the same
     # node are launched, the previous one is kicked off. The
     # anonymous=True flag means that rospy will choose a unique
     # name for our 'listener' node so that multiple listeners can
     # run simultaneously.
    rospy.init_node('predict', anonymous=True)

    rospy.Subscriber("img_subscriber", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    print("start")
    listener()
