#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
import numpy as np
from segmentation_models import Linknet
from segmentation_models.backbones import get_preprocessing

global model
global preprocessing_fn
global bridge
global pub

def callback(data):

    start = time.time()

    rospy.loginfo(rospy.get_caller_id() + "I heard")
    
    img = bridge.compressed_imgmsg_to_cv2(data, "rgb8")
    img  = cv2.resize(img, (640,256))
    orig = img.copy() # For visualisation in RViz
    img = np.expand_dims(img, axis=0) 
    img = preprocessing_fn(img)

    pred = model.predict(img)
    
    pred = pred.squeeze()
    pred = np.argmax(pred, axis=-1)  
    pred = pred.astype(np.uint8)   

    #image_message = bridge.cv2_to_imgmsg(pred, 'mono8')\

    pred *= 255//2
    image_message = bridge.cv2_to_imgmsg(cv2.addWeighted(orig,1,cv2.applyColorMap(pred,cv2.COLORMAP_OCEAN),1,0), 'rgb8') # For visualisation in RViz

    print("predicting finished")
    pub.publish(image_message)
    print(time.time() - start)

def listener():

    rospy.init_node('predict', anonymous=True)
    rospy.Subscriber("/pylon_camera_node/image_raw/compressed", CompressedImage, callback)
    #rospy.Subscriber("/apollo/sensor/camera/perception/image_front_camera/compressed", CompressedImage, callback)

    rospy.spin()

if __name__ == '__main__':

    model = Linknet(backbone_name='resnet18', input_shape=(256, 640, 3), classes=3, activation='softmax')
    model.load_weights("weights/innopolis/linknet-resnet18_overfit/2019-01-21 11-14-12.hdf5")
    model._make_predict_function()

    preprocessing_fn = get_preprocessing('resnet18')
    bridge = CvBridge()
    pub = rospy.Publisher('image_publisher', Image, queue_size=1)

    print("start")
    listener()
