#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import time
import numpy as np
from segmentation_models import Linknet
from segmentation_models.backbones import get_preprocessing

global model
global preprocessing_fn
global bridge
global pub

def extract_largest_blobs(mask, area_threshold, num_blobs=1):

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=4)
                
    st = stats[:,-1][stats[:,-1] > area_threshold] #Ban areas smaller than threshold
                    
    if nb_components == 1 or len(st) < 2:
        return None, None, None

    if (num_blobs <= len(st)-1):
        n = num_blobs+1
    else:
        n = len(st)
    
    blob_index = np.argsort(stats[:,-1])[-n:-1]
                
    return output, blob_index[::-1], centroids[blob_index[::-1]]

def centroids(centroids_a, centroids_b):
    
    if len(centroids_b) < 2:
        return 1
    
    cb_sorted = centroids_b[:,0].copy()
    cb_sorted.sort()

    if (cb_sorted[0] < centroids_a[0,0] < cb_sorted[1]):
        return 2
    else:
        return 1

def fill_holes(mask):

    mask_floodfill = mask.astype('uint8').copy()
    h, w = mask.shape[:2]
    cv2.floodFill(mask_floodfill, np.zeros((h+2, w+2), np.uint8), (0,0), 255)

    out = mask | cv2.bitwise_not(mask_floodfill)
    
    return out.astype(np.bool)

def merge(mask_a, mask_b):
    
    out = np.zeros((256,640), dtype=np.uint8)

    out[mask_b] = 2
    out[mask_a] = 1
    
    return out

def postProcess(mask_a, mask_b, area_threshold):
        
    output_a, blob_index_a, centroids_a = extract_largest_blobs(mask_a, area_threshold)
    output_b, blob_index_b, centroids_b = extract_largest_blobs(mask_b, area_threshold, num_blobs=2)
        
    anon = centroids_a is None
    bnon = centroids_b is None
    
    if anon:
        mask_a = np.zeros_like(mask_a)
    
    if bnon:
        mask_b = np.zeros_like(mask_b)
    
    if anon and bnon:
        return merge(mask_a, mask_b)
    
    if not anon:
        mask_a = output_a == blob_index_a[0]
        
    if not bnon:
        mask_b = output_b == blob_index_b[0]
        
    if not anon and not bnon:
        keep = centroids(centroids_a, centroids_b) # Num of centroids_b to keep
        if keep > 1:
            mask_b += output_b == blob_index_b[1]
    
    if not anon:
        mask_a = fill_holes(mask_a)
        
    if not bnon:
        mask_b = fill_holes(mask_b)
            
    return merge(mask_a, mask_b)

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

    a = pred[...,1] > 1/3
    b = pred[...,2] > 1/3

    pred = postProcess(a, b, area_threshold=5000)

    #image_message = bridge.cv2_to_imgmsg(pred, 'mono8')\

    pred *= 255//2
    image_message = bridge.cv2_to_imgmsg(cv2.addWeighted(orig,1,cv2.applyColorMap(pred,cv2.COLORMAP_OCEAN),1,0), 'rgb8') # For visualisation in RViz

    pub.publish(image_message)
    print(time.time() - start)

def listener():

    rospy.init_node('predict', anonymous=True)
    rospy.Subscriber("/pylon_camera_node/image_raw/compressed", CompressedImage, callback)
    rospy.spin()

if __name__ == '__main__':

    model = Linknet(backbone_name='resnet18', input_shape=(256, 640, 3), classes=3, activation='softmax')
    model.load_weights("weights/innopolis/linknet-resnet18_overfit/2019-01-21 11-14-12.hdf5")
    model._make_predict_function()

    preprocessing_fn = get_preprocessing('resnet18')
    bridge = CvBridge()
    pub = rospy.Publisher('image_publisher_postprocess', Image, queue_size=1)

    print("start")
    listener()
