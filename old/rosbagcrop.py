#!/usr/bin/env python

# In[1]:
import os
import cv2
import numpy as np

import rosbag
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from tqdm import tqdm

# In[20]:
bag_name = "2019-05-20-17-13-29_1.bag"
bag_file = "/home/kenny/dgx/home/datasets/kia/20-05-2019/" + bag_name
output_dir = "/home/kenny/dgx/colddata/segmification/im1/"
# image_topic = "/pylon_camera_node/image_raw/compressed"
# image_topic = "sensor_msgs/CompressedImage"
image_topic = "/apollo/sensor/camera/perception/image_front_camera/compressed"
step = 1

compressed = True

# In[21]:
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# In[22]:
bag = rosbag.Bag(bag_file, "r")
bridge = CvBridge()
count = 0

# In[23]:
bag.get_type_and_topic_info()

# In[24]:
# for topic, msg, t in bag.read_messages(topics=[image_topic]):

#     print(msg)

#     count += 1   
    
#     if count == 5:
#         break

# bag.close()

# In[25]:
for topic, msg, t in bag.read_messages(topics=[image_topic]):
    if count % step == 0:
        
        if compressed:
            raw_data = np.fromstring(msg.data, np.uint8)
            cv_img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        else:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)
        print("Wrote image %i" % count)

    count += 1   
    
#     if count == 5:
#         break

bag.close()

