#!/usr/bin/env python3
# coding: utf-8

import sys
import rospy
import pickle
import pathlib
import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from rospy.numpy_msg import numpy_msg

path_to_current_directory = str(pathlib.Path().parent.absolute())
path_to_model = path_to_current_directory + "/models/"

model = "iRTA_RF_trained_model.sav"

canopy_position = {
    0 : 'center',
    1 : 'left',
    2 : 'none',
    3 : 'right'
}

loaded_model = pickle.load(open(path_to_model + model, 'rb'))

max_effective_spraying_distance = 0.8 # meters
effective_height_percentage = 0.4
effective_width_percentage = 0.3
max_depth_distance = 14.0 # meters
min_close_percentage = 0.3

valve_publisher = None

canopy_detected = False
valve = False

def usbCamCallback(msg):
    global canopy_detected
    np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    image = PILImage.fromarray(np.uint8(np_arr)).convert('RGB')
    X = np.transpose(np.expand_dims(np.array(image.resize((16,16))).flatten(),axis=1))
    result = loaded_model.predict(X)
    canopy_detected = canopy_position[result[0]] != "none" and canopy_position[result[0]] != "left"

def depthCamCallback(msg):
    global valve
    need_valve_on = False
    if canopy_detected:
        np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        poi = np_arr[int((1.0 - effective_height_percentage) * msg.height):,:int(effective_width_percentage * msg.width)] / 255.0 * max_depth_distance
        need_valve_on = np.count_nonzero(poi <= max_effective_spraying_distance) >= min_close_percentage * poi.size
    if need_valve_on and not valve:
        valve_publisher.publish(True)
        valve = True
    elif not need_valve_on and valve:
        valve_publisher.publish(False)
        valve = False

rospy.init_node("irta_canopy_detection")
rospy.Subscriber("/usb_cam/image_raw", numpy_msg(Image), usbCamCallback)
rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", numpy_msg(Image), depthCamCallback)
valve_publisher = rospy.Publisher("/kymco_maxxer90_ackermann_steering_controller/spray_valve", Bool, queue_size=1)
rospy.spin()

