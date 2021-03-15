#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import rospy
import rospkg
import pickle
import pathlib
import numpy as np
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, Range
from PIL import Image as PILImage
from geometry_msgs.msg import Twist
from rospy.numpy_msg import numpy_msg

model = "iRTA_RF_trained_model.sav"
rospack = rospkg.RosPack()

path_to_model = os.path.join(rospack.get_path("irta_canopy_detection"), "models", model)

#path_to_current_directory = str(pathlib.Path().parent.absolute())
#path_to_model = path_to_current_directory + "/models/"
#loaded_model = pickle.load(open(path_to_model + model, 'rb'))
loaded_model = pickle.load(open(path_to_model, 'rb'))

canopy_position = {
    0 : 'center',
    1 : 'left',
    2 : 'none',
    3 : 'right'
}

max_effective_spraying_distance = 0.8 # meters
min_effective_height_percentage = 0.3
max_effective_height_percentage = 0.6
effective_width_percentage = 0.3
max_depth_distance = 14.0 # meters
min_close_percentage = 0.3

fov_msg = Range()
fov_msg.header.frame_id = "spray_valve_link"
fov_msg.radiation_type = 1
fov_msg.field_of_view = 3.1415/3
fov_msg.min_range = 0
fov_msg.max_range = 1.0

valve_publisher = None
fov_pub = None

canopy_detected = False
on_the_move = False
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
        poi = np_arr[int(min_effective_height_percentage * msg.height):int(max_effective_height_percentage * msg.height),:int(effective_width_percentage * msg.width)] / 255.0 * max_depth_distance
        need_valve_on = np.count_nonzero(poi <= max_effective_spraying_distance) >= min_close_percentage * poi.size
    if need_valve_on and on_the_move and not valve:
        print("Sending on command")
        valve_publisher.publish(True)
        valve = True
    elif (not need_valve_on or not on_the_move) and valve:
        print("Sending off command")
        valve_publisher.publish(False)
        valve = False
    fov_msg.range = fov_msg.max_range if valve else fov_msg.min_range

    fov_pub.publish(fov_msg)

def cmdVelCallback(msg):
    global on_the_move
    on_the_move = msg.linear.x > 0

rospy.init_node("irta_canopy_detection")
rospy.Subscriber("/usb_cam/image_raw", numpy_msg(Image), usbCamCallback)
#  rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", numpy_msg(Image), depthCamCallback)
if not rospy.get_param("use_sim_time", False):
    rospy.Subscriber("/camera/depth/image_rect_raw", numpy_msg(Image), depthCamCallback)
else:
    rospy.Subscriber("/camera/depth/image_raw", numpy_msg(Image), depthCamCallback)
    min_close_percentage = 0.12
rospy.Subscriber("/kymco_maxxer90_ackermann_steering_controller/cmd_vel", Twist, cmdVelCallback)
valve_publisher = rospy.Publisher("/kymco_maxxer90_ackermann_steering_controller/spray_valve", Bool, queue_size=1)
fov_pub = rospy.Publisher("spray_valve_viz", Range, queue_size=1)
rospy.spin()

