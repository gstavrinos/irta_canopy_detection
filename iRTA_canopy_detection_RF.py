#!/usr/bin/env python3
# coding: utf-8

import sys
import rospy
import pickle
import pathlib
import numpy as np
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from rospy.numpy_msg import numpy_msg

path_to_current_directory = str(pathlib.Path().parent.absolute())
path_to_model = path_to_current_directory + "/models/"

# TODO if there are multiple models to select from
# model = sys.argv[2]

model = "iRTA_RF_trained_model.sav"

canopy_position = {
    0 : 'center',
    1 : 'left',
    2 : 'none',
    3 : 'right'
}

loaded_model = pickle.load(open(path_to_model + model, 'rb'))

def callback(msg):
    np_arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    image = PILImage.fromarray(np.uint8(np_arr)).convert('RGB')
    X = np.transpose(np.expand_dims(np.array(image.resize((16,16))).flatten(),axis=1))
    result = loaded_model.predict(X)
    print(msg.header.stamp, " -- ", canopy_position[result[0]])

rospy.init_node("irta_canopy_detection")
rospy.Subscriber("/camera/color/image_raw", numpy_msg(Image), callback)
rospy.spin()

