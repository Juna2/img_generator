#!/usr/bin/env python

import cv2
from cv_bridge import CvBridge
import numpy as np
import sys, time
import roslib
import rospy
import os
from sensor_msgs.msg import Image

def main():
    fname = '/home/juna/catkin_ws/src/img_generator/img/26.jpg'
    imgs = []
    bridge = CvBridge();
    img_pub = rospy.Publisher('/croppedRoI', Image, queue_size=10)
    rospy.init_node('image_feature', anonymous=True)
    rate = rospy.Rate(1) # 10hz

    count = 0
    for root, dirs, files in os.walk('/home/juna/catkin_ws/src/img_generator/img'):
        for fname in files:
            full_fname = os.path.join(root, fname)
            print(full_fname)
            # img_names.append(full_fname)
            img = cv2.imread(full_fname, cv2.IMREAD_COLOR)
            img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img_color)
            count += 1


    
    #img = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
    
    #x = 25; y = 100; 
    #w = 400; h = 430; 
    #img = img[y:y+h, x:x+w]  

     
    #print(original.shape)
    #cv2.imshow('Original', original)

    #rospy.sleep(5.)
    #cv2.waitKey(0)
    count2 = 1
    while not rospy.is_shutdown():
        num = count2 % len(imgs)
        img_pub.publish(bridge.cv2_to_imgmsg(imgs[num], "rgb8"))
        rate.sleep()
        count2 += 1

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass




# import rospy
# from std_msgs.msg import String

# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass



# import cv2

# fname = 'lena.jpg'

# original = cv2.imread(fname, cv2.IMREAD_COLOR)
# gray = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
# unchange = cv2.imread(fname, cv2.IMREAD_UNCHANGED)

# cv2.imshow('Original', original)
# cv2.imshow('Gray', gray)
# cv2.imshow('Unchange', unchange)
