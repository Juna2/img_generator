#!/usr/bin/env python

# import cv2
# from cv_bridge import CvBridge
# import numpy as np
# import sys, time
# import rospy
# import os
# from sensor_msgs.msg import Image

# def main():
#     fname = '/home/juna/catkin_ws/src/generate_img/img/105.jpg'
#     bridge = CvBridge();
#     img_pub = rospy.Publisher('rcnn/image_raw', Image, queue_size=10)
#     rospy.init_node('image_feature', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     #im = Image.open('/home/juna/catkin_ws/src/generate_img/src/090.jpg')
#     img = cv2.imread(fname, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     x = 25; y = 100; 
#     w = 400; h = 430; 
#     img = img[y:y+h, x:x+w]  

     
#     #print(original.shape)
#     #cv2.imshow('Original', original)

#     #rospy.sleep(5.)
#     #cv2.waitKey(0)
#     while not rospy.is_shutdown():
#         img_pub.publish(bridge.cv2_to_imgmsg(img, "rgb8"))
#         rate.sleep()

# if __name__ == '__main__':
#     try:
#         main()
#     except rospy.ROSInterruptException:
#         pass

from img_generator.msg import points as p
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math as m
import sys, time
import os
from sensor_msgs.msg import Image
import rospy
from ros_faster_rcnn.msg import *



# def callback(data):
#     #rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data.data)
#     data.detections.data[0].width
#     data.image

# def main():
#     # In ROS, nodes are uniquely named. If two nodes with the same
#     # name are launched, the previous one is kicked off. The
#     # anonymous=True flag means that rospy will choose a unique
#     # name for our 'listener' node so that multiple listeners can
#     # run simultaneously.
#     rospy.init_node('point', anonymous=True)

#     rospy.Subscriber('rcnn/res/full1', DetectionFull, callback)

#     # spin() simply keeps python from exiting until this node is stopped
#     rospy.spin()

# if __name__ == '__main__':
#     main()


class image_converter:

    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic_2", Image, queue_size=10)
        self.point_pub = rospy.Publisher("points", p, queue_size=5)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("rcnn/res/full1", DetectionFull, self.callback)
        self.cv_image = None
        self.mat = np.eye(4)

    def drw_cord(self, name):
        X_trans = np.array([[1, 0, 0, 30],
                            [0, 1, 0,  0],
                            [0, 0, 1,  0], 
                            [0, 0, 0,  1]])
        Y_trans = np.array([[1, 0, 0,  0],
                            [0, 1, 0, 30],
                            [0, 0, 1,  0], 
                            [0, 0, 0,  1]])

        text = "{}({:d}, {:d})".format(name, int(self.mat[0,3]), int(self.mat[1,3]))
        cv2.putText(self.cv_image, text, (int(self.mat[0,3]), int(self.mat[1,3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 250, 250), 1, cv2.LINE_AA)
                                                
        cv2.line(self.cv_image, (int(self.mat[0,3]), int(self.mat[1,3])), (int(self.mat.dot(X_trans)[0,3]), int(self.mat.dot(X_trans)[1,3])), (255, 0, 0), 5)
        cv2.line(self.cv_image, (int(self.mat[0,3]), int(self.mat[1,3])), (int(self.mat.dot(Y_trans)[0,3]), int(self.mat.dot(Y_trans)[1,3])), (0, 0, 255), 5)

        


    
    def callback(self,data):

        if data.detections.size == 2:

            try:
                up_x = data.detections.data[0].x
                up_y = data.detections.data[0].y
                up_width = data.detections.data[0].width
                
                down_x = data.detections.data[1].x
                down_y = data.detections.data[1].y
                down_width = data.detections.data[1].width
                down_height = data.detections.data[1].height

                up_middle = up_x + (up_width / 2)
                down_middle = down_x + (down_width / 2)
                down_y = down_y + down_height

                d1 = ((up_middle - down_middle)**2 + (up_y - down_y)**2)**0.5
                d2 = d1 * 0.3
                d3 = d1 * 1
                d4 = d1 * 0.61

                

                rot_rad_1 = m.radians(70) + m.atan2(down_y - up_middle, up_middle - down_middle)
                rot_rad_2 = m.radians(22) + m.atan2(down_y - up_middle, up_middle - down_middle)
                rot_rad_3 = m.radians(30) + m.atan2(down_y - up_middle, up_middle - down_middle)

                M1 = np.array([[1, 0, 0, up_middle],
                            [0, 1, 0,      up_y],
                            [0, 0, 1,         0], 
                            [0, 0, 0,         1]])

                M2_1 = np.array([[m.cos(rot_rad_1), -m.sin(rot_rad_1), 0, 0],
                                [m.sin(rot_rad_1),  m.cos(rot_rad_1), 0, 0],
                                [               0,                 0, 1, 0], 
                                [               0,                 0, 0, 1]])

                M3_1 = np.array([[1, 0, 0, d2],
                                [0, 1, 0,  0],
                                [0, 0, 1,  0], 
                                [0, 0, 0,  1]])

                M2_2 = np.array([[m.cos(rot_rad_2), -m.sin(rot_rad_2), 0, 0],
                                [m.sin(rot_rad_2),  m.cos(rot_rad_2), 0, 0],
                                [               0,                 0, 1, 0], 
                                [               0,                 0, 0, 1]])

                M3_2 = np.array([[1, 0, 0, d3],
                                [0, 1, 0,  0],
                                [0, 0, 1,  0], 
                                [0, 0, 0,  1]])
                
                M2_3 = np.array([[m.cos(rot_rad_3), -m.sin(rot_rad_3), 0, 0],
                                [m.sin(rot_rad_3),  m.cos(rot_rad_3), 0, 0],
                                [               0,                 0, 1, 0], 
                                [               0,                 0, 0, 1]])

                M3_3 = np.array([[1, 0, 0, d4],
                                [0, 1, 0,  0],
                                [0, 0, 1,  0], 
                                [0, 0, 0,  1]])

                points = p()

                #down_y+down_height
                self.cv_image = self.bridge.imgmsg_to_cv2(data.image, desired_encoding="passthrough")      
                self.mat = M1.dot(M2_1)
                self.drw_cord("ori")
                self.mat = self.mat.dot(M3_1)

                # camera_matrix:
                # rows: 3
                # cols: 3
                # data: [1583.572913,             0,          634.823565, 
                #                  0,   1586.708187,   569.9147809999999, 
                #                  0,             0,                   1]

                #change to real-life coordinate
                points.x1 = (1200*(self.mat[0,3] - 634.823565))/1583.572913
                points.y1 = (1200*(self.mat[1,3] - 569.9147809999999))/1586.708187
                self.drw_cord("A")
                self.mat = M1.dot(M2_2).dot(M3_2)     
                points.x2 = (1200*(self.mat[0,3] - 634.823565))/1583.572913
                points.y2 = (1200*(self.mat[1,3] - 569.9147809999999))/1586.708187
                self.drw_cord("B")
                self.mat = M1.dot(M2_3).dot(M3_3)
                points.x3 = (1200*(self.mat[0,3] - 634.823565))/1583.572913
                points.y3 = (1200*(self.mat[1,3] - 569.9147809999999))/1586.708187
                self.drw_cord("C")



                #cv2.circle(self.cv_image, (int(self.mat[0,3]), int(self.mat[1,3])), 10, 255) 

            except CvBridgeError as e:
                print(e)
            #print(cv_image)
            
            #cv2.line(cv_image, (up_middle, up_y), (down_middle, down_y+down_height), (255, 0, 0), 5)
            #cv2.line(cv_image, (up_middle, up_y), (down_middle, down_y+down_height), (255, 0, 0), 5)
            cv2.line(self.cv_image, (up_middle, up_y), (down_middle, down_y), (186, 85, 211), 2)


            #(up_middle, up_y) (down_middle, down_y+down_height)

            
            # (rows,cols,channels) = cv_image.shape
            # if cols > 60 and rows > 60 :
            #     cv2.circle(cv_image, (50,50), 10, 255)
        
            #cv2.imshow("Image window", cv_image)
            #cv2.waitKey(3)
        
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.cv_image, "rgb8"))
                self.point_pub.publish(points)
            except CvBridgeError as e:
                print(e)

        else:
            print("Not yet")


def main(args):
    ic = image_converter()
    rospy.init_node('point', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
