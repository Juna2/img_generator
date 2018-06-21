#!/usr/bin/env python
import os

for root, dirs, files in os.walk('/home/juna/catkin_ws/src/img_generator/img'):
    for fname in files:
        full_fname = os.path.join(root, fname)

        print full_fname 
