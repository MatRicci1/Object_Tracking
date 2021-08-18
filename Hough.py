############################################
# Hough functions
############################################

import numpy as np
import cv2
from collections import defaultdict
from Images_functions import *
from Visualization import *

# Create R-Table
def create_R_Table(arg_roi, indexs, dis_ang):
    center =  np.array( [ arg_roi.shape[1] // 2, arg_roi.shape[0] // 2 ] )
    arg_roi = arg_roi * dis_ang // np.pi
    R_Table = defaultdict(list)

    for idx, idy in indexs:
        dist = center -  np.array([idx, idy])
        R_Table[arg_roi[idx, idy]].append(dist)

    return R_Table

# Create R-Table
def Gen_Hough_Transform(R_Table, arg_frame, indexs, dis_ang, decay):
    hough_img = np.zeros(arg_frame.shape, dtype = np.float64)
    arg_frame = arg_frame * dis_ang // np.pi

    for idx, idy in indexs:
        coef = 1
        for dist in R_Table[arg_frame[idx, idy]]:
            dist_x = idx + dist[0]
            dist_y = idy + dist[1]

            if not (0 <= dist_x < arg_frame.shape[0]) or not (0 <= dist_y < arg_frame.shape[1]):
                continue

            hough_img[dist_x,dist_y] += 1*coef
            coef *= decay

    return hough_img

# Track max value of Hough Tranform
def Track_Hough(img_0, window):
    img = img_0.copy()
    _, _, h, w = window
    max_pose_x, max_pose_y = mat_argmax(img)
    window_hough =   max_pose_x - (w // 2), max_pose_y - (h // 2), h, w

    return window_hough