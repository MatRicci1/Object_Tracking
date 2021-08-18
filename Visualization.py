############################################
# Visualizatio functions
############################################

import numpy as np
import cv2
from Images_functions import *
from Hough import *

### Plot Functions

# Show Image
def img_show(name,img):
    cv2.imshow(name,normalize(img))

# Show pixels with high value of orientation
def best_arg_show(mod_img_0, arg_img_0, tsh):

    mod_img = mod_img_0.copy()
    arg_img = arg_img_0.copy()

    # Normalizing Module Image
    mod_img = cv2.convertScaleAbs(255*normalize(mod_img))

    # Normalizing Arg Image
    cv2.normalize(arg_img, arg_img,0,255,cv2.NORM_MINMAX)
    arg_img = cv2.convertScaleAbs(arg_img)
    #arg_img = cv2.convertScaleAbs(255*normalize(arg_img))

    # Best_arg img
    best_arg_img = cv2.cvtColor(arg_img, cv2.COLOR_GRAY2RGB)
    idx_pixels = np.where(mod_img<255*tsh)
    best_arg_img[idx_pixels[0],idx_pixels[1],:] = [0,0,255]
    return best_arg_img
    

# Create track window over image
def track_display(img_0, window):
    img = img_0.copy()
    r, c, h, w = window
    frame_tracked = cv2.rectangle(img, (r,c), (r+h,c+w), (255,0,0) ,2)

    return frame_tracked