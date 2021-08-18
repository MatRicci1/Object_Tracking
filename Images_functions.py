############################################
# Images functions
############################################

import numpy as np
import cv2
from Hough import *

# Calcul Gradient
def grad(img):
    img = np.float64(img)

    # Méthode filter2D - Ix
    Ix_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
    grad_x = cv2.filter2D(img,-1,Ix_kernel)

    # Méthode filter2D - Iy
    Iy_kernel = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]])
    grad_y = cv2.filter2D(img,-1,Iy_kernel)

    return grad_x, grad_y

# Module gradient
def mod_grad(img):
    grad_x, grad_y = grad(img)
    mod = np.sqrt(grad_x**2 + grad_y**2)

    return mod

# Orientation Gradient
def arg_grad(img):
    grad_x, grad_y = grad(img)
    arg = np.arctan2(grad_y,grad_x)

    return arg

# Normalize 
def normalize(img):
    img = img/img.max().max()

    return img

# Normalize 
def normalize_abs(img_0):
    img = img_0.copy()
    cv2.normalize(img, img,0,255,cv2.NORM_MINMAX)
    img = cv2.convertScaleAbs(img)

    return img

# Select pixels with more defined orientations from ROI and frame
def pixel_selection(frame, roi, tsh):
    frame = normalize(frame)
    roi =  normalize(roi)

    idx_frame = np.argwhere(frame>tsh)
    idx_roi   = np.argwhere(roi>tsh)

    return idx_frame, idx_roi

# Get Max value index of Matrix
def mat_argmax(Matrix):
    j, i = np.unravel_index(Matrix.argmax(), Matrix.shape)
    return [i, j]


# Get Histogram of ROi of a specific channel of HSV
def hist_roi(R, channel, hist_size, range_channel):
    roi = R.copy()
    
    # Conversion to Hue-Saturation-Value space 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Computation mask of the histogram: Pixels with S<30, V<20 or V>235 are ignored
    mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
    
    # Marginal histogram of the Hue component
    roi_hist = cv2.calcHist([hsv_roi],[channel],mask,[hist_size],range_channel)

    # Histogram values are normalised to [0,255]
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

    return roi_hist