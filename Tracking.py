import numpy as np
import cv2
from Hough import *
from Images_functions import *
from Visualization import *

################
#### Parameters
################

# Initial Parameters
roi_defined = False
tsh= 45/255
decay = 0.8
MS_ite = 20
dis_ang = 360

# Update Histograms
update_hist_Hue = False
update_hist_Sat = False

# Update R-Table
update_hg = False

# Turn on intersection of H and S (HSV)
Comb_HSV_update = False

# Turn on combination of intersection and Hough
Comb_all_update = False 

################
#### Videos
################

cap = cv2.VideoCapture('./Sequences/VOT-Ball.mp4')
# cap = cv2.VideoCapture('./Sequences/Antoine_Mug.mp4')
# cap = cv2.VideoCapture('./Sequences/VOT-Woman.mp4')
# cap = cv2.VideoCapture('./Sequences/VOT-Basket.mp4')
# cap = cv2.VideoCapture('./Sequences/VOT-Car.mp4')
# cap = cv2.VideoCapture('./Sequences/VOT-Sunshade.mp4')

########################################################################
#### Functions
########################################################################

# Define Region of Interest (ROI) 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined

	# Left mouse button clicked record the starting ROI coordinates
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False

	# Left mouse button released record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)
		roi_defined = True

# Combination of H and S (HSV)
def Combine_HSV(bp_Hue, bp_Sat):
    bp_Hue = cv2.convertScaleAbs(255*normalize(bp_Hue))
    bp_Sat = cv2.convertScaleAbs(255*normalize(bp_Sat))

    _, mask_and = cv2.threshold(bp_Sat, 30, 255, cv2.THRESH_BINARY)
    Intersection = cv2.bitwise_and(bp_Sat, bp_Hue, mask = mask_and)
    Intersection = cv2.convertScaleAbs(255*normalize(Intersection))
    return Intersection

# Combination of transformations 
def Combine_all(bp_Hue, bp_Sat, HG):
    HG = cv2.convertScaleAbs(255*normalize(HG))
    Intersection = Combine_HSV(bp_Hue, bp_Sat)
    Comb = Intersection + HG
    normalize(Comb)
    return Comb

########################################################################
#### Main
########################################################################

# Take first frame of the video
ret,frame = cap.read()

# Load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# Print Hint
print("Select the region of interest and press 'Q' to continue.")


# Keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break

# Print Hint
print("Press 'S' to save frame.")

# Track Windows
track_window_hist_Hue = (r,c,h,w)
track_window_hist_Sat = (r,c,h,w)
track_window_HG = (r,c,h,w)
track_window_HG_0 = (r,c,h,w)   
track_window_Comb = (r,c,h,w)
track_window_Intersection = (r,c,h,w)

# ROI for HSV histograms
roi_hist_Hue = hist_roi(frame[c:c+w, r:r+h], 0, 180,[0,180])
roi_hist_Sat = hist_roi(frame[c:c+w, r:r+h], 1, 255, [50, 255])

# Calculate R-Table 
gray_scale = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
arg_frame = arg_grad(gray_scale)
mod_frame = mod_grad(gray_scale)
_, idxs_roi = pixel_selection(mod_frame.copy(), mod_frame[c:c+w, r:r+h].copy(), tsh)
R_Table = create_R_Table(arg_frame[c:c+w, r:r+h].copy(), idxs_roi, dis_ang)

# Setup the termination criteria: either 10 iterations, or move by less than 1 pixel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, MS_ite, 1 )

cpt = 1
while(1):
    ret ,frame = cap.read()

    if ret == True:
        
        #############################################
        #### Update windows for combination
        #############################################

        if Comb_HSV_update:
            track_window_hist_Hue = track_window_Intersection
            track_window_hist_Sat = track_window_Intersection

        if Comb_all_update:
            track_window_hist_Hue = track_window_Comb
            track_window_hist_Sat = track_window_Comb
            track_window_HG = track_window_Comb

        #############################################
        #### Histogram
        #############################################

        hsv = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
		
        # Backproject the model histogram roi_hist onto the current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
        bp_Hue = cv2.calcBackProject([hsv],[0],roi_hist_Hue,[0,180],1)
        bp_Sat = cv2.calcBackProject([hsv],[1],roi_hist_Sat,[50,255],1)
   
        #############################################
        #### Hough
        #############################################

        r,c,h,w = track_window_HG

        # Norme and orientation of Frame and ROI
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arg_frame = arg_grad(gray_scale)
        mod_frame = mod_grad(gray_scale)

        # Pixel Selection
        idxs_frame, idxs_roi = pixel_selection(mod_frame, mod_frame[c:c+w, r:r+h], tsh)


        # Generalized Hough
        HG = Gen_Hough_Transform(R_Table, arg_frame, idxs_frame, dis_ang, decay)
        track_window_HG_0 = Track_Hough(HG, track_window_HG_0)
        tracked_HG_0 = track_display(frame, track_window_HG_0)

        #############################################
        #### Combination
        #############################################
        
        if Comb_HSV_update:
            Intersection = Combine_HSV(bp_Hue, bp_Sat)
        
        if Comb_all_update:
            Comb = Combine_all(bp_Hue, bp_Sat, HG)

        #############################################
        #### Mean-Shift
        #############################################

        # Hue
        ret, track_window_hist_Hue = cv2.meanShift(bp_Hue, track_window_hist_Hue, term_crit)
        MS_tracked_hist_Hue = track_display(frame, track_window_hist_Hue)

        # Saturation
        ret, track_window_hist_Sat = cv2.meanShift(bp_Sat, track_window_hist_Sat, term_crit)
        MS_tracked_hist_Sat = track_display(frame, track_window_hist_Sat)

        # Hough    
        ret, track_window_HG = cv2.meanShift(HG, track_window_HG, term_crit)
        MS_tracked_HG = track_display(frame, track_window_HG)

        # Combined  HSV
        if Comb_HSV_update:
            ret, track_window_Intersection = cv2.meanShift(Intersection, track_window_Intersection, term_crit)
            MS_tracked_Intersection = track_display(frame, track_window_Intersection)

        # Combined  all
        if Comb_all_update:
            ret, track_window_Comb = cv2.meanShift(Comb, track_window_Comb, term_crit)
            MS_tracked_Comb = track_display(frame, track_window_Comb)

        #############################################
        #### Updates
        #############################################

        # Update Hue Histogram
        if update_hist_Hue == 1:
            # Rectangle
            r,c,h,w = track_window_hist_Hue
            # Update the ROI
            roi_hist_Hue = hist_roi(frame[c:c+w, r:r+h], 0, 180,[0,180])

        # Update Saturation Histogram
        if update_hist_Sat == 1:
            # Rectangle
            r,c,h,w = track_window_hist_Sat
            # Update the ROI
            roi_hist_Sat = hist_roi(frame[c:c+w, r:r+h], 1, 255, [30, 255])

        # R-Table update 
        if update_hg or cpt == 1:
            R_Table = create_R_Table(arg_frame[c:c+w, r:r+h], idxs_roi, dis_ang)

        #############################################
        #### Images
        #############################################

        # Orientation and Module Transform Show
        # cv2.normalize(arg_frame, arg_frame,0,255,cv2.NORM_MINMAX)
        # arg_frame = cv2.convertScaleAbs(arg_frame)
        # cv2.imshow('Arg ', arg_frame)
        img_show('Arg ', arg_frame)
        img_show('Mod ', mod_frame)

        # Pixel Selection Show
        best_arg_img = best_arg_show(mod_frame,arg_frame, tsh)
        cv2.imshow('Sequence Pixel Votant',best_arg_img)

        # Show Histogram Sequence
        img_show('Hist Hue', bp_Hue)
        img_show('Hist Saturation', bp_Sat)

        # Show Hough
        img_show('HG',HG)
        img_show('track HG',tracked_HG_0)

        # Mean-Shift
        img_show('Mean Shift Hist Hue', MS_tracked_hist_Hue)
        img_show('Mean Shift Hist Sat', MS_tracked_hist_Sat)  
        img_show('Mean Shift HG', MS_tracked_HG)
        
        # Show HS Combination
        if Comb_HSV_update:
            img_show('Intersection', Intersection)
            img_show('Mean Shift Intersection', MS_tracked_Intersection)
        
        # Show HS and Hough Combination
        if Comb_all_update:
            img_show('Comb', Comb)
            img_show('Mean Shift Comb', MS_tracked_Comb)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('./Images/frame_%04d.png'%cpt,normalize_abs(frame))
            cv2.imwrite('./Images/mod_%04d.png'%cpt,normalize_abs(mod_frame))
            cv2.imwrite('./Images/arg_%04d.png'%cpt,normalize_abs(arg_frame))
            cv2.imwrite('./Images/best_arg_img_%04d.png'%cpt,normalize_abs(best_arg_img))
            cv2.imwrite('./Images/Intersection_%04d.png'%cpt,normalize_abs(Intersection))
            cv2.imwrite('./Images/Comb_%04d.png'%cpt,normalize_abs(Comb))
            cv2.imwrite('./Images/bp_Hue%04d.png'%cpt,normalize_abs(bp_Hue))
            cv2.imwrite('./Images/bp_Sat%04d.png'%cpt,normalize_abs(bp_Sat))
            cv2.imwrite('./Images/hough_%04d.png'%cpt,normalize_abs(HG))
            cv2.imwrite('./Images/tracked_HG_0_%04d.png'%cpt,normalize_abs(tracked_HG_0))
            cv2.imwrite('./Images/MS_tracked_hist_Hue_%04d.png'%cpt,normalize_abs(MS_tracked_hist_Hue))
            cv2.imwrite('./Images/MS_tracked_hist_Sat_%04d.png'%cpt,normalize_abs(MS_tracked_hist_Sat))
            cv2.imwrite('./Images/MS_tracked_HG_%04d.png'%cpt,normalize_abs(MS_tracked_HG))
            cv2.imwrite('./Images/MS_tracked_Comb_%04d.png'%cpt,normalize_abs(MS_tracked_Intersection))
            cv2.imwrite('./Images/MS_tracked_Comb_%04d.png'%cpt,normalize_abs(MS_tracked_Comb))

        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
