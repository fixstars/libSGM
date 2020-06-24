#!/usr/bin/env python3

import sys

import cv2
import numpy as np

import pysgm

sgm_opencv_wrapper = pysgm.LibSGMWrapper(
    numDisparity=int(128), P1=int(10), P2=int(120), uniquenessRatio=np.float32(0.95),
    subpixel=False, pathType=pysgm.PathType.SCAN_8PATH, minDisparity=int(0),
    lrMaxDiff=int(1))


I1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

disp = sgm_opencv_wrapper.execute(I1, I2)

if sgm_opencv_wrapper.hasSubpixel():
    disp = disp.astype('float') / pysgm.SUBPIXEL_SCALE()

disp_color = cv2.applyColorMap(disp.astype("uint8"), cv2.COLORMAP_JET)

if sgm_opencv_wrapper.hasSubpixel():
    disp_color[disp <= sgm_opencv_wrapper.getInvalidDisparity() / pysgm.SUBPIXEL_SCALE(), :] = 0
else:
    disp_color[disp <= sgm_opencv_wrapper.getInvalidDisparity(), :] = 0

cv2.imshow("disp_color", disp_color)
cv2.waitKey(0)
