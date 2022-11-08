#!/usr/bin/env python3

import sys

import cv2
import numpy as np

import pysgm

disp_size = 128

I1 = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

disp = np.zeros_like(I2)

I1_ptr, _ = I1.__array_interface__['data']
I2_ptr, _ = I2.__array_interface__['data']
disp_ptr, _ = disp.__array_interface__['data']

height, width = I1.shape

params = pysgm.StereoSGM.Parameters(P1=int(10), P2=int(120), uniqueness=np.float32(0.95), subpixel=False,
                                    PathType=pysgm.PathType.SCAN_8PATH, min_disp=int(0), LR_max_diff=int(1))

sgm = pysgm.StereoSGM(width=width, height=height, disparity_size=int(disp_size), input_depth_bits=int(8),
                      output_depth_bits=int(8), inout_type=pysgm.EXECUTE_INOUT.EXECUTE_INOUT_HOST2HOST,
                      param=params)

sgm.execute(I1_ptr, I2_ptr, disp_ptr)

mask = disp == np.uint8(sgm.get_invalid_disparity())
disp = (255. * disp / disp_size)

disp_color = cv2.applyColorMap(disp.astype("uint8"), cv2.COLORMAP_JET)
disp_color[mask] = 0

cv2.imshow("disp_color", disp_color)
cv2.waitKey(0)
