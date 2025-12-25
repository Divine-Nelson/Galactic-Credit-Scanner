# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 00:49:27 2025

@author: zahra
"""

import cv2 as cv #type: ignore
import numpy as np

img = cv.imread("5.bmp")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5,5), 0)

edges1 = cv.Canny(gray, 30, 100)
edges2 = cv.Canny(blur, 10, 60)
edges3 = cv.Canny(blur, 5, 30)

cv.imshow("GRAY", gray)
cv.imshow("BLUR", blur)
cv.imshow("CANNY (30,100)", edges1)
cv.imshow("CANNY (10,60)", edges2)
cv.imshow("CANNY (5,30)", edges3)

cv.waitKey(0)
