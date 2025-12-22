# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:42:32 2025

@author: zahra
"""

import cv2 as cv
import numpy as np

def image_processing(image):
    # Step 1: grayscale
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
        
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Step 2: blur
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: adaptive threshold
    thresh = cv.adaptiveThreshold(
        blur, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        11, 2
    )
    
    # Step 4: find contours
    contours, _ = cv.findContours(
        thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    
    
    valid = []
    for c in contours:
        area = cv.contourArea(c)
        if area < 500:
            continue

        valid.append(c)
            
    if not valid:
        print("No credit detected!")
        return
    
    cnt = max(valid, key=cv.contourArea)
    
    x, y, w, h = cv.boundingRect(cnt)
    roi = image[y:y+h, x:x+w]
    
    debug = image.copy()
    cv.rectangle(debug, (x, y), (x+w, y+h), (0, 0, 255), 2)    
   
    
    cv.imshow("Gray", gray)
    cv.waitKey(0)
    
    cv.imshow("Threshold", thresh)
    cv.waitKey(0)
    
    cv.imshow("Debug - Bounding Box", debug)
    cv.waitKey(0)
    
    cv.imshow("Detected Credit ROI", roi)
    cv.waitKey(0)
    
    cv.destroyAllWindows()


    
    
    
img = cv.imread("1.bmp")
cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()
image_processing(img)