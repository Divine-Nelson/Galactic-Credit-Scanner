# -*- coding: utf-8 -*-
"""
Utility detection methods for preprocessing and contour extraction.
Author: Divine
"""

import cv2 as cv
import numpy as np


class MyDetectionMethods:

    def canny_edges(self, image):
        """Return edges using Canny + Gaussian blur."""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        blur = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blur, 50, 150)

        return edges

    def adaptive_thresh_contours(self, image):
        """
        Fallback contour detection using adaptive-threshold binarization.
        Useful for finding card shape if edge detection fails.
        """
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            gray = image

        blur = cv.GaussianBlur(gray, (5, 5), 0)

        thresh = cv.adaptiveThreshold(
            blur, 255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        
        cv.imshow("CardThresh", thresh)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
        return contours

    def hsv_mask(self, image, lower, upper):
        """Return HSV mask for a given range."""
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        return cv.inRange(hsv, lower, upper)
