# -*- coding: utf-8 -*-
"""
Main pipeline for Star Wars Credit Detection
Author: Divine
"""

import cv2 as cv
import numpy as np
from features import MyDetectionMethods
from collections import Counter
import math

detector = MyDetectionMethods()

CARD_W_MM = 100   # Physical width of the credit in millimeters


# ---------------------------------------------------------
#  Block Width Classification → chip type
# ---------------------------------------------------------
def classify_chip(w):
    if w <= 45:
        return 1
    elif 45 < w < 100:
        return 2
    elif w >= 100:
        return 3


# ---------------------------------------------------------
# Convert array of chip types → digit
# ---------------------------------------------------------
def figure_from_array(arr):
    counts = Counter(arr)

    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    c3 = counts.get(3, 0)

    # Only 1s  → digits 0,1,2
    if c2 == 0 and c3 == 0:
        return c1 - 1

    # One 2 and some 1s → digits 3,4,5
    if c2 == 1 and c3 == 0:
        return 3 + c1

    # One 3 and some 1s → digits 6,7,8,9
    if c3 == 1 and c2 == 0:
        return 6 + c1

    raise ValueError("Invalid block combination")


# ---------------------------------------------------------
# Detect credit card as reference object for size
# ---------------------------------------------------------
def find_reference_object(image):

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Lower V threshold to ONLY isolate the black/gray card body
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 45])

    mask = cv.inRange(hsv, lower_dark, upper_dark)

    # Fill holes caused by texture
    kernel = np.ones((35,35), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No card detected.")
        cv.imshow("CardMask", mask)
        return None, None

    cnt = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)

    pixel_to_mm = CARD_W_MM / max(w, h)

    debug = image.copy()
    cv.rectangle(debug, (x, y), (x+w, y+h), (0,255,0), 3)
    cv.imshow("CardMask", mask)
    cv.imshow("DetectedCard", debug)
    cv.waitKey(0)

    return (x, y, w, h), pixel_to_mm




# ---------------------------------------------------------
# Detect red blocks INSIDE the ROI of the credit card
# ---------------------------------------------------------
def detect_objects(image, card_bbox, pixel_to_mm):

    cx, cy, cw, ch = card_bbox

    # Crop card region
    card = image[cy:cy+ch, cx:cx+cw]

    # -----------------------------
    # HSV mask for yellow/orange blocks
    # -----------------------------
    hsv = cv.cvtColor(card, cv.COLOR_BGR2HSV)
    v_mean = np.mean(hsv[:,:,2])

    # Dynamic saturation threshold
    s_min = 60 if v_mean < 80 else 40

    # Yellow/orange range (your blocks)
    lowerYellow = np.array([10, s_min, 40])
    upperYellow = np.array([35, 255, 255])
    maskYellow = cv.inRange(hsv, lowerYellow, upperYellow)

    # Red-orange range (sometimes appears red in low light)
    lowerRed = np.array([0, s_min, 40])
    upperRed = np.array([10, 255, 255])
    maskRed = cv.inRange(hsv, lowerRed, upperRed)
    
    # Blue blocks (for 12.bmp)
    lowerBlue = np.array([85, 60, 40])
    upperBlue = np.array([130, 255, 255])
    maskBlue = cv.inRange(hsv, lowerBlue, upperBlue)
    
    mask = cv.bitwise_or(maskYellow, maskRed)
    mask = cv.bitwise_or(mask, maskBlue)

    # Clean small noise, but keep thin blocks
    kernel = np.ones((3,3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)

    # Find block contours
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    blocks = []
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        if w*h < 100:  # keep small blocks too
            continue
        blocks.append((x, y, w, h))

    if not blocks:
        print("No blocks found inside card!")
        cv.imshow("BlockMask", mask)
        return

    # Sort left → right
    blocks = sorted(blocks, key=lambda b: b[0])

    # -------------------------------------------------
    #  Group blocks into columns using overlap
    # -------------------------------------------------
    columns = []
    current_col = [blocks[0]]

    for b in blocks[1:]:
        prev = current_col[-1]

        # Horizontally close enough?
        if abs(b[0] - prev[0]) < (prev[2] * 1.5):
            current_col.append(b)
        else:
            columns.append(current_col)
            current_col = [b]

    columns.append(current_col)

    # -------------------------------------------------
    # Decode digits
    # -------------------------------------------------
    digits = []

    for col in columns:
        col_sorted = sorted(col, key=lambda b: b[1])  # top → bottom

        # Convert pixel width → mm
        widths_mm = [ (w * pixel_to_mm) for (_,_,w,_) in col_sorted ]

        # Classify each block
        chip_types = [ classify_chip(w) for w in widths_mm ]

        # Convert block pattern → digit
        digit = figure_from_array(chip_types)
        digits.append(digit)

    detected_value = int("".join(map(str, digits)))
    print("Detected credit value:", detected_value)

    # -------------------------------------------------
    # Draw boxes
    # -------------------------------------------------
    for (x, y, w, h) in blocks:
        cv.rectangle(card, (x,y), (x+w, y+h), (0,255,0), 2)

    cv.putText(card, f"Value: {detected_value}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv.imshow("BlockMask", mask)
    cv.imshow("Card Blocks", card)
    cv.waitKey(0)



# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------
image = cv.imread("4.bmp")
cv.imshow("Debug Raw", image)
cv.waitKey(0)


if image is None:
    raise FileNotFoundError("Image not found")

card_bbox, pixel_to_mm = find_reference_object(image)

if card_bbox is None:
    print("No card detected.")
else:
    print(f"Scale factor (mm per pixel): {pixel_to_mm:.4f}")

    x, y, w, h = card_bbox
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv.putText(image, "Credit Card", (x, y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    detect_objects(image, card_bbox, pixel_to_mm)

cv.imshow("Result", image)
cv.waitKey(0)
cv.destroyAllWindows()

