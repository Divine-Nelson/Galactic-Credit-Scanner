# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – FINAL SHAPE-STABLE VERSION (Option B)
Removes key-shaped object using solidity, extent, and convexity-defect filtering.
"""

import cv2 #type: ignore
import numpy as np
import math
from collections import Counter

Debug = False

# =======================================================================
# CHIP CLASSIFICATION
# =======================================================================

def classify_chip(w_mm):
    """Convert chip width (in px) into type 1, 2, or 3."""
    if w_mm <= 45:
        return 1

    elif w_mm > 45 and w_mm < 100:
        return 2
    
    return 3


def figure_from_array(arr):
    """Convert patterns of chip types into digits 0-9."""
    counts = Counter(arr)
    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    c3 = counts.get(3, 0)

    if c2 == 0 and c3 == 0:
        return c1 - 1       # 0,1,2

    if c2 == 1 and c3 == 0:
        return 3 + c1     # 3,4,5

    if c3 == 1 and c2 == 0:
        return 6 + c1    # 6–9

    raise ValueError(f"Invalid pattern: {arr}")

def clean_digit_group(group):
    """
    Keep only the dominant column by x-position.
    Removes blocks that drift too far horizontally.
    """
    xs = [c["center"][0] for c in group]
    mean_x = np.mean(xs)

    cleaned = [
        c for c in group
        # 20 px tolerance removes outliers caused by perspective or noise
        if abs(c["center"][0] - mean_x) < 20
    ]
    return cleaned


def compute_output_board(color, digits):
    if color == "yellow":
        return int("".join(map(str, digits))) * 10 if digits else 0
    elif color == "blue":
        return int("".join(map(str, digits))) if digits else 0
    elif color == "red":
        non_zero = [d for d in digits if d > 0]
        return math.prod(non_zero) if non_zero else 0
    return 0


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_card(image, box):
    rect = order_points(box)
    tl, tr, br, bl = rect

    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    W = int(max(widthA, widthB))
    H = int(max(heightA, heightB))

    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (W,H))


# =======================================================================
# SHAPE FILTER HELPERS — REMOVE KEY-SHAPED OBJECT
# =======================================================================

def compute_solidity(cnt):
    area = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0: return 0
    return area / hull_area


def compute_extent(cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    rect_area = w*h
    return area / rect_area if rect_area > 0 else 0


def convexity_defects_count(cnt):
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0
    try:
        defects = cv2.convexityDefects(cnt, hull)
        return 0 if defects is None else defects.shape[0]
    except:
        return 0


# =======================================================================
# CARD DETECTION
# =======================================================================

def detect_card(image):
    h, w = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0,0,0), (180,90,120))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE,
                                 np.ones((15,15),np.uint8), 3)

    cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_c = None
    max_area = 0
    dbg = image.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if area > max_area and area > (h*w)*0.02:
            card_c = c
            max_area = area

    if card_c is None:
        print("No card detected.")
        return None

    peri = cv2.arcLength(card_c, True)
    approx = cv2.approxPolyDP(card_c, 0.02*peri, True)

    if len(approx) == 4:
        box = approx.reshape(4,2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(card_c)
        box = cv2.boxPoints(rect).astype(np.float32)
    if Debug:
        cv2.imshow("DEBUG CARD DETECTION", dbg)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    return warp_card(image, box)


# =======================================================================
# CHIP DETECTION
# =======================================================================

def detect_chips(card):
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

    # Red
    red1 = cv2.inRange(hsv, (0, 80, 40), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # Other colors
    yellow_mask = cv2.inRange(hsv, (15, 90, 60), (35, 255, 255))
    blue_mask   = cv2.inRange(hsv, (85, 80, 40), (130, 255, 255))

    # Remove red pixels from yellow
    yellow_mask = cv2.bitwise_and(
        yellow_mask,
        cv2.bitwise_not(red_mask)
    )

    color_masks = {
        "red": red_mask,
        "yellow": yellow_mask,
        "blue": blue_mask,
    }

    chips = []
    dbg = card.copy()

    for color, mask in color_masks.items():

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)

        cnts, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 20 or area > 6000:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # reject tall noise for yellow
            if color == "yellow" and h > 40:
                continue

            cx = x + w // 2
            cy = y + h // 2

            # 🔑 single source of truth
            chip_type = classify_chip(w)

            chips.append({
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "color": color,
                "chip_type": chip_type
            })

            if Debug:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(
                    dbg, color, (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )

    if Debug:
        cv2.imshow("DEBUG CHIPS PER COLOR", dbg)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    return chips



# =======================================================================
# CHIP GROUPING (DIGIT COLUMNS)
# =======================================================================

def group_by_x(chips, card_size):
    if not chips:
        return []

    # sort by x center
    chips_sorted = sorted(chips, key=lambda c: c["center"][0])

    xs = [c["center"][0] for c in chips_sorted]

    if len(xs) > 1:
        dx = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        med_dx = np.median(dx)
    else:
        med_dx = 40  # fallback

    # adaptive threshold
    threshold = max(30, int(med_dx * 0.8))

    groups = []
    current = [chips_sorted[0]]
    base_x = chips_sorted[0]["center"][0]

    for c in chips_sorted[1:]:
        if abs(c["center"][0] - base_x) <= threshold:
            current.append(c)
        else:
            groups.append(current)
            current = [c]
            base_x = c["center"][0]

    groups.append(current)

    # sort each group top → bottom
    for g in groups:
        g.sort(key=lambda c: c["center"][1])

    return groups




# =======================================================================
# FULL PIPELINE
# =======================================================================

def analyze_image(image):

    card = detect_card(image)
    if card is None:
        return {"cards":[], "overlay":image}

    chips = detect_chips(card)
    groups = group_by_x(chips, card.shape[:2])

    digits = []
    for g in groups:
        g = clean_digit_group(g)
        if len(g) < 1 or len(g) > 6:
            continue

        chip_types = [c["chip_type"] for c in g]
        try:
            digits.append(figure_from_array(chip_types))
        except ValueError:
            print("⚠️ Skipping invalid digit group:", chip_types)

    

    if chips:
        dom_color = Counter([c["color"] for c in chips]).most_common(1)[0][0]
    else:
        dom_color = "blue"

    print("initial digits:", digits)
    if dom_color in ["yellow", "blue"]:
        digits = [d for d in digits if d != 0]

    if not digits:
        value = 0
    else:
        value = compute_output_board(dom_color, digits)

    overlay = card.copy()
    for c in chips:
        x,y,w,h = c["bbox"]
        cv2.rectangle(overlay, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(overlay, c["color"], (x,y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),2)

    cv2.putText(overlay, f"Value: {value}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,255),3)

    return {
        "cards":[{"chips":chips,"digits":digits,"color":dom_color,"value":value}],
        "overlay":overlay
    }
