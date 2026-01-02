# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – FEATURES (Stable Version)
Author: Divine Ezeilo
"""

import cv2  # type: ignore
import numpy as np
import math
from collections import Counter, deque

#GLobal Constants
BLUE_WIDTH_REF = deque(maxlen=3)


# ==========================================================
# CARD DETECTION
# ==========================================================
def detect_card(image):
    """
    Detect the card, return:
    - warped card (for chip detection)
    - debug image (original with contour)
    - card metrics for fake detection
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Dark card segmentation
    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 90, 120))
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        np.ones((15, 15), np.uint8),
        iterations=3
    )

    cnts, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not cnts:
        return None, None, None
    

    # 2. Pick largest contour (the card)
    card_c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(card_c)

    # Reject tiny false positives
    if area < image.shape[0] * image.shape[1] * 0.02:
        return None, None, None

    # 3. Bounding box → SIZE MEASUREMENT (important!)
    x, y, w, h = cv2.boundingRect(card_c)
    aspect_ratio = w / h

    card_metrics = {
        "bbox_w": w,
        "bbox_h": h,
        "aspect_ratio": aspect_ratio,
        "area": area
    }

    # 4. Perspective warp for later chip detection
    peri = cv2.arcLength(card_c, True)
    approx = cv2.approxPolyDP(card_c, 0.02 * peri, True)

    if len(approx) == 4:
        box = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(card_c)
        box = cv2.boxPoints(rect).astype(np.float32)

    warped = warp_card(image, box)

    # 5. Debug image (persistent contour)
    dbg = image.copy()
    cv2.drawContours(dbg, [card_c], -1, (0, 255, 0), 2)
    cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print(card_metrics)
    return warped, dbg, card_metrics


def detect_many_cards(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 90, 120))
    dark_mask = cv2.morphologyEx(
        dark_mask, cv2.MORPH_CLOSE,
        np.ones((15, 15), np.uint8), 3
    )

    cnts, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w
    cards = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.01:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), _ = rect
        if rw == 0 or rh == 0:
            continue

        rot_w = max(rw, rh)
        rot_h = min(rw, rh)
        ratio = rot_w / (rot_h + 1e-6)

        if not (1.8 <= ratio <= 4.9):
            continue

        box = cv2.boxPoints(rect).astype(np.float32)
        warped = warp_card(image, box)

        cards.append({
            "warped": warped,
            "metrics": {
                "bbox_x": bx,
                "bbox_y": by,
                "bbox_w": bw,
                "bbox_h": bh,
                "rot_w": rot_w,
                "rot_h": rot_h,
                "ratio": ratio,
                "rel_area": area / img_area,
                "area": area
            },
            "contours": c
        })

    """for card in cards:
        rot_w = card["metrics"]["rot_w"]
        print(rot_w)"""

    return cards


# ==========================================================
# PERSPECTIVE HELPERS
# ==========================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_card(image, box):
    rect = order_points(box)
    (tl, tr, br, bl) = rect

    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (W, H))


# ==========================================================
# CHIP DETECTION
# ==========================================================

def classify_chip(w_px):
    if w_px <= 45:
        return 1
    elif w_px < 100:
        return 2
    else:
        return 3


def detect_chips(card):
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 70, 40), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 40), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    yellow_mask = cv2.inRange(hsv, (15, 80, 60), (40, 255, 255))
    blue_mask = cv2.inRange(hsv, (90, 70, 40), (140, 255, 255))

    masks = {
        "red": red_mask,
        "yellow": yellow_mask,
        "blue": blue_mask
    }

    chips = []

    for color, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if not (20 <= cv2.contourArea(c) <= 6000):
                continue

            x, y, w, h = cv2.boundingRect(c)

            chips.append({
                "color": color,
                "chip_type": classify_chip(w),
                "center": (x + w // 2, y + h // 2),
                "bbox": (x, y, w, h),
                "w": w,
                "h": h,
            })

    return chips


# ==========================================================
# DIGIT DECODING
# ==========================================================

def group_by_x(chips):
    chips = sorted(chips, key=lambda c: c["center"][0])
    groups, cur = [], []

    for c in chips:
        if not cur or abs(c["center"][0] - cur[-1]["center"][0]) < 40:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]

    if cur:
        groups.append(cur)
    return groups


def figure_from_array(arr):
    counts = Counter(arr)
    if counts.get(3, 0):
        return 6 + counts.get(1, 0)
    if counts.get(2, 0):
        return 3 + counts.get(1, 0)
    return counts.get(1, 0) - 1


def decode_digits(chips):
    digits, invalid = [], False

    for g in group_by_x(chips):
        if len(g) == 1:
            continue
        try:
            digits.append(figure_from_array([c["chip_type"] for c in g]))
        except Exception:
            invalid = True

    return digits, invalid


def compute_output_board(color, digits):
    if not digits:
        return 0
    if color == "yellow":
        return int("".join(map(str, digits))) * 10
    if color == "blue":
        return int("".join(map(str, digits)))
    if color == "red":
        return math.prod(d for d in digits if d > 0)
    return 0


# ==========================================================
# FAKE CHECK (GEOMETRY-FIRST)
# ==========================================================

def is_fake_card(metrics, chips, dom_color, digits, invalid_digit):
    rot_w  = metrics["rot_w"]
    rot_h  = metrics["rot_h"]
    ratio  = metrics["ratio"]
    area   = metrics["area"]

    # -------------------------------
    # SHAPE (rotation invariant)
    # -------------------------------
    if dom_color == "red":
        if not (1.8 <= ratio <= 2.7):
            return True, "Aspect ratio mismatch"

        if rot_h > 220:
            return True, "Red card too thick"

        if area > 100_000:
            return True, "Red card too large"

    elif dom_color == "yellow":
        if not (2.4 <= ratio <= 3.9):
            return True, "Aspect ratio mismatch"

        if rot_h > 245:
            return True, "Yellow card too thick"

        if area > 140_000:
            return True, "Yellow card too large"

    elif dom_color == "blue":
        if not (3.0 <= ratio <= 5.0):
            return True, "Aspect ratio mismatch"

        if not (810 < rot_w < 1100):
            return True, "Blue card width mismatch"

        if rot_h > 250:
            return True, "Blue card too thick"

        if area > 190_000:
            return True, "Blue card too large"

    # -------------------------------
    # MINIMUM SIZE (rotation invariant)
    # -------------------------------
    MIN_AREA = {
        "red":    78_000,
        "yellow": 110_000,
        "blue":   140_000,
    }

    MIN_HEIGHT = {
        "red":    110,
        "yellow": 140,
        "blue":   160,
    }

    if area < MIN_AREA[dom_color]:
        return True, "Card too small"

    if rot_h < MIN_HEIGHT[dom_color]:
        return True, "Card too thin"

    # -------------------------------
    # DIGITS
    # -------------------------------
    if invalid_digit:
        return True, "Invalid digit encoding"

    value = compute_output_board(dom_color, digits)
    if value == 0:
        return True, "Zero value"

    return False, "Real"
