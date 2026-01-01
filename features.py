# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – FEATURES (Stable Version)
Author: Divine Ezeilo
"""

import cv2  # type: ignore
import numpy as np
import math
from collections import Counter

# ==========================================================
# CARD DETECTION
# ==========================================================

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

        w = max(rw, rh)
        h = min(rw, rh)
        ratio = w / (h + 1e-6)

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
                "rot_w": w,
                "rot_h": h,
                "ratio": ratio
            }
        })

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
                "center": (x + w // 2, y + h // 2)
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
    ratio = metrics["ratio"]
    area  = metrics["bbox_w"] * metrics["bbox_h"]
    width = metrics["bbox_w"]
    height= metrics["bbox_h"]

    # -------------------------------
    # SHAPE (aspect ratio)
    # -------------------------------
    if dom_color == "red" and not (1.9 <= ratio <= 2.7):
        return True, "Aspect ratio mismatch"

    if dom_color == "yellow" and not (2.7 <= ratio <= 3.9):
        return True, "Aspect ratio mismatch"

    if dom_color == "blue" and not (3.8 <= ratio <= 5.0):
        return True, "Aspect ratio mismatch"

    # -------------------------------
    # SIZE (THIS IS WHAT YOU WERE MISSING)
    # -------------------------------
    MIN_SIZE = {
        "red":    {"area": 93000, "width": 380,},
        "yellow": {"area": 130000, "width": 420},
        "blue":   {"area": 150000, "width": 460},
    }
    min_height = 190  # Minimum height for all cards
    limits = MIN_SIZE[dom_color]

    if width < limits["width"] or area < limits["area"] or height < min_height:
        return True, "Card too small"

    # -------------------------------
    # DIGITS
    # -------------------------------
    if invalid_digit:
        return True, "Invalid digit encoding"

    value = compute_output_board(dom_color, digits)
    if value == 0:
        return True, "Zero value"

    return False, "Real"
