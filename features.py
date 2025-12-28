# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – FINAL SHAPE-STABLE VERSION (Option B)
Removes key-shaped object using solidity, extent, and convexity-defect filtering.
"""

from matplotlib.pylab import det
import cv2 #type: ignore
import numpy as np
import math
from collections import Counter

Debug = False
REF_CARD = {
    "w": None,
    "h": None,
    "ratio": None
}

SIZE_TOL = 0.05  # ±5%




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
        dark_mask,
        cv2.MORPH_CLOSE,
        np.ones((10, 10), np.uint8),
        iterations=3
    )

    cv2.imshow("Dark Mask", dark_mask)
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    cnts, _ = cv2.findContours(
        dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cards = []
    img_h, img_w = image.shape[:2]
    img_area = img_h * img_w

    for card_c in cnts:
        area = cv2.contourArea(card_c)
        if area < img_area * 0.01:
            continue

        # ---- geometry (ROTATION AWARE) ----
        peri = cv2.arcLength(card_c, True)
        approx = cv2.approxPolyDP(card_c, 0.02 * peri, True)

        if len(approx) == 4:
            box = approx.reshape(4, 2).astype(np.float32)
            w = np.linalg.norm(box[0] - box[1])
            h = np.linalg.norm(box[1] - box[2])
        else:
            rect = cv2.minAreaRect(card_c)
            box = cv2.boxPoints(rect).astype(np.float32)
            w, h = rect[1]

        if w == 0 or h == 0:
            continue

        x,y,w,h = cv2.boundingRect(card_c)
        """rect = cv2.minAreaRect(card_c)
        (_, _), (w, h), _ = rect
        h = min(w, h)
        w = max(w, h)"""

        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

        # ---- metrics ----
        card_metrics = { 
            "bbox_x": x, 
            "bbox_y": y, 
            "bbox_w": w, 
            "bbox_h": h, 
            "aspect_ratio": aspect_ratio, 
            "area": area, 
            "rel_w": w / img_w, 
            "rel_h": h / img_h, 
            "rel_area": area / (img_w * img_h) 
        }
        if not (1.8 < aspect_ratio < 4.5):
            continue

        warped = warp_card(image, box)
        size = (warped.shape[1], warped.shape[0])
        cards.append({
            "warped": warped,
            "metrics": card_metrics,
            "contour": card_c,
            "box": box,
            "size": size
        })        

    if not cards:
        cnt_masks = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in cnt_masks:
            area = cv2.contourArea(c)
            if area < img_area * 0.01:
                continue
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)
            warped = warp_card(image, box)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [box.astype(np.int32)], 255)
            size = (warped.shape[1], warped.shape[0])
            cards.append({
                "warped": warped,
                "metrics": card_metrics,
                "contour": c,
                "box": box,
                "size": size
            })
    cards.sort(key=lambda c: c['size'][0]*c['size'][1], reverse=True)


    print(f"Detected {len(cards)} cards.")
    print(f"Card metrics: {[card['metrics'] for card in cards]}\n")

    return cards


def detect_cards(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark_mask = cv2.inRange(hsv, (0,0,0), (180,90,120))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE,
                                 np.ones((15,15),np.uint8), 3)

    cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    img_area = image.shape[0] * image.shape[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.02:
            continue

        x,y,w,h = cv2.boundingRect(c)
        ratio = w / h

        card_metrics = {
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "aspect_ratio": ratio,
            "area": area
        }


        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            box = approx.reshape(4,2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(np.float32)

        warped = warp_card(image, box)

        cards.append({
            "warped": warped,
            "metrics": card_metrics,
            "contour": c
        })

    print(f"Detected {len(cards)} cards.")
    print(f"Card metrics: {[card['metrics'] for card in cards]}/n")
    return cards


def size_mismatch(card_metrics):
    rel_w = card_metrics["rel_w"]
    rel_h = card_metrics["rel_h"]
    ratio = card_metrics["aspect_ratio"]

    # Physical card constraints (scale invariant)
    if not (0.16 <= rel_h <= 0.24):
        return True

    # --- WIDTH CLASSES (multiple valid formats)
    valid_width = (
        0.20 <= rel_w <= 0.45 or   # small
        0.45 <= rel_w <= 0.60 or   # medium
        0.60 <= rel_w <= 0.80      # large / silver
    )
    if not valid_width:
        return True

    if not (2.0 <= ratio <= 4.8):
        return True

    return False





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

def detect_many_chips(cards):
    card_mask = []
    for card in cards:
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
        card_mask.append(chips)
    return card_mask


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
# Detect Fake Card
# =======================================================================

def is_fake_card(card, chips, dom_color, invalid_digit):
    if card is None:
        return True, "No card"

    if size_mismatch(card):
        return True, "Size mismatch"

    for c in chips:
        if c["color"] != dom_color:
            return True, "Color misalignment"

    if invalid_digit:
        return True, "Invalid digit encoding"

    return False, "Real"



# =======================================================================
# FULL PIPELINE
# =======================================================================

def analyze_image(image):
    cards = detect_many_cards(image)
    if not cards:
        return {"cards": [], "overlay": image, "dbg_card": None}

    overlay = image.copy()
    results = []
    card_colors = []
    card_values = []

    for card in cards:
        warped = card["warped"]
        metrics = card["metrics"]

        invalid_digit = False

        # 1. Detect chips
        chips = detect_chips(warped)

        # 2. Dominant color
        if chips:
            dom_color = Counter([c["color"] for c in chips]).most_common(1)[0][0]
        else:
            dom_color = "blue"

        # 3. Decode digits
        groups = group_by_x(chips, warped.shape[:2])
        digit_groups = []
        digits = []

        for g in groups:
            g = clean_digit_group(g)
            if len(g) < 1 or len(g) > 6:
                continue

            chip_types = [c["chip_type"] for c in g]
            try:
                digits.append(figure_from_array(chip_types))
            except ValueError:
                invalid_digit = True

        # 4. Fake detection
        is_fake = False
        reason = "Real"

        if size_mismatch(metrics):
            is_fake = True
            reason = "Size mismatch"
        elif invalid_digit:
            is_fake = True
            reason = "Invalid digit encoding"
        else:
            for c in chips:
                if c["color"] != dom_color:
                    is_fake = True
                    reason = "Color misalignment"
                    break

        # 5. Compute value
        if is_fake:
            value = 0
        else:
            if dom_color in ["yellow", "blue"]:
                digit_groups.append(digits)
                digits = [d for d in digits if d != 0]
            value = compute_output_board(dom_color, digits)

        # 6. Draw bounding box on ORIGINAL image
        x, y, w, h = (
            metrics["bbox_x"],
            metrics["bbox_y"],
            metrics["bbox_w"],
            metrics["bbox_h"]
        )

        color = (0, 0, 255) if is_fake else (0, 255, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)

        label = f"FAKE ({reason})" if is_fake else f"Value: {value}"
        cv2.putText(
            overlay,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        results.append({
            "digits_groups": digit_groups,
            "chips": chips,
            "digits": digits,
            "color": dom_color,
            "value": value,
            "fake": is_fake
        })
    total_value = 0
    card_digits = []
    
    for r in results:
        total_value += r["value"]
        card_digits.extend(r["digits"])
    print(f"Total Value: {total_value}")

    cv2.putText(
        overlay,
        f"Total Value: {total_value}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 0),
        2
    ) 

    return {
        "cards": results,
        "overlay": overlay,
        "dbg_card": None,
        "total_value": total_value,
    }
