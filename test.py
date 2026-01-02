import cv2 as cv  # type: ignore
import numpy as np
import os


def preprocess_image(image):
    output = image.copy()

    hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

    dark_mask = cv.inRange(hsv, (0, 0, 0), (180, 90, 120))
    dark_mask = cv.morphologyEx(
        dark_mask,
        cv.MORPH_CLOSE,
        np.ones((10, 10), np.uint8),
        iterations=3
    )
    cv.imshow("Dark Mask", dark_mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cnts, _ = cv.findContours(dark_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_area = output.shape[0] * output.shape[1]
    print(f"Total contours found: {len(cnts)}")

    cards_info = []
    card_id = 1

    for c in cnts:
        area = cv.contourArea(c)
        if area < img_area * 0.01:
            continue

        rect = cv.minAreaRect(c)
        w_rot, h_rot = rect[1]
        if w_rot == 0 or h_rot == 0:
            continue

        aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot)
        if not (1.8 < aspect_ratio < 4.5):
            continue

        # Bounding box
        x, y, w, h = cv.boundingRect(c)

        # Extreme points (simple & stable)
        leftmost   = (x, y + h // 2)
        rightmost  = (x + w, y + h // 2)
        topmost    = (x + w // 2, y)
        bottommost = (x + w // 2, y + h)

        width_px  = rightmost[0] - leftmost[0]
        height_px = bottommost[1] - topmost[1]

        # Store card info
        card_data = {
            "id": card_id,
            "bbox": (x, y, w, h),
            "width_px": width_px,
            "height_px": height_px,
            "aspect_ratio": aspect_ratio,
            "area": area,
            "rot_width": w_rot,
            "rot_height": h_rot
        }
        cards_info.append(card_data)

        # Draw bounding box
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw label
        label = f"Card {card_id} | W:{width_px} H:{height_px} rot_w:{w_rot:.1f} rot_h:{h_rot:.1f}"
        cv.putText(
            output,
            label,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv.LINE_AA
        )

        print(
            f"Card {card_id}: "
            f"Width={width_px}px, Height={height_px}px, "
            f"Aspect={aspect_ratio:.2f}, Area={area:.0f}, "
            f"rot_w={w_rot:.1f}, rot_h={h_rot:.1f}"
        )

        card_id += 1

    cv.imshow("Detected Cards", output)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return output, cards_info


img = cv.imread("dataset/frame.jpg")
preprocessed_img, cards = preprocess_image(img)
