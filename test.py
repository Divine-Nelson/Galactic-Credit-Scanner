import cv2 as cv #type: ignore
import numpy as np

def preprocess_image(image):
    image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)
    output = image.copy()

    hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)

    dark_mask = cv.inRange(hsv, (0, 0, 0), (180, 90, 120))
    dark_mask = cv.morphologyEx(
        dark_mask,
        cv.MORPH_CLOSE,
        np.ones((10, 10), np.uint8),
        iterations=3
    )

    cnts, _ = cv.findContours(dark_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_area = output.shape[0] * output.shape[1]
    print(f"Total contours found: {len(cnts)}")

    for c in cnts:
        area = cv.contourArea(c)
        if area < img_area * 0.01:
            continue

        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int32(box)

        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)
        if not (1.8 < aspect_ratio < 4.5):
            continue

        cv.drawContours(output, [box], 0, (0, 255, 0), 2)

    return output


image = cv.imread("images/extra/good_20.jpg")
if image is None:
    raise FileNotFoundError("Image not found")

processed =preprocess_image(image)

cv.imshow("Detected Cards", processed)
cv.waitKey(0)
cv.destroyAllWindows()
