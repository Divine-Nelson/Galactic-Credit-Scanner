# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – MAIN ENTRY FILE
Works with the final patched detection.py
"""

import argparse
import os
import glob
import cv2 #type: ignore
from features import analyze_image, REF_CARD

TOTAL_SUM = 0
REF_CARD["w"] = None
REF_CARD["h"] = None
REF_CARD["ratio"] = None
# ================================================================
#  IMAGE PATH HANDLING
# ================================================================

def list_image_paths(path):
    """Return list of image file paths from a file or directory."""
    if os.path.isdir(path):
        files = []
        for f in os.listdir(path):
            full = os.path.join(path, f)
            if os.path.isfile(full):
                ext = os.path.splitext(f)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                    files.append(full)
        return sorted(files)
    return [path]



def ensure_dir(path):
    if path is None:
        return None
    os.makedirs(path, exist_ok=True)
    return path


# ================================================================
#  PROCESS IMAGES
# ================================================================

def process_images(path, display, save_dir):
    global TOTAL_SUM

    paths = list_image_paths(path)

    if not paths:
        print("No images found.")
        return

    for p in paths:
        print(f"\nProcessing: {p}")
        img = cv2.imread(p)

        if img is None:
            print("Could not read image.")
            continue

        result = analyze_image(img)
        overlay = result["overlay"]

        cards = result["cards"]
        if cards:
            card = cards[0]

            print("Digits:", card["digits"])
            print("Color:", card["color"])
            print("Value:", card["value"])
            print("Fake:", card["fake"])

            # 💰 Money-counter logic
            if not card["fake"]:
                TOTAL_SUM += card["value"]

            print("Running total:", TOTAL_SUM)

        else:
            print("No card detected.")

        # Display window
        if display:
            cv2.imshow("Result", overlay)

            if result.get("dbg_card") is not None:
                cv2.imshow("Detected Card Contour", result["dbg_card"])

            cv2.waitKey(0)

        # Save if requested
        if save_dir:
            name = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(save_dir, f"{name}_overlay.png")
            cv2.imwrite(out_path, overlay)

    if display:
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# ================================================================
#  PROCESS VIDEO OR WEBCAM
# ================================================================

def process_video(source, display, save_dir):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Cannot open video/webcam.")
        return

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = analyze_image(frame)
        overlay = result["overlay"]

        if display:
            cv2.imshow("Result", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break

        if save_dir:
            save_path = os.path.join(save_dir, f"frame_{idx}.png")
            cv2.imwrite(save_path, overlay)

        idx += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()


# ================================================================
#  ARGUMENT PARSER
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Image file or folder")
    parser.add_argument("--video", type=str, help="Video file")
    parser.add_argument("--webcam", type=int, help="Webcam index")
    parser.add_argument("--display", action="store_true", help="Show overlay window")
    parser.add_argument("--save", type=str, help="Directory to save output images")
    return parser.parse_args()


# ================================================================
#  MAIN FUNCTION
# ================================================================

def main():
    args = parse_args()
    save_dir = ensure_dir(args.save)

    # Priority: images → video → webcam
    if args.images:
        process_images(args.images, args.display, save_dir)
        return

    if args.video:
        process_video(args.video, args.display, save_dir)
        return

    if args.webcam is not None:
        process_video(args.webcam, args.display, save_dir)
        return

    print("No input provided. Use --images, --video, or --webcam.")


if __name__ == "__main__":
    main()

"""
python main.py --images images/good/good_1.jpg  --display
python main.py --images images/sedate_images/  --display
python main.py --images images/fake/  --display
python main.py --images images/zahras_images/  --display
python main.py --video videos/test_video.mp4  --display
python main.py --webcam 0  --display
"""