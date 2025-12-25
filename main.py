# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – MAIN ENTRY FILE
Works with the final patched detection.py
"""

import argparse
import os
import glob
import cv2 #type: ignore
from features import analyze_image


# ================================================================
#  IMAGE PATH HANDLING
# ================================================================

def list_image_paths(path):
    """Return list of images from a path or directory."""
    if os.path.isdir(path):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
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

        # Extract info
        cards = result["cards"]
        if cards:
            print("Detected value:", cards[0]["value"])
            print("Digits:", cards[0]["digits"])
            print("Color:", cards[0]["color"])
        else:
            print("No card detected.")

        # Display window
        if display:
            cv2.imshow("Result", overlay)
            cv2.waitKey(1)

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
