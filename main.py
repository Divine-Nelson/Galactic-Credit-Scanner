# -*- coding: utf-8 -*-
"""
Star Wars Credit Detection – MAIN ENTRY FILE
Compatible with FINAL multi-card features.py
"""

import argparse
import os
import cv2 #type: ignore
from analyzer import analyze_image

# Optional Basler support
try:
    from pypylon import pylon #type: ignore
    BASLER_AVAILABLE = True
except ImportError:
    BASLER_AVAILABLE = False


# ================================================================
# IMAGE PATH HANDLING
# ================================================================

def list_image_paths(path):
    if os.path.isdir(path):
        return sorted([
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.splitext(f)[1].lower()
            in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
        ])
    return [path]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def preprocess_image(image, max_dim=1500):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    return image


# ================================================================
# IMAGE PROCESSING
# ================================================================

def process_images(path, display=False, save_dir=None):
    run_total = 0
    paths = list_image_paths(path)

    if not paths:
        print("No images found.")
        return

    for p in paths:
        print(f"\nProcessing: {p}")
        img = cv2.imread(p)
        if img is None:
            print(f"❌ Could not read image: {p}")
            continue

        img = preprocess_image(img)
        result = analyze_image(img, video_mode=False)


        overlay = result["overlay"]
        cards = result["cards"]
        image_total = result["total_value"]

        print(f"Detected {len(cards)} cards.")
        print(f"Image total value: {image_total}")

        for i, card in enumerate(cards, 1):
            print(f"  Card {i}:")
            print(f"    Digits: {card['digits']}")
            print(f"    Color : {card['color']}")
            print(f"    Value : {card['value']}")
            print(f"    Fake  : {card['fake']}")

        run_total += image_total
        print(f"Running total: {run_total}")

        if display:
            cv2.imshow("Result", overlay)
            cv2.waitKey(0)

        if save_dir:
            name = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(
                os.path.join(save_dir, f"{name}_overlay.png"),
                overlay
            )

    if display:
        cv2.destroyAllWindows()

    print(f"\n✅ FINAL TOTAL VALUE: {run_total}")


# ================================================================
# VIDEO / WEBCAM
# ================================================================

def process_video(source, display=False, save_dir=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Cannot open video/webcam.")
        return

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = analyze_image(frame, video_mode=True)

        overlay = result["overlay"]

        if display:
            cv2.imshow("Result", overlay)
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

        if save_dir:
            cv2.imwrite(
                os.path.join(save_dir, f"frame_{idx}.png"),
                overlay
            )
        idx += 1

    cap.release()
    if display:
        cv2.destroyAllWindows()


# ================================================================
# BASLER REAL-TIME
# ================================================================

def process_basler_realtime(display=False, save_dir=None):
    if not BASLER_AVAILABLE:
        print("❌ Basler SDK not available.")
        return

    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice()
    )
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    idx = 0
    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab.GrabSucceeded():
            frame = converter.Convert(grab).GetArray()
            result_data = analyze_image(frame, video_mode=True)
            overlay = result_data["overlay"]

            if display:
                cv2.imshow("Basler Result", overlay)
                if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                    break

            if save_dir:
                cv2.imwrite(
                    os.path.join(save_dir, f"basler_{idx}.png"),
                    overlay
                )
            idx += 1
        grab.Release()

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()


# ================================================================
# ARGUMENT PARSER
# ================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, help="Image file or folder")
    parser.add_argument("--video", type=str, help="Video file")
    parser.add_argument("--webcam", type=int, help="Webcam index")
    parser.add_argument("--basler", action="store_true", help="Use Basler camera")
    parser.add_argument("--display", action="store_true", help="Show overlay")
    parser.add_argument("--save", type=str, help="Save output directory")
    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():
    args = parse_args()
    save_dir = ensure_dir(args.save)

    if args.images:
        process_images(args.images, args.display, save_dir)
    elif args.video:
        process_video(args.video, args.display, save_dir)
    elif args.webcam is not None:
        process_video(args.webcam, args.display, save_dir)
    elif args.basler:
        process_basler_realtime(args.display, save_dir)
    else:
        print("No input provided.")


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