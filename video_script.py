import cv2 #type: ignore
import os
from pypylon import pylon #type: ignore

os.makedirs("dataset", exist_ok=True)

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Grab ONE frame first to get size
result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
img = converter.Convert(result)
frame = img.GetArray()
h, w = frame.shape[:2]
result.Release()

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(
    "dataset/conveyor_video.avi",
    fourcc,
    30,
    (w, h)
)

assert out.isOpened(), "VideoWriter failed"

while camera.IsGrabbing():
    result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if result.GrabSucceeded():
        img = converter.Convert(result)
        frame = img.GetArray()

        out.write(frame)
        cv2.imshow("Recording…", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    result.Release()

out.release()
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
