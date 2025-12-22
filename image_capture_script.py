from pypylon import pylon
import cv2

# Initialize camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        img = converter.Convert(grabResult)
        frame = img.GetArray()

        cv2.imshow("Basler Feed", frame)

        # Press 's' to save a frame, 'q' to quit
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite("dataset/frame.jpg", frame)
        elif key == ord('q'):
            break

grabResult.Release()
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
