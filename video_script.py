import cv2
from pypylon import pylon

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()

converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("dataset/conveyor_video.mp4", fourcc, 30, (1280, 720))


while camera.IsGrabbing():
    result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if result.GrabSucceeded():
        img = converter.Convert(result)
        frame = img.GetArray()

        out.write(frame)
        cv2.imshow("Recording…", frame)

        if cv2.waitKey(1) == ord('q'):
            break

out.release()
camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
