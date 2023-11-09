import cv2
from cv import Preproc
from utils import *

if __name__ != "__main__":
    print("This is not for import")

obj = Preproc(0)
cap = obj.getCamera()

while cv2.waitKey(obj.fps) < 0:
    ret, frame = cap.read()
    if not ret:
        continue
    landmarks = obj.extractLandmark(frame=frame)
    handLandmarks = obj.getHandLandmarks(landmarks)
    # for handLandmark in handLandmarks:
    #     for i in range(len(handLandmark)-1):
    #         print(parseLM2XYZ(handLandmark[i]), parseLM2XYZ(handLandmark[i+1]))
    if len(handLandmarks) > 0:
        print(crossProductList(handLandmarks=handLandmarks))
    frame = obj.drawLandmarkOnFrame(
        frame=frame, detection_result=landmarks)
    cv2.imshow(obj.windowTitle, cv2.flip(src=frame, flipCode=1))

cap.release()
cv2.destroyAllWindows()
