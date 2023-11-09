import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe import solutions, tasks
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


class Preproc():
    def __init__(self, camIdx=0) -> None:
        self.camIdx = camIdx
        cap = cv2.VideoCapture(self.camIdx)
        self.fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        self.windowTitle = "Preproc"

        model_path = './model/hand_landmarker.task'
        options = vision.HandLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE
        )
        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )
        self.mp_drawing = solutions.drawing_utils
        self.mp_hands = solutions.hands
        self.gx_kernel = np.array([[1, 0], [0, -1]])
        self.gy_kernel = np.array([[0, 1], [-1, 0]])

    def getCamera(self):
        return cv2.VideoCapture(self.camIdx)

    def extractEdgeFilter(self, frame: cv2.Mat) -> cv2.Mat:
        return cv2.filter2D(frame, -1, self.gx_kernel) + cv2.filter2D(frame, -1, self.gy_kernel)

    def extractLandmark(self, frame: cv2.Mat) -> vision.HandLandmarkerResult:
        return self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))

    def getHandLandmarks(self, detection_result: vision.HandLandmarkerResult):
        return detection_result.hand_landmarks

    def getHandedness(self, detection_result: vision.HandLandmarkerResult):
        return detection_result.handedness

    def drawLandmarkOnFrame(self, frame: cv2.Mat, detection_result: vision.HandLandmarkerResult):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = frame.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            frame = cv2.flip(src=frame, flipCode=1)
            cv2.putText(frame, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            frame = cv2.flip(src=frame, flipCode=1)
        return frame

    def cameraInput(self) -> None:
        cap = cv2.VideoCapture(self.camIdx)
        while cv2.waitKey(self.fps) < 0:
            ret, frame = cap.read()
            if not ret:
                continue
            landmarks = self.extractLandmark(frame=frame)
            frame = self.drawLandmarkOnFrame(
                frame=frame, detection_result=landmarks)
            cv2.imshow(self.windowTitle, cv2.flip(src=frame, flipCode=1))
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = Preproc()
    obj.cameraInput()
