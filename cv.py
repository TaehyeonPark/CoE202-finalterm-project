import cv2
import mediapipe as mp
import numpy as np
import csv
from pathlib import Path

import itertools
from collections import deque

import gNet
from utils import *


class CV():
    def __init__(self, args) -> None:
        self.camIdx = args.get("camIdx")
        self.width = args.get("width")
        self.height = args.get("height")
        self.allowedPoints = args.get("allowedPoints")
        self.windowTitle = args.get("windowTitle")
        self.LMPtCIn = args.get("LandmarkPointColorInner")
        self.LMPtCOut = args.get("LandmarkPointColorOuter")
        self.LMLineC = args.get("LandmarkLineColor")
        self.bboxC = args.get("BoundingBoxBorderColor")
        self.min_tracking_confidence = args.get("min_tracking_confidence")
        self.min_detection_confidence = args.get(
            "min_detection_confidence")
        self.use_static_image_mode = args.get("use_static_image_mode")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.maxQsize = args.get("max_queue_size")

        self.datasetMode = False
        self.cindex = 0
        self.clabel = getLabel()[self.cindex]
        print(self.cindex, self.clabel)
        self.queue = []

        self.recognitionMode = True
        gNetArgs = getGNetArgs()
        self.model = None
        if self.recognitionMode:
            gnet = gNet.GestureNet(args=args)
            self.model = gnet.model()
            self.model.summary()
            self.model.load_weights(
                "./model/primary/model_output")
            self.label = getLabel()

    def handLM2dataset(self):
        if not self.datasetMode:
            return None
        print("Create dataset")
        with open("train.csv", 'a', newline='') as f:
            w = csv.writer(f)
            tmp = []
            while (len(self.queue)):
                hand, landmark = self.queue.pop()
                landmark.insert(0, hand)
                tmp.append(landmark)
            w.writerows(tmp)

    def extractEdgeFilter(self, frame: cv2.Mat) -> cv2.Mat:
        return cv2.filter2D(frame, -1, self.gx_kernel) + cv2.filter2D(frame, -1, self.gy_kernel)

    def extractFeatures(self, frame: cv2.Mat) -> cv2.Mat:
        result = self.hands.process(frame)
        return result if result.multi_hand_landmarks is not None else None

    def handLM2List(self, img_w: int, img_h: int, landmarks) -> list[list[int, int]]:
        return [[min(img_w-1, int(lm.x*img_w)), min(img_h-1, int(lm.y * img_h))] for lm in landmarks.landmark]

    def absLMs2Relative(self, landmarkList: list) -> list:
        (std_x, std_y) = landmarkList[0]
        m = max(list(itertools.chain.from_iterable(landmarkList)))
        lms = [[(lm_x-std_x) / m, (lm_y-std_y) / m]
               for (lm_x, lm_y) in landmarkList]
        return lms

    def drawLandmarkOnFrame(self, frame: cv2.Mat, lm) -> cv2.Mat:
        if len(lm) > 0:
            cv2.line(frame, tuple(lm[2]), tuple(lm[3]), self.LMLineC, 3)
            cv2.line(frame, tuple(lm[3]), tuple(lm[4]), self.LMLineC, 3)
            for i in range(5, 19, 4):
                cv2.line(frame, tuple(lm[i]), tuple(lm[i+1]), self.LMLineC, 3)
                cv2.line(frame, tuple(lm[i+1]),
                         tuple(lm[i+2]), self.LMLineC, 3)
                cv2.line(frame, tuple(lm[i+2]),
                         tuple(lm[i+3]), self.LMLineC, 3)
        for idx, lmk in enumerate(lm):
            if idx in self.allowedPoints:
                cv2.circle(frame, (lmk[0], lmk[1]), 8, self.LMPtCIn,
                           -1)
                cv2.circle(
                    frame, (lmk[0], lmk[1]), 8, self.LMPtCOut, 1)
        return frame

    def LMs2BBox(self, img_w: int, img_h: int, lms) -> list[int, int, int, int]:
        x_min, y_min, x_max, y_max = img_w-1, img_h-1, 0, 0
        for lm in lms.landmark:
            x_min = min(x_min, int(lm.x*img_w))
            y_min = min(y_min, int(lm.y*img_h))
            x_max = max(x_max, int(lm.x*img_w))
            y_max = max(y_max, int(lm.y*img_h))
        return [x_min, y_min, x_max, y_max]

    def drawBBoxOnFrame(self, frame: cv2.Mat, bbox: list) -> cv2.Mat:
        cv2.rectangle(frame, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), self.bboxC, 4)
        return frame

    def cameraInput(self) -> None:
        cap = cv2.VideoCapture(self.camIdx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        while True:
            key = cv2.waitKey(10)
            if key == 27:
                break
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_width, image_height = frame.shape[1], frame.shape[0]
            res = self.extractFeatures(frame=frame)
            if res:
                for handLMs, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    bbox = self.LMs2BBox(image_width, image_height, handLMs)
                    frame = self.drawBBoxOnFrame(frame, bbox)
                    lmList = self.handLM2List(
                        image_width, image_height, handLMs)

                    hand = handedness.classification[0].label
                    normal = list(itertools.chain.from_iterable(
                        self.absLMs2Relative(lmList)))
                    if hand[0] == self.clabel[0] and self.datasetMode:
                        self.queue.append([self.cindex, normal])

                    if len(self.queue) == self.maxQsize:
                        self.handLM2dataset()

                    if self.recognitionMode and not self.datasetMode:
                        df = pd.DataFrame(normal).T
                        pred = self.model.predict(df, verbose=0).tolist()
                        print(
                            f"{pred[0].index(max(pred[0]))} - {max(pred[0])}")
                        print(self.label[pred[0].index(max(pred[0]))])

                    frame = self.drawLandmarkOnFrame(frame, lmList)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.windowTitle, frame)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = getCVArgs()
    obj = CV(args=args)
    obj.cameraInput()
