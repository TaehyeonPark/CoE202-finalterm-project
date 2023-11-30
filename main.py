from utils import *
from gNet import GestureNet
from cv import CV
import cv2
import itertools
import game


def start(proc, api) -> None:
    cap = cv2.VideoCapture(proc.camIdx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, proc.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, proc.height)
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
        res = proc.extractFeatures(frame=frame)
        if res:
            for handLMs, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                bbox = proc.LMs2BBox(image_width, image_height, handLMs)
                frame = proc.drawBBoxOnFrame(frame, bbox)
                lmList = proc.handLM2List(
                    image_width, image_height, handLMs)

                hand = handedness.classification[0].label
                normal = list(itertools.chain.from_iterable(
                    proc.absLMs2Relative(lmList)))
                if hand[0] == proc.clabel[0] and proc.datasetMode:
                    proc.queue.append([proc.cindex, normal])

                if len(proc.queue) == proc.maxQsize:
                    proc.handLM2dataset()

                if proc.recognitionMode and not proc.datasetMode:
                    df = pd.DataFrame(normal).T
                    pred = proc.model.predict(df, verbose=0).tolist()
                    print(
                        f"{pred[0].index(max(pred[0]))} - {max(pred[0])}")
                    print(proc.label[pred[0].index(max(pred[0]))])

                frame = proc.drawLandmarkOnFrame(frame, lmList)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(proc.windowTitle, frame)
    cap.release()
    cv2.destroyAllWindows()


# if __name__ == "__main__":


g = game.Game()
cvArgs = getCVArgs()
cv = CV(cvArgs, g)
start(cv)
