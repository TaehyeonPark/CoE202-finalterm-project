from utils import *
from gNet import GestureNet
from cv import CV
import cv2
import itertools
import game


def start(cvproc, gameapi: game.Game = None) -> None:
    cap = cv2.VideoCapture(cvproc.camIdx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cvproc.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cvproc.height)
    state = 0
    gameapi.game_start()
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
        res = cvproc.extractFeatures(frame=frame)
        if res:
            for handLMs, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                bbox = cvproc.LMs2BBox(image_width, image_height, handLMs)
                frame = cvproc.drawBBoxOnFrame(frame, bbox)
                lmList = cvproc.handLM2List(
                    image_width, image_height, handLMs)

                hand = handedness.classification[0].label
                normal = list(itertools.chain.from_iterable(
                    cvproc.stdLMs2Normal(lmList)))
                if hand[0] == cvproc.clabel[0] and cvproc.datasetMode:
                    cvproc.queue.append([cvproc.cindex, normal])

                if len(cvproc.queue) == cvproc.maxQsize:
                    cvproc.handLM2dataset()

                if cvproc.recognitionMode and not cvproc.datasetMode:
                    df = pd.DataFrame(normal).T
                    pred = cvproc.model.predict(df, verbose=0).tolist()
                    print(
                        f"[RESULT] {cvproc.label[pred[0].index(max(pred[0]))]}\t{(max(pred[0]) * 100):.2f}%", end='\r')
                    if 'palm' in cvproc.label[pred[0].index(max(pred[0]))] and state == 0:
                        state = 1
                        gameapi.start_crane()
                    if 'fist' in cvproc.label[pred[0].index(max(pred[0]))] and state == 1:
                        state = 2
                        gameapi.start_push()

                frame = cvproc.drawLandmarkOnFrame(frame, lmList)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(cvproc.windowTitle, frame)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    g = game.Game()
    cvArgs = getCVArgs()
    cv = CV(cvArgs)
    start(cv, g)
    start(cv)
