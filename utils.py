import pandas as pd
import numpy as np


class Args:
    def __init__(self) -> None:
        self.__args = {}

    def add(self, key, e):
        if key in self.__args.keys():
            print("key already exist")
            return None
        if e == None:
            print("key cannot be None")
            return None
        self.__args[key] = e

    def update(self, key, e):
        self.__args.update({key: e})

    def delete(self, key):
        if not key in self.__args.keys():
            return None
        ret = self.__args[key]
        del self.__args[key]
        return ret

    def get(self, key):
        if not key in self.__args.keys():
            return None
        return self.__args[key]

    def find_key(self, e):
        return list(self.__args.keys())[list(self.__args.values()).index(e)]


def getCVArgs():
    args = Args()
    args.add("camIdx", 0)
    args.add("width", 960)
    args.add("height", 540)
    args.add("allowedPoints", [4, 8, 12, 16, 20])
    args.add("windowTitle", "Preproc")
    args.add("min_tracking_confidence", 0.5)
    args.add("min_detection_confidence", 0.7)
    args.add("use_static_image_mode", False)
    args.add("LandmarkPointColorOuter", (255, 255, 255))
    args.add("LandmarkPointColorInner", (0, 0, 0))
    args.add("LandmarkLineColor", (0, 0, 0))
    args.add("BoundingBoxBorderColor", (255, 255, 255))
    args.add("max_queue_size", 10)
    return args


def getGNetArgs():
    args = Args()
    args.add("inputShape", (21, 2))
    args.add("outputShape", (6))
    args.add("getLabel", getLabel())
    args.add("model_location", "./model/1700693117")
    return args


def initArgs():
    args = Args()
    return args


def getLabel():
    ret = {}
    df = pd.read_csv('label.csv')
    for d in df.values:
        ret[d[0]] = d[1]
    return ret


def getTrain2DF():
    df = pd.read_csv('train.csv')
    X_train = df.loc[:, df.columns != 'mo_class']
    Y_train = df.loc[:, df.columns == 'mo_class']
    return X_train, Y_train


print(getTrain2DF())
