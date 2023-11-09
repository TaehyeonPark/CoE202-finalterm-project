import numpy as np


def parseLM2XYZ(src) -> list:
    return [src.x, src.y, src.z]


def crossProductList(handLandmarks: list) -> list:
    matrix = []
    for handLandmark in handLandmarks:
        vectors = []
        for i in range(len(handLandmark)-1):
            vectors.append(np.cross(parseLM2XYZ(
                handLandmark[i]), parseLM2XYZ(handLandmark[i+1]))  # .tolist()
            )
        matrix.append(vectors)
    return matrix
