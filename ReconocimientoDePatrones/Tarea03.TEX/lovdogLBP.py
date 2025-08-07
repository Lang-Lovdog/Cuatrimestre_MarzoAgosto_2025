import cv2 as cv
import numpy as np
import os

LBPKernel = 2**cv.Mat(np.array([
    0, 1, 2,
    7, 0, 3,
    6, 5, 4
])).reshape(3,3)

def lovdogGetImageGrayScale(file):
    if file is None:
        return
    if type(file) is not str:
        return
    if not os.path.isfile(file):
        return
    return cv.imread(file, cv.IMREAD_GRAYSCALE)

def lovdogGetLBP(img):
    if img is None:
        return 0
    i=0
    j=0
    cols = img.shape[1]
    rows = img.shape[0]
    lbp = np.array([])
    for i in range(rows-2):
        for j in range(cols-2):
            ventana = img[i:i+3, j:j+3]
            lbp = np.append(lbp, lovdogWindowLBP(ventana))

    return cv.Mat(lbp).reshape(cols-2,rows-2)

def lovdogWindowLBP(window):
    if window is None:
        return 0

    # Matriz B = Matriz A - Escalar Pc
    # For All P in Matriz B, if P < 0, P=0
    # For All P in Matriz B, if P >= 0, P=1
    window = window - window[1,1]
    return np.sum(np.multiply(
        cv.threshold(window, 0, 255, cv.THRESH_BINARY)[1]//255, LBPKernel
    ))


if __name__ == "__main__":
    img = lovdogGetImageGrayScale("img.png")
    lbp = lovdogGetLBP(img)
    cv.imshow("LBP", lbp)
    cv.waitKey(0)
    cv.destroyAllWindows()
