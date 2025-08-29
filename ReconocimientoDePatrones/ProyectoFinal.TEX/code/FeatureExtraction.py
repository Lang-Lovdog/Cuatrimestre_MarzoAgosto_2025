import sys
import pandas as pd
import cv2 as cv
import numpy as np

from LovdogFeatureExtraction import extract_glcm_features
from LovdogFeatureExtraction import extract_glrl_features

def compute_idf(glcm):
    glcm_normalized = glcm / np.sum(glcm)  # Ensure normalized
    idf = 0.0
    rows, cols = glcm.shape[0:2]
    
    for i in range(rows):
        for j in range(cols):
            idf += glcm_normalized[i,j,0,0] / (1 + abs(i-j))
    
    return idf


def MAINGLCM(image,props):
    flag = True
    distancias = [1,3,7]
    angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    df = pd.DataFrame(columns=props)
    for img in image:
        src = cv.imread(img, cv.IMREAD_GRAYSCALE)
        glcmFeatures = extract_glcm_features(src, distancias, angulos, props)
        if flag:
            df = pd.concat([df, glcmFeatures], ignore_index=True)
            flag = False
        else:
            df = glcmFeatures
    return glcmFeatures

def MAINGLRL(image,props):
    flag = True
    df = pd.DataFrame(columns=props)
    angulos = [0, np.pi/4, np.pi/2]
    for img in image:
        src = cv.imread(img, cv.IMREAD_GRAYSCALE)
        glrlFeatures = extract_glrl_features(src, angulos, props)
        if flag:
            df = pd.concat([df, glrlFeatures], ignore_index=True)
            flag = False
        else:
            df = glrlFeatures
    return glrlFeatures

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python FeatureExtraction.py <image | image list> ")
        sys.exit(1)

    featuresGLCM = [
        "variance",
        "entropy",
        "energy",
        "correlation",
        "contrast",
        "homogeneity"
    ]
    featuresGLRL = [
        "SRE",
        "LRE",
        "GLN",
        "RLN",
        "RP"
    ]

    glcmFeatureMatrix = pd.DataFrame(columns=featuresGLCM)
    imagenes = sys.argv[1:]
    print(MAINGLCM(imagenes, featuresGLCM))
    print(MAINGLRL(imagenes, featuresGLRL))
