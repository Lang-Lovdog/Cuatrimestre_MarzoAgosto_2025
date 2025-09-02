import cv2 as cv
import pandas as pd
import numpy as np
import radiomics.glrlm as glrlm 
import SimpleITK as sitk
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image, distance, angle, props):
    # Assure props, distance and angle are lists
    if type(distance) is not list:
        distance = [distance]

    if type(angle) is not list:
        angle = [angle]

    if type(props) is not list:
        props = [props]


    # Get GLCM
    glcm = graycomatrix(image, distance, angle, levels=256)

    # For each GLCM, get features as a new pandas Row
    features = {}
    for p in props:
        features[p] = np.reshape(graycoprops(glcm, p), -1)

    # Append features to a pandas DataFrame
    salida = pd.DataFrame.from_dict(features)

    return salida

def extract_glrl_features(image, angle, props):
    # Assure props, distance and angle are lists
    if type(angle) is not list:
        angle = [angle]

    if type(props) is not list:
        props = [props]

    # Intialize GLRLM
    # Set GLRLM parameters
    settings = {
        'distances': [1],                 # Pixel distances
        'angles': [0, np.pi/4, np.pi/2]   # Directions (0°, 45°)
    }

    # Convert to radiomics compatible array
    # Convert to SimpleITK format
    img = sitk.GetImageFromArray(image.astype(np.float32))
    print(img)

    # Get GLRL
    glrl = glrlm.RadiomicsGLRLM(img, img).execute()

    # For each GLRL, get features as a new pandas Row
    #features = {}
    #for p in props:
    #    features[p] = np.reshape(glrl, -1)

    # Append features to a pandas DataFrame
    #features = pd.DataFrame.from_dict(features)
    #return features
