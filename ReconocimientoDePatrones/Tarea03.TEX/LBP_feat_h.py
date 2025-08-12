import os
import json
import numpy as np
import pandas as pd
import cv2 as cv
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

class LBPFeatures:
    features = [
        "File",
        "mean",
        "variance",
        "correlation",
        "contrast",
        "homogeneity",
        "class"
    ]
    def __init__(self,josnFile):
        self.LBP={}
        self.fileNames = {}
        self.windowSize = [3]
        self.ratio = 1
        self.points = 8
        self.mathod = "uniform"
        self.filePattern = ".png"
        self.featuresTable = pd.DataFrame(columns=self.features)
        with open(josnFile) as josnFile:
            self.jsonData = json.load(josnFile)
        self.csvName=self.jsonData["csv"]
        self.samples = self.jsonData["samples"]
        self.images={}

    def setFilePattern(self, pattern):
        self.filePattern = pattern

    def getImages(self):
        if len(self.images) > 0:
            self.images = {}

        for sample in self.samples:
            self.images[sample["class"]] = [
                os.path.join(sample["path"], file)
                for file in os.listdir(sample["path"])
                if file.endswith(self.filePattern)
            ]

    def defineLBParameters(self, ratio=None, points=None, mathod=None, windowSize=None):
        if ratio is not None:
            self.ratio = ratio
        if points is not None:
            self.points = points
        if mathod is not None:
            self.mathod = mathod
        if windowSize is not None:
            if type(windowSize) is not list:
                windowSize = [windowSize]

    def computeLBP(self):
        for key in self.images:
            self.LBP[key] = []
            for file in self.images[key]:
                img = cv.imread(file, cv.IMREAD_GRAYSCALE)
                lbp = local_binary_pattern(img, self.points, self.ratio, self.mathod).astype(np.uint8)
                self.LBP[key].append({"File": os.path.basename(file), "LBP": lbp})

    @staticmethod
    def appendFeaturesRow(row, dataframe):
        if type(row) is not pd.DataFrame:
            dataframe = pd.DataFrame(row)
        return pd.concat([dataframe, row], ignore_index=True, axis=0)

    @staticmethod
    def getFeaturesFromWindow(window):
        features = {}
        glcm = graycomatrix(window, [1], [0], symmetric=True, normed=True, levels=59)
        features["mean"] = graycoprops(glcm, 'mean')[0][0]
        features["variance"] = graycoprops(glcm, 'variance')[0][0]
        features["correlation"] = graycoprops(glcm, 'correlation')[0][0]
        features["contrast"] = graycoprops(glcm, 'contrast')[0][0]
        features["homogeneity"] = graycoprops(glcm, 'homogeneity')[0][0]
        return features

    @staticmethod
    def lbpWindowAnalysis(lbpimg, windowSize, _class=None):
        x=0
        y=0
        lbpImage = lbpimg["LBP"]
        lbpFile  = lbpimg["File"]
        analysis = []
        for x in range(lbpImage.shape[0]-windowSize):
            for y in range(lbpImage.shape[1]-windowSize):
                window = lbpImage[x:x+windowSize, y:y+windowSize]
                lbpfeat=LBPFeatures.getFeaturesFromWindow(window)
                lbpfeat["File"] = lbpFile
                if _class is not None:
                    lbpfeat["class"] = _class
                analysis.append(lbpfeat)
        return pd.DataFrame(analysis, columns=LBPFeatures.features)
            
    def computeFeaturesFromLBP(self):
        for c in self.LBP.keys():
            for i in self.LBP[c]:
                for ws in self.windowSize:
                    result=self.lbpWindowAnalysis(i, ws, _class=c)
                    self.featuresTable = LBPFeatures.appendFeaturesRow(result, self.featuresTable)
        print(self.featuresTable)

    def saveFileCSV(self):
        self.featuresTable.to_csv(self.csvName, index=False)
