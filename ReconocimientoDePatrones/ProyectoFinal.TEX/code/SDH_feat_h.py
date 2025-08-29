import numpy as np
import pandas as pd

class SDHFeatures:
    _features = [
        "File"       ,
        "mean"       ,
        "variance"   ,
        "correlation",
        "contrast"   ,
        "homogeneity",
        "shadowness" ,
        "prominence" ,
        "class"
    ]
    ANGLE={
        0:0,
       45:1,
       90:2,
      135:3,
    }
    SUM=0
    DIFF=1
    def __init__(self, d=1, angle=0, src=None):
        self.d        = d
        self.angle    = SDHFeatures.ANGLE[angle]
        self.dx       =  d*np.cos(np.radians(angle))
        self.dy       = -d*np.sin(np.radians(angle))
        self.sumHist  = []
        self.diffHist = []
        features      = {}
        if src is not None:
            self.getSDH(src)

    def getSDH(self, src):
        if (self.angle) == SDHFeatures.ANGLE[0]:
            operand01 = src[      0:src.shape[1]         ,        0:src.shape[0] - self.d ]
            operand02 = src[      0:src.shape[1]         ,   self.d:src.shape[0]          ]
        elif (self.angle) ==                                 SDHFeatures.ANGLE[45]:      
            operand01 = src[ self.d:src.shape[1]         ,        0:src.shape[0] - self.d ]
            operand02 = src[      0:src.shape[1] - self.d,   self.d:src.shape[0]          ]
        elif (self.angle) ==                                 SDHFeatures.ANGLE[90]:      
            operand01 = src[ self.d:src.shape[1]         ,        0:src.shape[0] - self.d ]
            operand02 = src[      0:src.shape[1]         ,        0:src.shape[0]          ]
        elif (self.angle) ==                                 SDHFeatures.ANGLE[135]:     
            operand01 = src[ self.d:src.shape[1]         ,   self.d:src.shape[0] - self.d ]
            operand02 = src[      0:src.shape[1]         ,        0:src.shape[0]          ]
        sum      = operand01 + operand02
        diff     = operand01 - operand02
        self.sumHist  = np.histogram(sum , bins=511, range=(   0, 511))[0]
        self.diffHist = np.histogram(diff, bins=511, range=(-255, 255))[0]
        self.sumHist  = self.sumHist / np.sum(self.sumHist)
        self.diffHist = self.diffHist / np.sum(self.diffHist)
        return self.sumHist, self.diffHist

    def computeFeatures(self):
        bin = {}
        bin[SDHFeatures.SUM]  = np.array([i      for i in range(511)])
        bin[SDHFeatures.DIFF] = np.array([i-255  for i in range(511)])
        # Mean Sum
        mean_sum    = np.sum( bin[SDHFeatures.SUM ]                 * self.sumHist )
        # Mean Diff 
        mean_diff   = np.sum( bin[SDHFeatures.DIFF]                 * self.diffHist)
        # Variance Sum
        var_sum     = np.sum((bin[SDHFeatures.SUM ] - mean_sum )**2 * self.sumHist )
        # Variance Diff
        var_diff    = np.sum((bin[SDHFeatures.DIFF] - mean_diff)**2 * self.diffHist)
        # Contrast
        contrast    = var_diff
        # Correlation
        correlation = (var_sum  - var_diff) / (var_sum  + var_diff + 1e-6)
        # Homogeneity
        homogeneity = np.sum(self.diffHist / (1 + np.abs(bin[SDHFeatures.DIFF])))
        # Shadowness
        shadowness  = np.sum((bin[SDHFeatures.DIFF] - mean_diff)**3 * self.diffHist)
        # Prominence
        prominence  = np.sum((bin[SDHFeatures.DIFF] - mean_diff)**4 * self.diffHist)

        self.features = {
            "mean"        : mean_diff  ,
            "variance"    : var_diff   ,
            "correlation" : correlation,
            "contrast"    : contrast   ,
            "homogeneity" : homogeneity,
            "shadowness"  : shadowness ,
            "prominence"  : prominence
        }

        return self.features

    @staticmethod
    def appendFeaturesRow(row, dataframe):
        if type(row) is not pd.DataFrame:
            row = pd.DataFrame(row, columns=SDHFeatures._features)
        return pd.concat([dataframe, row], ignore_index=True, axis=0)

    @staticmethod
    def _promediateFeatures(sdhList):
        features = {}
        for feature in SDHFeatures._features:
            if feature == "File" or feature == "class":
                continue
            features[feature] = np.mean([ sdh.features[feature] for sdh in sdhList ])
        return features
    
    @staticmethod
    def lbpWindowAnalysis(lbpimg, windowSize, _class=None):
        SDHList = []
        SDHDict = {}
        x=0
        y=0
        lbpImage = lbpimg["LBP"]
        lbpFile  = lbpimg["File"]
        for x in range(lbpImage.shape[0]-windowSize):
            for y in range(lbpImage.shape[1]-windowSize):
                window = lbpImage[x:x+windowSize, y:y+windowSize]
                sdhElement = SDHFeatures(src=window)
                sdhElement.computeFeatures()
                SDHList.append(sdhElement)
        SDHDict = SDHFeatures._promediateFeatures(SDHList)
        SDHDict["File"] = lbpFile
        if _class is not None:
            SDHDict["class"] = _class
        return pd.DataFrame(SDHDict, index=[0], columns=SDHFeatures._features)

    @staticmethod
    def computeSDHFromLBP(lbpObject, customAnalysisFunction=None):
        featuresTable=pd.DataFrame(columns=SDHFeatures._features)
        print(featuresTable)
        for c in lbpObject.LBP.keys():
            for i in lbpObject.LBP[c]:
                for ws in lbpObject.windowSize:
                    result=SDHFeatures.lbpWindowAnalysis(i, ws, _class=c)
                    featuresTable = SDHFeatures.appendFeaturesRow(result, featuresTable)
        return featuresTable
