from ActiveContourFitting_h import ActiveContourFitting as acf
import numpy as np
import matplotlib
import os

def clean():
    # If error ignore
    try:
        dir="Amoeba-moving-Timelapse_frames_ActiveContourFitting_Prueba01-200iteraciones"
        for file in os.listdir(dir):
            os.remove(dir+"/"+file)
        os.rmdir(dir)
    except Exception as e:
        print("Error"+e)
    try:
        os.remove("Amoeba-moving-Timelapse_frames_ActiveContourFitting_Prueba01-200iteraciones/*")
    except Exception as e:
        print("Error"+e)

def calculateAC(a, b, g, nl):
    activeContour = acf(
        alpha=0.015,
        beta= b,
        gamma=0.001,
        gSigma=1,
        w_line=3,
        snakeIterations=200,
        firstFrame=0,
        lastFrame =1
    ) 

    activeContour.getVideo()
    activeContour.setInit(n_sides=nl, x=210, y=260, height=100, width=100)
    activeContour.moreContrast()
    activeContour.cropCoef(xCoef=12, yCoef=12)
    activeContour.activeContourFitting()
    #activeContour.showInitFrame()
    #activeContour.showActiveContourFitting()
    activeContour.saveActiveContourFitting(f"Results/Norm/{nl}NL/Prueba_{a:02.3f}a-{b:02.3f}b-{g:02.3f}g-200it")
    #aContour.saveVideo(f"Prueba{index:02d}-200iteraciones")

def calculateACInv(a, b, g, nl):
    activeContour = acf(
        alpha=0.015,
        beta= b,
        gamma=0.001,
        gSigma=1,
        w_line=-10,
        snakeIterations=200,
        firstFrame=0,
        lastFrame =1
    ) 

    activeContour.getVideo()
    activeContour.setInit(n_sides=nl, x=210, y=260, height=100, width=100)
    activeContour.moreContrast()
    activeContour.cropCoef(xCoef=12, yCoef=12)
    activeContour.invertLevels()
    activeContour.activeContourFitting()
    #activeContour.showInitFrame()
    activeContour.saveActiveContourFitting(f"Results/Inv/Prueba_{a:02.3f}a-{b:02.3f}b-{g:02.3f}g-200it-Inv")
    #aContour.saveVideo(f"Prueba{index:02d}-200iteraciones")

def pruebas():
    for g in np.linspace(0, 0.01, 4):
        for a in np.linspace(0, 0.02, 4):
            for b in np.linspace(0, 1, 5):
                for nl in np.arange(10, 200, 15):
                    calculateAC(a, b, g, nl)
                matplotlib.pyplot.close()


def fullAC(a, b, g, nl, s):
    activeContour = acf(
        alpha=a,
        beta= b,
        gamma=g,
        gSigma=s,
        w_line=1,
        snakeIterations=10,
    ) 

    activeContour.getVideo()
    activeContour.setInit(n_sides=nl, x=210, y=260, height=100, width=100)
    activeContour.moreContrast()
    activeContour.thresholdingFrames()
    activeContour.cropCoef(xCoef=12, yCoef=12)
    activeContour.activeContourFitting()
    #activeContour.showInitFrame()
    #activeContour.showActiveContourFitting()
    activeContour.saveActiveContourFitting(f"Final/{nl}NL/Prueba_tres_{a:02.3f}a-{b:02.3f}b-{g:02.3f}g-{s:04d}s-20it")
    activeContour.saveVideo(f"{nl}NL_Prueba_tres_{a:02.3f}a-{b:02.3f}b-{g:02.3f}g-{s:04d}s-20it")

if __name__ == "__main__":
#    pruebas()
    fullAC(0.015, 0.01, 0.012, 20 , 2)
    fullAC(0.015, 0.01, 0.012, 60 , 2)
    fullAC(0.015, 0.01, 0.012, 100, 2)
    fullAC(0.015, 0.01, 0.012, 150, 2)
    fullAC(0.015, 0.01, 0.012, 200, 2)
