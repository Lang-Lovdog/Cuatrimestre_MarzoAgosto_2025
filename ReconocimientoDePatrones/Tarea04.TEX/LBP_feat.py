from LBP_feat_h import LBPFeatures as lbpf
import numpy as np

container = lbpf("images.json")
container.defineLBParameters(windowSize=[x for x in np.arange(3,26,2)])
container.getImages()
container.computeLBP()
container.computeFeaturesFromLBP()
container.saveFileCSV()
