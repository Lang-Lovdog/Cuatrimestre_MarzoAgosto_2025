from LBP_feat_h import LBPFeatures as lbpf
from SDH_feat_h import SDHFeatures as sdhf
import numpy as np

container = lbpf("images.json")
container.defineLBParameters(windowSize=[x for x in np.arange(3,26,2)])
container.getImages()
container.computeLBP()

sdhf.computeSDHFromLBP(container).to_csv(container.csvName, index=False)
