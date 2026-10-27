import numpy as np
from Evaluation_All import seg_evaluation
from Glob_Vars import Glob_Vars
from Model_ADLabV3p import Model_ADLabV3p


def Objfun_seg(Soln):
    Images = Glob_Vars.Images
    GT = Glob_Vars.GT
    Target = Glob_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)

    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            pred = Model_ADLabV3p(Images, GT, sol)
            Eval = seg_evaluation(pred, GT)
            Fitn[i] = 1 / (Eval[1])  # IOU
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        pred = Model_ADLabV3p(Images, GT, sol)
        Eval = seg_evaluation(pred, GT)
        Fitn = 1 / (Eval[1])  # IOU
        return Fitn

