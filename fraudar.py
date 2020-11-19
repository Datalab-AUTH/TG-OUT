import numpy as np
from math import log10,log,sqrt,pow
import numpy as np


def get_non_zero_row_vals(matrix,index):
    r = matrix[index,:]
    ones = np.where(r == 1)[1]
    return ones.tolist()

def get_num_of_cols(matrix,indexList):
    totalCols=set()
    nonZeroVals = 0
    common = set(list(range(0,matrix.shape[1])))
    for i in indexList:
        setOfCols = set(get_non_zero_row_vals(matrix,i))
        common = set.intersection(setOfCols)
        nonZeroVals = nonZeroVals + len(setOfCols)
        nonzeroCols = setOfCols
        totalCols = totalCols.union(nonzeroCols)
    non_common = totalCols - common
    # print (nonZeroVals,totalCols,common)
    return nonZeroVals, len(totalCols) , common , non_common

def get_first_factor(matrix,indices):
    rows = len(indices)
    nonzeroVals , cols , common ,non_common = get_num_of_cols(matrix,indices)
    ff = nonzeroVals / sqrt( pow(rows,2) + pow(cols,2) )
    return ff, common , non_common, cols

def get_fraudar_score(matrix,indices):
    ff = get_first_factor(matrix,indices)[0]
    return ff

