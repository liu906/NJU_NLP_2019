from sklearn import linear_model
import numpy as np
import language_check
import pandas as pd
import random

"""
    Compute evaluation matrix.
"""
def weights_matrix(predicted, actual, score_max):
    actual = np.array(actual)
    weights = []
    for i in range(score_max + 1):
        row = []
        for j in range(score_max + 1):
            row.append((i - j) * (i - j) / ((score_max - 1) * (score_max - 1)))
        weights.append(row)


    return weights

def observed_matrix(predicted, actual,score_max):
    predicted = np.array(predicted)
    actual = np.array(actual)

    weights = np.zeros([score_max + 1, score_max + 1])
    # print("weights: ",weights)

    for pred in predicted:
        for actu in actual:
            weights[pred][actual] += 1
    return weights

def expected_matrix(predicted, actual, score_max):
    observed = observed_matrix(predicted, actual, score_max)
    ob = np.sum(observed, axis=0)
    ex = np.sum(observed, axis=1)
    outer = np.multiply.outer(ob, ex)
    outer = outer / np.sum(observed)
    return outer

def QWK(predicted, actual, score_max):

    WO = weights_matrix(predicted, actual, score_max) * observed_matrix(predicted, actual, score_max)
    WE = weights_matrix(predicted, actual, score_max) * expected_matrix(predicted, actual, score_max)

    # print("=========================================")
    # print(WO)
    # print( np.sum(WO) )
    # print("=========================================")
    # print( np.sum(WE) )
    # print(WE)
    k = 1 - np.sum(WO) / np.sum(WE)

    print("QWK is ", k)
