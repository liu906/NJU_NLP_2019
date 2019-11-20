from sklearn import linear_model
import numpy as np
import pandas as pd
import random

def weights_matrix(predicted,actual):

    actual = np.array(actual)
    weights = []
    for i in range(actual.min(),actual.max()):
        row = []
        for j in range(actual.min(),actual.max()):
            row.append((i - j) * (i - j) / ((actual.max() - 1) * (actual.max() - 1)))
        weights.append(row)


    return weights

def observed_matrix(predicted,actual):
    predicted = np.array(predicted)
    actual = np.array(actual)
    maxv = actual.max() + 1
    weights = np.zeros([maxv ,maxv])
    for pred in predicted:
        for actu in actual:
            weights[pred][actual] += 1
    return weights

def expected_matrix(predicted,actual):
    observed = observed_matrix(predicted, actual)
    ob = np.sum(observed,axis=0)
    ex = np.sum(observed,axis=1)
    outer = np.multiply.outer(ob, ex)
    outer = outer / np.sum(observed)
    return outer

def QWK(predicted,actual):
    k = 1 - sum(weights_matrix(predicted, actual)*observed_matrix(predicted,
                                actual)) / sum(weights_matrix(predicted, actual) * expected_matrix(predicted, actual))

