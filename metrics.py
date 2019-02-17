import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#A. def mse(y_predicted, y_true) - return Mean-Squared Error.
def mse(y_predicted, y_true):
    return ((y_predicted - y_true) ** 2).mean()
#B. def rmse(y_predicted, y_true) - return Root Mean-Squared Error.
def rmse(y_predicted, y_true):
    return np.sqrt(((y_predicted - y_true) ** 2).mean())
#C. def rsq(y_predicted, y_true) - return R^2
def rsq(y_predicted, y_true):
    v = ((y_true - y_true.mean()) ** 2).mean()
    score = 1 - mse(y_predicted, y_true) / v
    return score