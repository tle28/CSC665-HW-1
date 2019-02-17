import numpy as np

#A. def mse(y_predicted, y_true) - return Mean-Squared Error.
def mse(y_predicted, y_true):
    return ((y_predicted - y_true) ** 2).mean()
#B. def rmse(y_predicted, y_true) - return Root Mean-Squared Error.
def rmse(y_predicted, y_true):
    return np.sqrt(mse(y_predicted, y_true))
#C. def rsq(y_predicted, y_true) - return R^2
def r2_score(y_predicted, y_true):
    v = ((y_true - y_true.mean()) ** 2).mean()
    print(y_predicted.shape)
    print(y_true.shape)
    print(v)
    print(mse(y_predicted, y_true))
    score = 1 - mse(y_predicted, y_true)/v
    return score
