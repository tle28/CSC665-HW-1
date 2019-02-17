import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import features
import metrics


file_name = "Melbourne_housing_FULL copy.csv"
X, y = features.preprocess_ver_1(file_name)

rf = RandomForestRegressor()
rf.fit(X, y)

X_train, X_test, y_train, y_test = features.train_test_split(X, y, 0.3, True, 17)

RANDOM_STATE = 10
rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate Impact of the Number of Trees
y_train_pred = rf.predict(X_train)
y_predicted = y_train_pred
y_true = y_train
r2_score_train = metrics.r2_score(y_predicted, y_true)
print("r2_score_train = " + str(r2_score_train))

y_test_pred = rf.predict(X_test)
y_predicted = y_test_pred
y_true = y_test
r2_score_test = metrics.r2_score(y_predicted, y_true)
print("r2_score_test = " + str(r2_score_test))


class Object(object): pass
var = Object()
var.m = RandomForestRegressor(n_estimators=100, oob_score=True)
var.y = y_test
fig = plt.figure(figsize=(10, 6))
plt.plot(var.y, label = "y_test")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Y-test')
plt.legend()
plt.show()
