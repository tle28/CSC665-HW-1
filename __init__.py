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

'''
print(X.head())
rf = RandomForestRegressor()
rf.fit(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.3, True, 12)

print(X_train, X_test, y_train, y_test )
'''
rf = RandomForestRegressor()
rf.fit(X, y)
print(len(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=True,  random_state =17)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
print(X_train,X_test,y_train,y_test)


X_train, X_test, y_train, y_test = features.train_test_splits(X, y, 0.3, True, 17)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train, X_test, y_train, y_test)
'''
mse = metrics.mse(y_hat, y_test)
rmse = metrics.rmse(y_hat, y_test)
rsq = metrics.rsq(y_hat, y_test)

class Object(object): pass 
var = Object()
var.m = RandomForestRegressor(n_estimators=100, oob_score=True)
var.x = X_test
var.y = y_test
print(y_test)
fig = plt.figure(figsize=(10, 6))
plt.plot(var.x, var.y, linewidth=500)
plt.show()
plt.plot(var.y_test, var.m.predict(var.y_test.reshape(-1, 1)), linewidth=500)
plt.show()
'''