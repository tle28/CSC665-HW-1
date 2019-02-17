#%matplotlib inline
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x):
    return np.power(1.10, x)

class Object(object): pass 
var = Object()
var.x = np.arange(0, 50)
var.y = f(var.x)
# We'll use values below to *test* our model.
var.x_test = np.arange(0.5, 50.5)
var.y_test = f(var.x_test)

from sklearn.ensemble import RandomForestRegressor
var.m = RandomForestRegressor(n_estimators=100, oob_score=True)
_ = var.m.fit(var.x.reshape(-1, 1), var.y)
var.m.oob_score_

var.m.score(var.x_test.reshape(-1, 1), var.y_test)

print(f(1.25), var.m.predict([[1.25]]))
print(f(7.77), var.m.predict([[7.77]]))

f(1.24)

var.m.predict([[1.24]])

print(f(42.45), var.m.predict([[42.45]]))

print(f(49.84), var.m.predict([[49.84]]))


fig = plt.figure(figsize=(10, 6))
plt.plot(var.x, var.y, linewidth=5)

'''
fig = plt.figure(figsize=(10, 6))
plt.plot(var.x_test, var.m.predict(var.x_test.reshape(-1, 1)), lin

fig = plt.figure(figsize=(10, 6))
plt.plot(var.x, var.y, linewidth=5)
plt.plot(var.x_test, var.m.predict(var.x_test.reshape(-1, 1)), lin
'''
