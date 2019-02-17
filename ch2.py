import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

csv_df = pd.read_csv("Melbourne_housing_FULL.csv")

rf = RandomForestRegressor()
print(csv_df.head())
feat_df = csv_df.drop('Price', axis=1)
print(csv_df.shape)
print(feat_df.shape)
y = csv_df['Price'].values
print(y)
print(y.shape)
rows_labeled_na = csv_df.isnull().any(axis=1)
rows_with_na = csv_df[rows_labeled_na]
rows_with_data = csv_df[~rows_labeled_na]
print(csv_df.shape, rows_with_na.shape, rows_with_data.shape)
feat_df = rows_with_data.drop('Price', axis=1)
print(feat_df.shape)
y = rows_with_data['Price'].values
print(y.shape)
suburbs = {}
for s in feat_df['Suburb'].values:
    if s not in suburbs: suburbs[s] = len(suburbs)
print(len(suburbs))
feat_df['Suburb'] = feat_df['Suburb'].replace(suburbs)
print(feat_df.head())
feat_df['Type'] = feat_df['Type'].astype('category').cat.codes
feat_df['Address'] = feat_df['Address'].astype('category').cat.codes
feat_df['Method'] = feat_df['Method'].astype('category').cat.codes
feat_df['SellerG'] = feat_df['SellerG'].astype('category').cat.codes
feat_df['CouncilArea'] = feat_df['CouncilArea'].astype('category').cat.codes
feat_df['Regionname'] = feat_df['Regionname'].astype('category').cat.codes
print(feat_df.head())
feat_df['Date'] = pd.to_datetime(feat_df['Date'], infer_datetime_format=True)
feat_df['Date'] = feat_df['Date'].astype(np.int64)
print(feat_df.head())
rf = RandomForestRegressor()
rf.fit(feat_df, y)

from sklearn.model_selection import train_test_split
#    ? train_test_split
X_train, X_test, y_train, y_test = train_test_split(feat_df, y, random_state =17)
print(feat_df.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
rf = RandomForestRegressor(n_estimators=100 , random_state=17)
#%time rf.fit(X_train, y_train)

print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))
print(rf.predict(X_test))

rf = RandomForestRegressor(random_state=17)
rf.fit(X_train, y_train)

print(rf.estimators_[0].predict(X_test))

print(rf.estimators_[1].predict(X_test))

print(rf.estimators_[2].predict(X_test))

print(rf.predict(X_test))

y_hat = rf.predict(X_test)

print(y_hat)

print(y_test)

print(y_hat.shape, y_test.shape)

mse = ((y_hat - y_test) ** 2).mean()
print(mse)

rmse = np.sqrt(mse)
print(mse, rmse)

v = ((y_test - y_test.mean()) ** 2).mean()
print(v)

score = 1 - mse / v
print(score)