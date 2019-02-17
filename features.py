import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

'''
A. def train_test_split(X, y, test_size, shuffle, random_state=None) :
    - X, y - features and the target variable.
    - test_size - between 0 and 1 - how much to allocate to the test set; the rest goes to the train set. 
    - shuffle - if True, shuffle the dataset, otherwise not.
    - random_state, integer; if None, then results are random, otherwise fixed to a given seed. 
    - Example:
        X_train, X_test, y_train, y_test = train_test_split(feat_df, y, 0.3, True, 12)
'''
def train_test_splits(X, y, test_size, shuffle, random_states):
    rf = RandomForestRegressor()
    rf.fit(X, y)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, shuffle, random_state = random_states)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =17)
    #print(X.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #print(rf.score(X_train, y_train))
    #print(rf.score(X_test, y_test))
    #print(rf.predict(X_test))
    rf.fit(X_train, y_train)
    #print(rf.estimators_[0].predict(X_test))
    #print(rf.estimators_[1].predict(X_test))
    #print(rf.estimators_[2].predict(X_test))
    #print(rf.predict(X_test))
    y_hat = rf.predict(X_test)
    return y_hat, X_train, X_test, y_train, y_test

'''
B. create_categories(df, list_columns)
    - Converts values, in-place, in the columns passed in the list_columns to numerical values. 
    Follow the same approach: "string" -> category -> code.
    - Replace values in df, in-place.
'''

def create_categories(df, list_columns):
    df[list_columns] = df[list_columns].astype('category').cat.codes
    return df[list_columns]


#X, y = preprocess_ver_1(csv_df)

'''
C. X, y = preprocess_ver_1(csv_df)
Apply the feature transformation steps to the dataframe, return new X and y for entire dataset. 
Do not modify the original csv_df .
    - Remove all rows with NA values
    - Convert datetime to a number
    - Convert all strings to numbers.
    - Split the dataframe into X and y and return these.
'''
def preprocess_ver_1(file_name):
    csv_df = pd.read_csv(file_name)
    feat_df = csv_df.drop('Price', axis=1)
    y = csv_df['Price'].values

    #Remove all rows with NA values
    rows_labeled_na = csv_df.isnull().any(axis=1)
    rows_with_na = csv_df[rows_labeled_na]
    rows_with_data = csv_df[~rows_labeled_na]
    feat_df = rows_with_data.drop('Price', axis=1)

    y = rows_with_data['Price'].values

    #Convert all strings to numbers.
    suburbs = {}
    for s in feat_df['Suburb'].values:
        if s not in suburbs: suburbs[s] = len(suburbs)
    feat_df['Suburb'] = feat_df['Suburb'].replace(suburbs)

    '''
    feat_df['Type'] = feat_df['Type'].astype('category').cat.codes
    feat_df['Address'] = feat_df['Address'].astype('category').cat.codes
    feat_df['Method'] = feat_df['Method'].astype('category').cat.codes
    feat_df['SellerG'] = feat_df['SellerG'].astype('category').cat.codes
    feat_df['CouncilArea'] = feat_df['CouncilArea'].astype('category').cat.codes
    feat_df['Regionname'] = feat_df['Regionname'].astype('category').cat.codes
    '''

    list_columns = ['Type', 'Address', 'Method', 'SellerG', 'CouncilArea', 'Regionname']

    for i in range(0, len(list_columns)):
        feat_df[list_columns[i]] = create_categories(feat_df, list_columns[i])

    #Convert datetime to a number
    feat_df['Date'] = pd.to_datetime(feat_df['Date'], infer_datetime_format=True)
    feat_df['Date'] = feat_df['Date'].astype(np.int64)

    #print(feat_df)
    rf = RandomForestRegressor()
    rf.fit(feat_df, y)

    X = feat_df
    return X, y





