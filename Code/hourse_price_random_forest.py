# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:25:10 2020

@author: Le Champion
"""



import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


def calcError(train_X,train_Y,text_x):
    model = RandomForestRegressor(max_leaf_nodes=400,random_state=1)
    model.fit(train_X,train_Y)

    predictions = model.predict(test_x)
    return mean_absolute_error(test_y,predictions)
    




data = pd.read_csv("../Datasets/home-data-for-ml-course/train.csv")

test = pd.read_csv("../Datasets/home-data-for-ml-course/test.csv")

y = data.SalePrice


features = ["PoolArea","TotRmsAbvGrd","2ndFlrSF","1stFlrSF",
            "YearBuilt","YearRemodAdd","LotArea","MSSubClass","OpenPorchSF","Fireplaces",
            "GrLivArea","YrSold","3SsnPorch","WoodDeckSF","HalfBath","FullBath",
        "EnclosedPorch","OverallQual","OverallCond", 'BsmtFinSF1',
      'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
        'BsmtHalfBath']


X = data[features]

#print(X.columns[X.isna().any()].tolist())  # to check if X contain any NaN columns

print("Training Model Error Results")
#print(test.columns[test.isna().any()].tolist())


train_X,test_x,train_Y,test_y = train_test_split(X,y,random_state=1)

print(calcError(train_X,train_Y,test_x))

#working with missed data

#1. Removing Columns with no missing data

cols_with_missing = [col for col in test.columns
                     if test[col].isnull().any()]
reduced_test_data = test.drop(cols_with_missing,axis=1)

print("Test Model with removed columns Results")
print(calcError(train_X,train_Y,reduced_test_data))

