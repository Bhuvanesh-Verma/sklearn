# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:25:10 2020

@author: Le Champion
"""



import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def calcError(train_X,train_Y,x):
    model = RandomForestRegressor(max_leaf_nodes=400,random_state=1)
    model.fit(train_X,train_Y)
    predictions = model.predict(x)
    return predictions
    




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

x = test[features]
#print(X.columns[X.isna().any()].tolist())  # to check if X contain any NaN columns

train_X,test_x,train_Y,test_y = train_test_split(X,y,random_state=1)


#working with missed data

#2 Imputation

my_imputer = SimpleImputer()

imputed_test_data = pd.DataFrame(my_imputer.fit_transform(x))

imputed_test_data.columns = x.columns

print(test)
prediction_list = calcError(train_X,train_Y,imputed_test_data)
print(x)

output = pd.DataFrame({'Id': test.Id,
                       'SalePrice': prediction_list})
output.to_csv('submission.csv', index=False)

