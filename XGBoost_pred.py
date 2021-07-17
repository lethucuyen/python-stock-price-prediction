# First XGBoost model for Pima Indians dataset
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
# load data
# df=pd.read_csv("diabetes.csv")
df=pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
df.head()

dataset=df.values


# split data into X and y
X = dataset[:,1:5]
Y = dataset[:,5]


# split data into train and test sets
# seed = 7
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# eval_set = [(X_train, y_train)]
# fit model no training data
model = XGBRegressor()
model.fit(X, Y)
model.save_model("xgboost_closed_model.model")
# make predictions for test data
y_pred = model.predict(X)
predictions = [round(value) for value in y_pred]
print(Y)
print(y_pred)
# print(model)

