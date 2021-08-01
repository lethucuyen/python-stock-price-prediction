import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
from xgboost import XGBRegressor


start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

df = web.DataReader("NOK",'yahoo',start,end)

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
print(Y)
print(y_pred)
# print(model)