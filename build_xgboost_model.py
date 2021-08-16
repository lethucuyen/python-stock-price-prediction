import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
from xgboost import XGBRegressor
import pickle


start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

df = web.DataReader("NOK",'yahoo',start,end)

df = df[['Adj Close']].copy()
df['target']=df['Adj Close'].shift(-1)
df.dropna(inplace=True)
train=df.values
print("df ", df)
print(len(train))

X = train[:,:-1]
Y = train[:,-1]
print("X: ",X)
model = XGBRegressor(objective="reg:squarederror",n_estimators=1000)
model.fit(X, Y)
pickle.dump(model, open("XGB_NOK_Model.pkl", "wb"))

y_pred = model.predict(X)
print(Y)
print(y_pred)


# dataset=df.values
# df['Adj Close']=df['Adj Close'].shift(-1)
# print("shift close", df)
# df = df[:-1]
#
# print(":-1 close", df)
#
# drop_cols = ['Volume','Close']
# df = df.drop(drop_cols, 1)
# # split data into X and y
# datasetY = df['Adj Close'].copy()
# datasetX = df.drop(['Adj Close'], 1)
#
# Y = datasetY.values
# X = datasetX.values
#
# model = XGBRegressor()
# model.fit(X, Y)
# pickle.dump(model, open("XGB_NFLX_Model.pkl", "wb"))
#
# y_pred = model.predict(X)
# print(Y)
# print(y_pred)

