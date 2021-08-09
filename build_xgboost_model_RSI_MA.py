import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
from xgboost import XGBRegressor
import pickle

#from sklearn.model_selection import  GridSearchCV

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn import preprocessing




#load Data



start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

df = web.DataReader("NOK",'yahoo',start,end)

#MA
df['EMA_9'] = df['Adj Close'].ewm(9).mean().shift()
df['SMA_5'] = df['Adj Close'].rolling(5).mean().shift()
df['SMA_10'] = df['Adj Close'].rolling(10).mean().shift()
df['SMA_15'] = df['Adj Close'].rolling(15).mean().shift()
df['SMA_30'] = df['Adj Close'].rolling(30).mean().shift()

#Relative Strength Index
def relative_strength_idx(df, n=14):
    close = df['Adj Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

#MACD
EMA_12 = pd.Series(df['Adj Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(df['Adj Close'].ewm(span=26, min_periods=26).mean())
df['MACD'] = pd.Series(EMA_12 - EMA_26)
df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())
df['RSI']=relative_strength_idx(df).fillna(0)


print("before: ",df['Adj Close'])

df['Adj Close']=df['Adj Close'].shift(-1)
df = df.iloc[33:]
df = df[:-1]
#df.index = range(len(df))


drop_cols = [ 'Volume', 'Open', 'Low', 'High','Close']

df = df.drop(drop_cols, 1)

print("DF: ",df)
datasetY = df['Adj Close'].copy()
datasetX = df.drop(['Adj Close'], 1)

Y = datasetY.values
X = datasetX.values

model = XGBRegressor()
model.fit(X, Y)
# model.save_model('0001.model')
pickle.dump(model, open("XGB_RSI_MA_Model.pkl", "wb"))

# make predictions for test data
y_pred = model.predict(X)
print(Y)
print(y_pred)
print("last item: ",y_pred[-1])
#Build the model LMST




#model.save("saved_xgboost_closed_model_NOK.h5")

