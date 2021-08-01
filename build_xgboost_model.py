import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as web
import xgboost as xgb
from sklearn.model_selection import  GridSearchCV
from dash.dependencies import Input, Output


from keras import Input


#load Data



start = dt.datetime(2012,1,1)
end = dt.datetime.now()

df = web.DataReader("NOK",'yahoo',start,end)

df['Date']=df.index
print("Data: ",df['Adj Close'])
print("Data: ",df['Date'])
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

df['Adj Close']=df['Adj Close'].shift(-1)
df = df.iloc[33:]
df = df[:-1]
df.index = range(len(df))

test_size  = 0.15
valid_size = 0.15

test_split_idx  = int(df.shape[0] * (1-test_size))
valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

train_df  = df.loc[:valid_split_idx].copy()
valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
test_df   = df.loc[test_split_idx+1:].copy()

drop_cols = ['Date', 'Volume', 'Open', 'Low', 'High','Close']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df  = test_df.drop(drop_cols, 1)


y_train = train_df['Adj Close'].copy()
X_train = train_df.drop(['Adj Close'], 1)

y_valid = valid_df['Adj Close'].copy()
X_valid = valid_df.drop(['Adj Close'], 1)

y_test  = test_df['Adj Close'].copy()
X_test  = test_df.drop(['Adj Close'], 1)

X_train.info()

#time
parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
clf = GridSearchCV(model, parameters)

clf.fit(X_train, y_train)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')

#Build the model LMST




#model.save("saved_xgboost_closed_model_NOK.h5")

