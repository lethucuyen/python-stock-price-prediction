import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas_datareader as web
import datetime as dt
import xgboost as xgb
import pickle


app = dash.Dash()
server = app.server

# scaler = MinMaxScaler(feature_range=(0, 1))
#
# df_nse = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv")
#
# df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
# df_nse.index = df_nse['Date']
#
# data = df_nse.sort_index(ascending=True, axis=0)
#
# close_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])
# for i in range(0, len(data)):
#     close_data["Date"][i] = data['Date'][i]
#     close_data["Close"][i] = data["Close"][i]
#
# roc_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Rate_Of_Change'])
# for i in range(0, len(data)):
#     roc_data["Date"][i] = data['Date'][i]
#     roc_data["Rate_Of_Change"][i] = (data["Close"][len(data) - 1] - data["Close"][i]) / data["Close"][i]
#
#
# closed_grap_data = close_data[987:]


#train data
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

#Load Test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

stocks = ['NOK', 'AAPL','FB','TSLA','NFLX']

# dataE = web.DataReader(stocks, 'yahoo', test_start, test_end)
sample = web.DataReader("NOK", 'yahoo', test_start, test_end)

dataXGBoost=web.DataReader("NFLX", 'yahoo', test_start, test_end)

drop_cols = [ 'Volume', 'Open', 'Low', 'High','Close']

sample = sample.drop(drop_cols, 1)
dataXGBoost = dataXGBoost.drop(drop_cols, 1)
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
# MA
def MA(df):
    df['EMA_9'] = df['Adj Close'].ewm(9).mean().shift()
    df['SMA_5'] = df['Adj Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Adj Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Adj Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Adj Close'].rolling(30).mean().shift()
def MACD(df):
    EMA_12 = pd.Series(df['Adj Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['Adj Close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

def XGBOOST_RSI_MA_predict_next_price(sticker):
    test_data = web.DataReader(sticker, 'yahoo', test_start, test_end)

    test_data['RSI'] = relative_strength_idx(test_data).fillna(0)
    MA(test_data)
    MACD(test_data)
    # test_data['Adj Close'] = test_data['Adj Close'].shift(-1)
    print("data adj: ", test_data)
    test_data = test_data.iloc[33:]
    # test_data = test_data[:-1]
    drop_cols = ['Volume', 'Open', 'Low', 'High', 'Close']

    test_data = test_data.drop(drop_cols, 1)

    print("DF: ", test_data)
    datasetX = test_data.drop(['Adj Close'], 1)

    X = datasetX.values
    model = pickle.load(open(f'XGB_RSI_MA_{sticker}_Model.pkl', "rb"))
    y_pred = model.predict(X)
    predicted_prices = test_data.copy()
    predicted_prices[f'XGBOOST_RSI_MA_predict_next_price_{sticker}'] = y_pred
    # return y_pred[-1]

    return predicted_prices

def XGBOOST_predict_next_price(sticker):
    test_data = web.DataReader(sticker, 'yahoo', test_start, test_end)
    datasetX = test_data['Adj Close'].copy()
    X = datasetX.values
    model = pickle.load(open(f'XGB_{sticker}_Model.pkl', "rb"))
    y_pred = model.predict(X)
    print("Xgboost", y_pred)

    # return y_pred[-1]
    return y_pred
def XGBOOST_predict_n_day(sticker,n):
    test_data = web.DataReader(sticker, 'yahoo', test_start, test_end)
    datasetX = test_data['Adj Close'].copy()
    X = datasetX.values
    model = pickle.load(open(f'XGB_{sticker}_Model.pkl', "rb"))
    for i in range(0, n):
        y_pred = model.predict(X)
        X = y_pred
        print("len", len(y_pred))

    return y_pred[-1]
    # return y_pred

#Du doan su dung LSTM
def LSTM_predict_next_price(data):
    clean_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        clean_data["Date"][i] = data['Date'][i]
        clean_data["Close"][i] = data["Close"][i]
    # for i in range(0,days):
    #     clean_data["Date"][len(data)+i]=clean_data["Date"][len(data) - 1] + pd.DateOffset(days=i+1)
    print("clean_data",clean_data)
    clean_data.index = clean_data.Date
    clean_data.drop("Date", axis=1, inplace=True)

    dataset = clean_data.values

    train = dataset[0:987, :]
    valid = dataset[987:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = load_model("saved_lstm_closed_model.h5")

    inputs = clean_data[len(clean_data) - len(valid) - 60:].values

    inputs = inputs.reshape(-1, 1)

    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    print("closing_price",closing_price,len(closing_price))
    return closing_price[len(closing_price)-1][0]
#update
def get_predict_by_sticker(modelName,sticker):
    data = web.DataReader(sticker, 'yahoo', start, end)
    #Prepare Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1,1))

    prediction_days=60

    test_data = web.DataReader(sticker, 'yahoo', test_start, test_end)

    print("test data Adj Close: ",test_data['Adj Close'])

    total_dataset = pd.concat((data['Adj Close'],test_data['Adj Close']),axis=0)

    model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.transform(model_inputs)

    #Make predictions on Test Data



    x_test = []

    for x in range(prediction_days,len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x,0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


    #predict LSTM close price

    model = load_model(f'saved_{modelName}_closed_model_{sticker}.h5')
    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    print("Predict: ",predicted_prices)
    test_data['PredictionLSTM'] = predicted_prices
    return predicted_prices

def predict_next_n_day(modelName,sticker,n):
    data = web.DataReader(sticker, 'yahoo', start, end)
    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

    prediction_days = 60

    test_data = web.DataReader(sticker, 'yahoo', test_start, test_end)

    total_dataset = pd.concat((data['Adj Close'], test_data['Adj Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

     #Predict Next day
    real_data = [model_inputs[len(model_inputs)+n-prediction_days:len(model_inputs)+n,0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    model = load_model(f'saved_{modelName}_closed_model_{sticker}.h5')
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction

def predictPOC(predict_price,close_price_today):
    price_change=predict_price-close_price_today
    POC=price_change/close_price_today
    POC=POC*100
    POC=np.round(POC,2)
    return POC

a = predict_next_n_day("lstm","NOK",30)
print("A30 ",a)
print("sample['Adj Close'][-1] ",sample['Adj Close'][-1])
a_today=sample['Adj Close'][-1]
pocA=predictPOC(a,a_today)
print("pocA: ",pocA)

lac=XGBOOST_predict_n_day("NFLX",5)
print("LAC: ",lac)

dataXGBoost[f'XGBOOST_predict_next_price_{"NFLX"}']=XGBOOST_predict_next_price("NFLX")
next_price_xgboost=dataXGBoost[f'XGBOOST_predict_next_price_{"NFLX"}'][-1]
dataXGBoost[f'XGBOOST_predict_next_price_{"NFLX"}']=dataXGBoost[f'XGBOOST_predict_next_price_{"NFLX"}'].shift(1)
dataXGBoost.dropna(inplace=True)
print("dataXGBoost: ",dataXGBoost)


XGBOOST_RSI_MA_Data=XGBOOST_RSI_MA_predict_next_price("NFLX")
Next_Price_XGBOOST_RSI_MA_Data=XGBOOST_RSI_MA_Data[f'XGBOOST_RSI_MA_predict_next_price_{"NFLX"}'][-1]
XGBOOST_RSI_MA_Data[f'XGBOOST_RSI_MA_predict_next_price_{"NFLX"}']=XGBOOST_RSI_MA_Data[f'XGBOOST_RSI_MA_predict_next_price_{"NFLX"}'].shift(1)
XGBOOST_RSI_MA_Data.dropna(inplace=True)
print("predicted_prices  ",XGBOOST_RSI_MA_Data)
print("predicted_prices RSI  ",XGBOOST_RSI_MA_Data['RSI'])


app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Stocks High vs Lows",
                        style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Nokia', 'value': 'NOK'},{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['NOK'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Nokia', 'value': 'NOK'},{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])

    ])
])

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"NOK": "Nokia","TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    trace2 = []


    for stock in selected_dropdown:
        sample[f'PredictionLSTM {stock}'] = get_predict_by_sticker("lstm",stock)
        trace1.append(
            go.Scatter(x=sample.index,
                       y=sample['Adj Close'],
                       mode='lines', opacity=0.5,
                       name=f'Close {dropdown[stock]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=sample.index,
                       y=sample[f'PredictionLSTM {stock}'],
                       mode='lines', opacity=0.6,
                       visible='legendonly',
                       name=f'Prediction LSTM {dropdown[stock]}', textposition='bottom center'))


    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
                                  title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month',
                                                                             'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'},
                                  yaxis={"title": "Price (USD)"})}
    return figure




if __name__=='__main__':
	app.run_server(debug=True)