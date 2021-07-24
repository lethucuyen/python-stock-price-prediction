import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model

import pandas_datareader as web
import datetime as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

app = dash.Dash()
server = app.server

stocks = ['NOK', 'TSLA','PEP','AMZN','GPS','HSBC']


#train data
start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

#Load Test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

sample = web.DataReader("NOK", 'yahoo', test_start, test_end)

def get_predict_by_sticker(sticker):
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

    model = load_model("saved_lstm_closed_model_NOK.h5")

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    print("Predict: ",predicted_prices)
    test_data['PredictionLSTM'] = predicted_prices
    return predicted_prices

def predict_next_n_day(sticker,n):
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

    model = load_model("saved_lstm_closed_model_NOK.h5")
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction


sample['PredictionLSTM'] = get_predict_by_sticker("NOK")
a = predict_next_n_day("NOK",30)
print("A ",a)

app.layout = html.Div([

    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Stocks High vs Lows",
                        style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
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
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook", "MSFT": "Microsoft", }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x=sample.index,
                       y=sample['Adj Close'],
                       mode='lines', opacity=0.5,
                       name=f'Close {dropdown[stock]}', textposition='bottom center'))
        trace2.append(
            go.Scatter(x=sample.index,
                       y=sample['PredictionLSTM'],
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




if __name__ == '__main__':
    app.run_server(debug=True)