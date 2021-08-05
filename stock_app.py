import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

df_nse = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv")

df_nse["Date"]=pd.to_datetime(df_nse.Date,format="%Y-%m-%d")
df_nse.index = df_nse['Date']

data = df_nse.sort_index(ascending=True, axis=0)

close_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    close_data["Date"][i] = data['Date'][i]
    close_data["Close"][i] = data["Close"][i]

roc_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Rate_Of_Change'])
for i in range(0, len(data)):
    roc_data["Date"][i] = data['Date'][i]
    roc_data["Rate_Of_Change"][i] = (data["Close"][len(data) - 1] - data["Close"][i]) / data["Close"][i]


closed_grap_data = close_data[987:]




# df= pd.read_csv("./stock_data.csv")
app.layout=html.Div([

                html.Div(id='output_div')
])
app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    html.Div([

    ]),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data',children=[
			html.Div([
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Input(id="dateInput", type="number", placeholder="Enter number here",style={"display": "block", 'margin':'auto', "width": "50%"}),
                html.Button(id='submit-button', type='submit', children='Submit',style={"display": "block", 'margin':'auto', "width": "10%"}),
				dcc.Graph(
					id="Closed Price Graph",
					figure={
						"data":[
							go.Scatter(
								x=close_data["Date"],
								y=close_data["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='Closed Price Graph',
                            xaxis={"title": "Date",
                                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                       'step': 'month',
                                                                       'stepmode': 'backward'},
                                                                      {'count': 6, 'label': '6M',
                                                                       'step': 'month',
                                                                       'stepmode': 'backward'},
                                                                      {'step': 'all'}])},
                                   'rangeslider': {'visible': True}, 'type': 'date'},
							yaxis={'title':'Closed Price'}
						)
					},
				),
                dcc.Input(id="Roc-dateInput", type="number", placeholder="Enter number here",style={"display": "block", 'margin':'auto', "width": "50%"}),
                html.Button(id='Roc-submit-button', type='submit', children='Submit',style={"display": "block", 'margin':'auto', "width": "10%"}),
                dcc.Graph(
					id="Rate of Change Graph",
					figure={
						"data":[
							go.Scatter(
								x=roc_data["Date"],
								y=roc_data["Rate_Of_Change"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='Rate Of Change Graph',
                            xaxis={"title": "Date",
                                  'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                                      'step': 'month',
                                                                      'stepmode': 'backward'},
                                                                     {'count': 6, 'label': '6M',
                                                                      'step': 'month',
                                                                      'stepmode': 'backward'},
                                                                     {'step': 'all'}])},
                                  'rangeslider': {'visible': True}, 'type': 'date'},
							yaxis={'title':'Rate Of Change'}
						)
					},
				)
			])        		


        ]),
        # dcc.Tab(label='Facebook Stock Data', children=[
        #     html.Div([
        #         html.H1("Stocks High vs Lows",
        #                 style={'textAlign': 'center'}),
        #
        #         dcc.Dropdown(id='my-dropdown',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple','value': 'AAPL'},
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft','value': 'MSFT'}],
        #                      multi=True,value=['FB'],
        #                      style={"display": "block", "margin-left": "auto",
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='highlow'),
        #         html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
        #
        #         dcc.Dropdown(id='my-dropdown2',
        #                      options=[{'label': 'Tesla', 'value': 'TSLA'},
        #                               {'label': 'Apple','value': 'AAPL'},
        #                               {'label': 'Facebook', 'value': 'FB'},
        #                               {'label': 'Microsoft','value': 'MSFT'}],
        #                      multi=True,value=['FB'],
        #                      style={"display": "block", "margin-left": "auto",
        #                             "margin-right": "auto", "width": "60%"}),
        #         dcc.Graph(id='volume')
        #     ], className="container"),
        # ])


    ])
])







# @app.callback(
#     Output('highlow', 'figure'),
#     [Input('my-dropdown', 'value')],
# )
# def update_graph(selected_dropdown):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     trace2 = []
#     for stock in selected_dropdown:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["High"],
#                      mode='lines', opacity=0.7,
#                      name=f'High {dropdown[stock]}',textposition='bottom center'))
#         trace2.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Low"],
#                      mode='lines', opacity=0.6,
#                      name=f'Low {dropdown[stock]}',textposition='bottom center'))
#     traces = [trace1, trace2]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                        'step': 'month',
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M',
#                                                        'step': 'month',
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Price (USD)"})}
#     return figure
# @app.callback(Output('volume', 'figure'),
#               [Input('my-dropdown2', 'value')])
# def update_graph(selected_dropdown_value):
#     dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
#     trace1 = []
#     for stock in selected_dropdown_value:
#         trace1.append(
#           go.Scatter(x=df[df["Stock"] == stock]["Date"],
#                      y=df[df["Stock"] == stock]["Volume"],
#                      mode='lines', opacity=0.7,
#                      name=f'Volume {dropdown[stock]}', textposition='bottom center'))
#     traces = [trace1]
#     data = [val for sublist in traces for val in sublist]
#     figure = {'data': data,
#               'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
#                                             '#FF7400', '#FFF400', '#FF0056'],
#             height=600,
#             title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
#             xaxis={"title":"Date",
#                    'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
#                                                        'step': 'month',
#                                                        'stepmode': 'backward'},
#                                                       {'count': 6, 'label': '6M',
#                                                        'step': 'month',
#                                                        'stepmode': 'backward'},
#                                                       {'step': 'all'}])},
#                    'rangeslider': {'visible': True}, 'type': 'date'},
#              yaxis={"title":"Transactions Volume"})}
#     return figure

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

#xuat data su dung RNN
def RNN_predict_next_price(data):
    return 0;
#du doan su dung XGBoost
def XGBoost_predict_next_price(data):
    return 0;

# return data for grahp
@app.callback(
        Output('Closed Price Graph', 'figure'),
        [Input('submit-button', 'n_clicks')],
        [State('dateInput', 'value')],
)
def update_output(clicks,value):
    print(clicks,value);
    if clicks is None:
        dateAhead=0
        print("1")
        df_nse = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv")

        df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
        df_nse.index = df_nse['Date']

        data = df_nse.sort_index(ascending=True, axis=0)

        close_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])
        for i in range(0, len(data)):
            close_data["Date"][i] = data['Date'][i]
            close_data["Close"][i] = data["Close"][i]
    else:
        dateAhead=value
        print("2")
        df_nse = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv")

        df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
        df_nse.index = df_nse['Date']

        data = df_nse.sort_index(ascending=True, axis=0)

        close_data = pd.DataFrame(index=range(0, len(df_nse)), columns=['Date', 'Close'])
        for i in range(0, len(data)):
            close_data["Date"][i] = data['Date'][i]
            close_data["Close"][i] = data["Close"][i]
        for i in range(0, dateAhead):
            Next_Date = close_data["Date"][len(close_data) - 1] + pd.DateOffset(days=1)
            Next_Price=LSTM_predict_next_price(close_data)
            print("check point 1",Next_Date,Next_Price)
            New_row=pd.DataFrame({"Date":[Next_Date],"Close":[Next_Price]})
            print("check point 2",New_row)
            close_data=close_data.append(New_row,ignore_index = True)
            print("check point 3",close_data)
    print("3")
    graph_data=close_data

    print("graph:", graph_data)
    figure = {
        "data":[
            go.Scatter(
            x=graph_data["Date"],
            y=graph_data["Close"],
            mode='markers'
        )

        ],
        "layout":go.Layout(
            title='scatter plot',
            xaxis={'title':'Date'},
            yaxis={'title':'Closed Price'}
        )
    }
    print("4")
    return figure

if __name__=='__main__':
	app.run_server(debug=True)