import dash
import dash_core_components as dcc
import dash_html_components as html
from keras.backend import log
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas_datareader as web
import datetime as dt

# Train Data
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

# Load Test Data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

stocks = ['FB', 'NOK', 'TSLA']
data = web.DataReader(stocks, 'yahoo', test_start, test_end)


def get_predict_by_sticker(modelName, sticker):
    dt = web.DataReader(sticker, 'yahoo', start, end)

    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1,1))
    prediction_days = 60

    test_data = web.DataReader(sticker, 'yahoo', test_start, test_end)

    total_dataset = pd.concat(
        (dt['Adj Close'], test_data['Adj Close']), axis=0)

    model_inputs = total_dataset[len(
        total_dataset)-len(test_data)-prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make predictions on Test Data

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict LSTM close price

    model = load_model(f'saved_{modelName}_closed_model_{sticker}.h5')

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices


method_choices = ["lstm", "XGBoost", "RNN"]
feature_choices = ["Close", "Price Of Change"]


# Style Dash Application
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Roboto&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Dashboard Phân Tích Giá Chứng Khoán"


# Defining the Layout of Dash Application
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.P(children="📈", className="header-emoji"),
                html.H1(
                    children="Dashboard Phân Tích Giá Chứng Khoán", className="header-title"
                ),
                html.P(
                    children="Phân tích và dự đoán"
                    " biến động của giá chứng khoán"
                    " của Tesla, Apple, Facebook, ..."
                    " từ năm 2013 đến năm 2018",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(children="Loại phương pháp dự đoán",
                                 className="menu-title"),
                        dcc.Dropdown(
                            id="pred-method-type",
                            options=[
                                {"label": method_type, "value": method_type}
                                for method_type in np.sort(method_choices)
                            ],
                            value="lstm",
                            clearable=False,
                            className="dropdown",
                        )
                    ]
                ),
                html.Div(
                    children=[
                        html.Div(children="Loại đặc trưng để dự đoán",
                                 className="menu-title"),
                        dcc.Dropdown(
                            id="pred-feature-type",
                            options=[
                                {"label": feature_type, "value": feature_type}
                                for feature_type in np.sort(feature_choices)
                            ],
                            value="Close",
                            multi=True,
                            clearable=False,
                            className="dropdown",
                        )
                    ]
                ),
                # html.Div(
                #     children=[
                #         html.Div(
                #             children="Khoảng thời gian", className="menu-title"
                #         ),
                #         dcc.DatePickerRange(
                #             id="date-range",
                #             min_date_allowed=test_start.date(),
                #             max_date_allowed=test_end.date(),
                #             start_date=test_start.date(),
                #             end_date=test_end.date(),
                #         ),
                #     ]
                # ),
                html.Div(
                    children=[
                        html.Div(
                            children="Chọn thời điểm để dự đoán (đvị: ngày)", className="menu-title"
                        ),
                        dcc.Input(
                            id="pred-number",
                            type="number",
                            placeholder="n ngày",
                            value=1,
                            min=1,
                            max=100,
                            step=3,
                            className="input"
                        )
                    ]
                ),
            ],
            className="menu"
        ),
        html.Div(
            children=[
                html.Div(
                    children=dcc.Graph(
                        id="pred-price-chart",
                        config={"displayModeBar": False},
                    ),
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)


# Add Interactivity to Dash Apps Using Callbacks
@app.callback(
    Output("pred-price-chart", "figure"),
    [
        Input("pred-method-type", "value"),
        Input("pred-feature-type", "value"),
        Input("pred-number", "value"),
        # Input("date-range", "start_date"),
        # Input("date-range", "end_date"),
    ]
)
def update_charts(method_type, feature_type, number):
    traces = []

    #method_type = "lstm"  # lstm | RNN | ???

    for stock in stocks:
        data[f'Prediction {stock}'] = get_predict_by_sticker(
            method_type, stock)

        traces.append(
            go.Scatter(
                x=data.index,
                y=data['Adj Close'][stock],
                mode='lines', opacity=0.5,
                name=f'Close {stock}',
                textposition='bottom center')
        )
        traces.append(
            go.Scatter(
                x=data.index,
                y=data[f'Prediction {stock}'],
                mode='lines', opacity=0.6,
                visible='legendonly',
                name=f'Prediction{method_type} {stock}',
                textposition='bottom center')
        )

    chart_data = traces
    pred_price_chart_figure = {
        'data': chart_data,
        'layout': go.Layout(
            colorway=[
                "#5E0DAC", '#FF4F00', '#375CB1',
                '#FF7400', '#FFF400', '#FF0056'
            ],
            height=600,
            title=f"...",
            xaxis={
                "title": "Date",
                'rangeselector': {
                  'buttons': list([
                      {'count': 1, 'label': '1M', 'step': 'month',
                          'stepmode': 'backward'},
                      {'count': 6, 'label': '6M', 'step': 'month',
                          'stepmode': 'backward'},
                      {'step': 'all'}
                  ])
                },
                'rangeslider': {'visible': True}, 'type': 'date'
            },
            yaxis={"title": "Price (USD)"}
        )
    }
    return pred_price_chart_figure


# Run
if __name__ == '__main__':
    app.run_server(debug=True)
