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
import pickle
import time

# Train Data
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

# Load Test Data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()


test_start_xgboost = dt.datetime(2021, 1, 1)

stock_choices = [
    {"label": "Facebook", "value": "FB"},
    {"label": "Nokia", "value": "NOK"},
    {"label": "Tesla", "value": "TSLA"},
    {"label": "Netflix", "value": "NFLX"},
    {"label": "Apple", "value": "AAPL"},
]

stocks = []
for _ in stock_choices:
    stocks.append(_["value"])

data = web.DataReader(stocks, "yahoo", test_start, test_end)


#########################################################################
# Get prediction data series by stock sticker
def get_predict_by_sticker(modelName, sticker):
    if (modelName == "lstm" or modelName == "RNN"):
        dt = web.DataReader(sticker, "yahoo", start, end)

        # Prepare Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(
            data["Adj Close"].values.reshape(-1, 1))
        prediction_days = 60

        test_data = web.DataReader(sticker, "yahoo", test_start, test_end)
        print("..test data Adj Close: ", test_data["Adj Close"])

        total_dataset = pd.concat(
            (dt["Adj Close"], test_data["Adj Close"]), axis=0)

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
        model = load_model(f"saved_{modelName}_closed_model_{sticker}.h5")
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        print("..Predict: ", predicted_prices)

        test_data["PredictionLSTM"] = predicted_prices
        return predicted_prices
    else:
        dataXGBoost = web.DataReader(sticker, "yahoo", test_start, test_end)
        drop_cols = ["Volume", "Open", "Low", "High", "Close"]
        dataXGBoost = dataXGBoost.drop(drop_cols, 1)
        if(modelName == "XGB"):
            dataXGBoost[f"XGBOOST_predict_next_price_{sticker}"] = XGBOOST_predict_next_price(
                sticker)
            # next_price_xgboost = dataXGBoost[f"XGBOOST_predict_next_price_{sticker}"][-1]
            dataXGBoost[f"XGBOOST_predict_next_price_{sticker}"] = dataXGBoost[f"XGBOOST_predict_next_price_{sticker}"].shift(
                1)
            dataXGBoost.dropna(inplace=True)
            print("..dataXGBoost: ", dataXGBoost)
            # print("..predicted_tommorow:", next_price_xgboost)
            return dataXGBoost[f"XGBOOST_predict_next_price_{sticker}"]
        else:
            XGBOOST_RSI_MA_Data = XGBOOST_RSI_MA_predict_next_price(sticker)
            # Next_Price_XGBOOST_RSI_MA_Data = XGBOOST_RSI_MA_Data[
            #     f"XGBOOST_RSI_MA_predict_next_price_{sticker}"][-1]
            XGBOOST_RSI_MA_Data[f"XGBOOST_RSI_MA_predict_next_price_{sticker}"] = XGBOOST_RSI_MA_Data[
                f"XGBOOST_RSI_MA_predict_next_price_{sticker}"].shift(1)
            XGBOOST_RSI_MA_Data.dropna(inplace=True)
            print("..predicted_prices  ", XGBOOST_RSI_MA_Data)
            # print("..predicted_tommorow:", Next_Price_XGBOOST_RSI_MA_Data)
            return XGBOOST_RSI_MA_Data[f"XGBOOST_RSI_MA_predict_next_price_{sticker}"]


# (Helper)
# Relative Strength Index
def relative_strength_idx(df, n=14):
    close = df["Adj Close"]
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


# (Helper)
def MA(df):
    df["EMA_9"] = df["Adj Close"].ewm(9).mean().shift()
    df["SMA_5"] = df["Adj Close"].rolling(5).mean().shift()
    df["SMA_10"] = df["Adj Close"].rolling(10).mean().shift()
    df["SMA_15"] = df["Adj Close"].rolling(15).mean().shift()
    df["SMA_30"] = df["Adj Close"].rolling(30).mean().shift()


# (Helper)
def MACD(df):
    EMA_12 = pd.Series(df["Adj Close"].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df["Adj Close"].ewm(span=26, min_periods=26).mean())
    df["MACD"] = pd.Series(EMA_12 - EMA_26)
    df["MACD_signal"] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())


# [XGBOOST_RSI_MA Method] Get prediction output in tomorrow by stock sticker
def XGBOOST_RSI_MA_predict_next_price(sticker):
    test_data = web.DataReader(sticker, "yahoo", test_start_xgboost, test_end)

    test_data["RSI"] = relative_strength_idx(test_data).fillna(0)
    MA(test_data)
    MACD(test_data)
    # test_data["Adj Close"] = test_data["Adj Close"].shift(-1)
    print("..data adj: ", test_data)
    test_data = test_data.iloc[33:]
    # test_data = test_data[:-1]
    drop_cols = ["Volume", "Open", "Low", "High", "Close"]
    test_data = test_data.drop(drop_cols, 1)
    print("..DF: ", test_data)

    datasetX = test_data.drop(["Adj Close"], 1)

    X = datasetX.values
    model = pickle.load(open(f"XGB_RSI_MA_{sticker}_Model.pkl", "rb"))
    y_pred = model.predict(X)
    predicted_prices = test_data.copy()
    predicted_prices[f"XGBOOST_RSI_MA_predict_next_price_{sticker}"] = y_pred
    # return y_pred[-1]
    return predicted_prices

# [XGBOOST Method] Get prediction output in tomorrow by stock sticker


def XGBOOST_predict_next_price(sticker):
    test_data = web.DataReader(sticker, "yahoo", test_start_xgboost, test_end)
    datasetX = test_data["Adj Close"].copy()
    X = datasetX.values
    model = pickle.load(open(f"XGB_{sticker}_Model.pkl", "rb"))
    y_pred = model.predict(X)
    print("..Xgboost", y_pred)
    # return y_pred[-1]
    return y_pred


# [LSTM/RNN Method] Get prediction output in n days by stock sticker
def predict_next_n_day(modelName, sticker, n):
    data = web.DataReader(sticker, "yahoo", start, end)
    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Adj Close"].values.reshape(-1, 1))

    prediction_days = 60

    test_data = web.DataReader(sticker, "yahoo", test_start, test_end)

    total_dataset = pd.concat(
        (data["Adj Close"], test_data["Adj Close"]), axis=0)

    model_inputs = total_dataset[len(
        total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Predict Next day
    real_data = [
        model_inputs[len(model_inputs)+n-prediction_days:len(model_inputs)+n, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(
        real_data, (real_data.shape[0], real_data.shape[1], 1))

    model = load_model(f"saved_{modelName}_closed_model_{sticker}.h5")
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    return prediction


# [??? Method] Get output of rate of change in n days by stock sticker
def predictPOC(predict_price, close_price_today):
    price_change = predict_price-close_price_today
    POC = price_change/close_price_today
    POC = POC*100
    POC = np.round(POC, 2)
    return POC


#########################################################################
method_choices = [
    {"label": "LSTM", "value": "lstm"},
    {"label": "RNN", "value": "RNN"},
    {"label": "XGBoost", "value": "XGB"},
    {"label": "XGBoost RSI/MA", "value": "XGB_RSI_MA"},
]
feature_choices = ["Close", "Price Of Change"]
disabledDays = False

#########################################################################
# Style Dash Application
external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?family=Roboto&display=swap",
        "rel": "stylesheet",
    },
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Dashboard Phân Tích Giá Chứng Khoán"


#########################################################################
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
                        html.Div(
                            children=[
                                html.Div(children="Loại phương pháp dự đoán",
                                         className="menu-title"),
                                dcc.Dropdown(
                                    id="pred-method-type",
                                    options=[
                                        {"label": method_type["label"],
                                            "value": method_type["value"]}
                                        for method_type in method_choices
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
                                        {"label": feature_type,
                                            "value": feature_type}
                                        for feature_type in np.sort(feature_choices)
                                    ],
                                    value=["Close", "Price Of Change"],
                                    multi=True,
                                    clearable=False,
                                    className="dropdown",
                                )
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Div(children="Công ty",
                                         className="menu-title"),
                                dcc.Dropdown(
                                    id="pred-company",
                                    options=[
                                        {"label": stock["label"],
                                            "value": stock["value"]}
                                        for stock in stock_choices
                                    ],
                                    value=["FB"],
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
                        # html.Div(
                        #     children=[
                        #         html.Div(
                        #             children="Chọn thời điểm để dự đoán (đvị: ngày)", className="menu-title"
                        #         ),
                        #         dcc.Input(
                        #             id="pred-number",
                        #             type="number",
                        #             placeholder="n ngày",
                        #             value=1,
                        #             min=1,
                        #             max=100,
                        #             step=3,
                        #             className="input"
                        #         )
                        #     ]
                        # ),
                    ],
                    className="form1",
                ),
                html.Div(
                    children=dcc.Loading(
                        id="loading-1",
                        type="circle",
                        children=html.Div(id="loading-output-1"),
                        style={"textAlign": "right"}
                    ),
                ),
                html.P("👉 Xem biểu đồ các dữ liệu training, validation and test data bên dưới", style={
                       "textAlign": "right"}),
                html.Div(
                    children=[
                        html.Div([
                            html.P(
                                "Dự Doán: Nhập số ngày rồi nhấn Lấy Kết Quả 👈"),
                            html.Div(
                                dcc.Input(
                                    id="input-on-submit", type="number", placeholder="Số ngày")
                            ),
                            html.Button("Lấy Kết Quả",
                                        id="submit-val", n_clicks=0),
                        ]),
                        html.Div(
                            children=[
                                html.Div(
                                    children=dcc.Loading(
                                        id="loading-2",
                                        type="circle",
                                        children=html.Div(
                                            id="loading-output-2"),
                                        style={"textAlign": "right"}
                                    ),
                                ),
                                html.Div(
                                    "Kết quả", className="menu-title"),
                                html.Div(id="container-result",
                                         children="23", className="result-box"),
                                html.P(id="container-poc-result",
                                       className="menu-title")
                            ],
                        ),
                    ],
                    className="form2",
                ),
            ],
            className="menu"
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        dcc.Graph(
                            id="pred-price-chart",
                            config={"displayModeBar": False},
                        ),
                    ],
                    className="card",
                ),
            ],
            className="wrapper",
        ),
    ]
)

#########################################################################
# Add Interactivity to Dash Apps Using Callbacks


# Get result value of stock prediction
@app.callback(
    [
        Output("container-result", "children"),
        Output("container-poc-result", "children"),
        Output("loading-output-2", "children")
    ],
    [Input("submit-val", "n_clicks")],
    [
        State("input-on-submit", "value"),
        State("pred-method-type", "value"),
        State("pred-feature-type", "value"),
        State("pred-company", "value"),
    ])
def update_output(n_clicks, value, method_type, feature_type, companies):
    if(value == None and (method_type == "lstm" or method_type == "RNN")):
        return "", "", ""
    results = []
    pocs_results = []
    for stock in companies:
        result = None
        if(method_type == "lstm" or method_type == "RNN"):
            result = predict_next_n_day(method_type, stock, value)
            results.append(f"{stock}: {np.round(float(result[0][0]), 2)}")
            predict_price = result[0][0]
        elif (method_type == "XGB"):
            dataXGBoost = XGBOOST_predict_next_price(stock)
            result = dataXGBoost[-1]
            results.append(f"{stock}: {np.round(float(result), 2)}")
            predict_price = result
        elif (method_type == "XGB_RSI_MA"):
            XGBOOST_RSI_MA_Data = XGBOOST_RSI_MA_predict_next_price(stock)
            result = XGBOOST_RSI_MA_Data[f"XGBOOST_RSI_MA_predict_next_price_{stock}"][-1]
            results.append(f"{stock}: {np.round(float(result), 2)}")
            predict_price = result

        stock_data = data["Adj Close"][stock]
        close_price_today = stock_data[-1]
        poc = predictPOC(predict_price, close_price_today)
        pocs_results.append(f"{stock}: {poc}")

        print(f"..result = {stock}: {result} (POC={poc})")

    if(len(results) == 0):
        return "", "", ""
    pocText = ""
    if("Price Of Change" in feature_type):
        pocText = "Price Of Change: " + \
            (" | ".join((str(val) for val in pocs_results if val)))
    return " | ".join((str(val) for val in results if val)), pocText, ""


@app.callback(
    [
        Output("input-on-submit", "value"),
        Output("input-on-submit", "disabled"),
    ],
    Input("pred-method-type", "value"),
)
def update_input(method_type):
    if(method_type == "lstm" or method_type == "RNN"):
        return None, False
    return 1, True


# Update Chart
@app.callback(
    [
        Output("pred-price-chart", "figure"),
        Output("loading-output-1", "children")
    ],
    [
        Input("pred-method-type", "value"),
        Input("pred-feature-type", "value"),
        Input("pred-company", "value"),
    ]
)
def update_charts(method_type, feature_type, companies):
    try:
        traces = []
        # method_type = "lstm"  # lstm | RNN | XGB | XGB_RSI_MA
        for stock in companies:
            chart_series = get_predict_by_sticker(method_type, stock)
            print("..input of charts:", chart_series)
            data[f"Prediction {stock}"] = chart_series
            # data[f"Prediction {stock}"] = get_predict_by_sticker(
            #     method_type, stock)

            traces.append(
                go.Scatter(
                    x=data.index,
                    y=data["Adj Close"][stock],
                    mode="lines", opacity=0.5,
                    name=f"Close {stock}",
                    textposition="bottom center")
            )
            traces.append(
                go.Scatter(
                    x=data.index,
                    y=data[f"Prediction {stock}"],
                    mode="lines", opacity=0.6,
                    # visible="legendonly",
                    name=f"Prediction{method_type} {stock}",
                    textposition="bottom center")
            )

        chart_data = traces
        pred_price_chart_figure = {
            "data": chart_data,
            "layout": go.Layout(
                colorway=[
                    "#5E0DAC", "#FF4F00", "#375CB1",
                    "#FF7400", "#FFF400", "#FF0056"
                ],
                height=600,
                title=f"...",
                xaxis={
                    "title": "Date",
                    "rangeselector": {
                      "buttons": list([
                          {"count": 1, "label": "1M", "step": "month",
                              "stepmode": "backward"},
                          {"count": 6, "label": "6M", "step": "month",
                              "stepmode": "backward"},
                          {"step": "all"}
                      ])
                    },
                    "rangeslider": {"visible": True}, "type": "date"
                },
                yaxis={"title": "Price (USD)"}
            )
        }
        time.sleep(1)
        return pred_price_chart_figure, ""
    except NameError:
        print(NameError)
    except:
        print("Something else went wrong")

    time.sleep(1)
    return {}, ""


#########################################################################
# Run
if __name__ == "__main__":
    app.run_server(debug=True)
