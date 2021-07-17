import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import pandas_datareader as web

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import streamlit as st

from PIL import Image


#load Data
company = 'NOK'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

data = web.DataReader(company,'yahoo',start,end)


#Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

prediction_days=60

x_train=[]
y_train=[]

for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#Build the model
model =Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # prediction of the next closing price

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=25,batch_size=32)



#Test the model accuracy on existing data

#Load Test Data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data =web.DataReader(company,'yahoo',test_start,test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'],test_data['Close']),axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Make predictions on Test Data

x_test = []

for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


#plot the Test Predictions

#plt.plot(actual_prices,color="black",label = f"Actual {company} Price")
#plt.plot(predicted_prices,color="green",label = f"Predicted {company} Price")
#plt.title(f"{company} Share Price")
#plt.xlabel('Time')
#plt.ylabel(f'{company} Share Price')
#plt.legend()
#plt.show()

# Predict Next Day

real_data = [model_inputs[len(model_inputs) +1 - prediction_days:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")


# web app
st.write("""
# Stock Market Web Application
**Visually** show data on a stock! Data range from jan 2,2020 - Aug 4,2020
""")
image = Image.open("D:/CNM/AI/nhom/python-stock-price-prediction/stock-image.jpg")
st.image(image, use_column_width=True)

# Create a sidebar header
st.sidebar.header('User Input')


df = web.DataReader(company,'yahoo',"2020-01-01",test_end)
df['Date'] = df.index
df['Predictions'] = predicted_prices

# Create a function to get the users input
def get_input():
    start_date = st.sidebar.text_input("Start Date", "2020-01-02")
    end_date = st.sidebar.text_input("End Date", "2020-08-04")
    stock_symbol = st.sidebar.text_input("Stock Symbol", "NOK")
    return start_date, end_date, stock_symbol


def get_company_name(symbol):
    return symbol


def get_data(symbol, startFe, endFe):
    start = pd.to_datetime(startFe)
    end = pd.to_datetime(endFe)

    start_row = 0
    end_row = 0


    for i in range(0,len(df)):
        if start <= pd.to_datetime(df['Date'][i]):
            start_row = i
            break

    for j in range(0,len(df)):
        if end >= pd.to_datetime(df['Date'][len(df)-1-j]):
            end_row = len(df)-1-j
            break

    return df.iloc[start_row:end_row+1,:]


startFe, endFe, symbol = get_input()
dfData = get_data(symbol,startFe,endFe)
company_name = get_company_name(symbol.upper())

st.header(company_name+" Close price\n")
st.line_chart(dfData['Close'])

st.header(company_name+" Predict Close price\n")
st.line_chart(dfData['Predictions'])