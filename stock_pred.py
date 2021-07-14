import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

df=pd.read_csv("NSE-Tata-Global-Beverages-Limited.csv")
df.head()

df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

plt.figure(figsize=(16,8))
plt.plot(df["Close"],label='Close Price history')

from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

data=df.sort_index(ascending=True,axis=0)
#close_price dataset
close_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
for i in range(0,len(data)):
    close_dataset["Date"][i]=data['Date'][i]
    close_dataset["Close"][i]=data["Close"][i]
#rate_of_change dataset
roc_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Rate_Of_Change'])
for i in range(0,len(data)):
    roc_dataset["Date"][i]=data['Date'][i]
    roc_dataset["Rate_Of_Change"][i]=(data["Close"][len(data)-1]-data["Close"][i])/data["Close"][i]
#close_price
#Normalize the new filtered dataset
close_dataset.index=close_dataset.Date
close_dataset.drop("Date",axis=1,inplace=True)

final_dataset=close_dataset.values

train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
#Build and train the LSTM model
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))



lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)


lstm_model.save("saved_lstm_closed_model.h5")
#rate_of_change
#Normalize the new filtered dataset
roc_dataset.index=roc_dataset.Date
roc_dataset.drop("Date",axis=1,inplace=True)

final_dataset=roc_dataset.values

train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)

x_train_data,y_train_data=[],[]

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])

x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))
#Build and train the LSTM model
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

inputs_data=roc_dataset[len(roc_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_roc=lstm_model.predict(X_test)
predicted_roc=scaler.inverse_transform(predicted_roc)

print(predicted_roc)
lstm_model.save("saved_lstm_roc_model.h5")

# train_data=close_dataset[:987]
# valid_data=close_dataset[987:]
# valid_data['Predictions']=predicted_closing_price
# plt.plot(train_data["Close"])
# plt.plot(valid_data[['Close',"Predictions"]])