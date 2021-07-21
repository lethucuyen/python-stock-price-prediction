import pandas_datareader as web
import datetime as dt

start = dt.datetime(2012, 1, 1)
end = dt.datetime.now()

stocks = ['NOK', 'TSLA','PEP','AMZN','GPS','HSBC']

data_set = web.DataReader(stocks, 'yahoo', start, end)
data_set_close = data_set['Close']
data_set_close.to_csv(r'D:\CNM\AI\nhom\python-stock-price-prediction\table.csv',columns=data_set_close.columns,index=True)
