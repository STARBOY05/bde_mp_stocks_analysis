# Import necessary libraries
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import yfinance as yf

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from numpy import array 

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

st.title("Big Data Mini Project -- Stock Analysis and Prediction")

# Tickers and Intervals used for analysis
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BAJFINANCE.NS"]
intervals = ["1m", "1d", "1wk", "1mo"]

# Created a Spark Session
spark = SparkSession.builder \
    .appName("Real-time Stock Data") \
    .getOrCreate()

symbol = st.sidebar.selectbox("Select a stock", tickers)
interval = st.sidebar.selectbox("Select Interval", intervals)

# Code for REAL-TIME Data
def display_real_time_data(ticker, interval):
    st.write(f"Real-time stock data => Ticker: {ticker}, Interval: {interval}")
    if interval == "1m":
      data = yf.download(ticker, period='1d', interval='1m')
    else:
      data = yf.download(ticker, start="2021-01-01", end="2023-05-01", interval=interval)
    data = pd.DataFrame(data)
    data = data.reset_index()
    if interval == "1m":
      data = data[["Datetime", "Close"]]
      st.line_chart(data.set_index('Datetime'))
    else:
      data = data[["Date", "Close"]]
      st.line_chart(data.set_index('Date'))

if st.sidebar.button("Show Real-Time Data"):
  display_real_time_data(symbol, interval)

# Code for PREDICTION
def split_sequence(sequence,steps):
    X,y=[],[]
    sequence=list(sequence)
    for start in range(len(sequence)):
        end_index = start+steps
        if end_index>len(sequence)-1:
            break
        sequence_x,sequence_y = sequence[start:end_index],sequence[end_index]
        X.append(sequence_x)
        y.append(sequence_y)
    return(array(X),array(y))

stock_df = yf.download(symbol, start="2021-01-01", end="2023-05-01")
spark_df = spark.createDataFrame(stock_df.reset_index())

if st.sidebar.button("Stock Prediction"):
    # df = yf.download(symbol, period="1d", interval="1m")
    df = yf.download(symbol, period="max")
    st.subheader("Prediction for {}".format(symbol))
    df = df[['Open', 'High', 'Low', 'Close']]
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    raw_sequence = df["Close"][-300:]
    rs = raw_sequence.copy()

    steps = 3
    pred = []

    with st.spinner('Model is training...'):
    # Predicting the 20 datapoints
      for _ in range(20):
          X, y = split_sequence(rs, steps)
          features = 1
          X = X.reshape((X.shape[0], X.shape[1], features))
          model = Sequential()
          model.add(Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=(steps, features)))
          model.add(MaxPooling1D(pool_size=2))
          model.add(Flatten())
          model.add(Dense(100, activation='relu'))
          model.add(Dense(1))
          model.compile(optimizer='adam', loss='mse')
          model.fit(X, y, epochs=200, verbose=0)  # Change epochs for response time
          x_input = array(rs[-3:])
          x_input = x_input.reshape((1, steps, features))
          y_pred = model.predict(x_input, verbose=0)
          pred.append(y_pred[0][0])
      
      st.success('Model training complete!')
      actual = raw_sequence.tolist() + [None] * 20
      predicted = [None] * 299 + [raw_sequence.tolist()[-1]] + pred
      data = pd.DataFrame({"Actual": actual, "Predicted": predicted})
      st.line_chart(data)

# Code for Analysis
if st.sidebar.button("Show Analysis"):
    # 1. What was the average daily price range (High - Low) for each stock?
    st.subheader("Daily Average Price Range of {}".format(symbol))
    # Compute average daily price range
    avg_price_range_df = spark_df.withColumn("price_range", col("High") - col("Low")).groupBy("Date").agg(avg("price_range").alias("Avg_Daily_Price")).orderBy("Date")
    avg_price_range_pandas = avg_price_range_df.toPandas()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(avg_price_range_pandas)
    avg_price_range_pandas.set_index("Date", inplace=True)
    with col2:
        st.line_chart(avg_price_range_pandas["Avg_Daily_Price"])

    # 2. What was the average daily percentage change in price ((Close - Open) / Open) for each stock?
    st.subheader("Daily Average Percentage Change of {}".format(symbol))
    # Compute average daily percentage change
    avg_change_df = spark_df.withColumn("daily_pct_change", (col("Close") - col("Open")) / col("Open")).groupBy("Date").agg(avg("daily_pct_change").alias("Avg_Daily_Pct_Change")).orderBy("Date")
    avg_change_pandas = avg_change_df.toPandas()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(avg_change_pandas)
    avg_change_pandas.set_index("Date", inplace=True)
    with col2:
        st.line_chart(avg_change_pandas["Avg_Daily_Pct_Change"])

    # 3. What was the largest daily price increase (Close - Open) for each stock?
    st.subheader("Largest Daily Price Increase of {}".format(symbol))
    # Compute daily price increase
    stock_priceIncrease = spark_df.withColumn("daily_priceIncrease", (col("Close") - col("Open")))
    stock_priceIncrease_pandas = stock_priceIncrease.toPandas()
    stock_priceIncrease_pandas["date"] = pd.to_datetime(stock_priceIncrease_pandas["Date"], format="%Y")
    stock_priceIncrease_pandas = stock_priceIncrease_pandas[["date", "daily_priceIncrease"]]
    stock_priceIncrease_pandas.set_index("date", inplace=True)
    col1, col2 = st.columns(2)
    with col1:
      st.dataframe(stock_priceIncrease_pandas)
    with col2:
      max_index_inc = stock_priceIncrease_pandas["daily_priceIncrease"].idxmax()
      max_value_inc = stock_priceIncrease_pandas.loc[max_index_inc, "daily_priceIncrease"]
      st.write("Max daily price increase occurred on: ", max_index_inc.date())
      st.write("Max daily price increase value: ", max_value_inc)
      chart = st.line_chart(stock_priceIncrease_pandas["daily_priceIncrease"])

    # 4. What was the largest daily price decrease (Open - Close) for each stock?
    st.subheader("Largest Daily Price Decrease of {}".format(symbol))
    # Compute daily price decrease
    stock_priceDecrease = spark_df.withColumn("daily_priceDecrease", (col("Open") - col("Close")))
    stock_priceDecrease_pandas = stock_priceDecrease.toPandas()
    stock_priceDecrease_pandas["date"] = pd.to_datetime(stock_priceDecrease_pandas["Date"], format="%Y")
    stock_priceDecrease_pandas = stock_priceDecrease_pandas[["date", "daily_priceDecrease"]]
    stock_priceDecrease_pandas.set_index("date", inplace=True)
    col1, col2 = st.columns(2)
    with col1:
      st.dataframe(stock_priceDecrease_pandas)
    with col2:
      max_index_dec = stock_priceDecrease_pandas["daily_priceDecrease"].idxmin()
      max_value_dec = stock_priceDecrease_pandas.loc[max_index_dec, "daily_priceDecrease"]
      st.write("Max daily price decrease occurred on: ", max_index_dec.date())
      st.write("Max daily price decrease value: ", max_value_dec)
      st.line_chart(stock_priceDecrease_pandas["daily_priceDecrease"])
