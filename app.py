import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import matplotlib.pyplot as plt

# App title
st.title("Stock Market Analysis App")
st.write("Analyze stock prices and market sentiment using AI.")

# Function to fetch stock data
@st.cache_data
def get_stock_data(stock, start, end):
    try:
        data = yf.download(stock, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Function to fetch news and analyze sentiment
def fetch_news_and_sentiment(stock):
    api_key = "b4a96f2fcd004c0c8c04c1038b9bb666"  # Replace with a valid News API key
    url = f"https://newsapi.org/v2/everything?q={stock}&sortBy=publishedAt&apiKey={api_key}"
    try:
        response = requests.get(url).json()
        if 'articles' in response:
            headlines = [article['title'] for article in response['articles'][:10]]
            analyzer = SentimentIntensityAnalyzer()
            sentiment_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
            average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            return headlines, average_sentiment
        else:
            return [], 0
    except Exception as e:
        st.error(f"Error fetching news data: {e}")
        return [], 0

# Sidebar inputs
st.sidebar.header("User Input")
stock = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))

# Load stock data and perform analysis
if st.sidebar.button("Analyze Stock"):
    data = get_stock_data(stock, start_date, end_date)
    if not data.empty:
        st.write(f"## {stock} Stock Data")
        st.line_chart(data['Close'])

        # Sentiment Analysis
        st.write("### Sentiment Analysis")
        headlines, sentiment_score = fetch_news_and_sentiment(stock)
        if headlines:
            st.write("Recent News Headlines:")
            for i, headline in enumerate(headlines):
                st.write(f"{i + 1}. {headline}")
            st.write(f"Average Sentiment Score: {sentiment_score}")
        else:
            st.write("No news headlines found for sentiment analysis.")

        # Feature scaling for prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        # Prepare data for LSTM
        time_steps = 60
        X, y = [], []
        for i in range(len(scaled_data) - time_steps):
            X.append(scaled_data[i:i + time_steps, 0])
            y.append(scaled_data[i + time_steps, 0])
        X = np.array(X).reshape(-1, time_steps, 1)
        y = np.array(y)
        
        # Build or load LSTM model
        try:
            model = load_model('model.h5')
        except:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=5, batch_size=32, verbose=0)
            model.save('model.h5')

        # Predict future prices
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        actual_prices = data['Close'].values[-len(predictions):]
        
        # Display predictions
#  data = {
#         "Actual": actual_prices,
#         "Predicted": predictions
#     }
        data ={
            "Actual": actual_prices.flatten(),
            "Predicted": predictions.flatten()
        }
        comparison_df = pd.DataFrame(data)
        st.write("### Predicted Stock Prices vs Actual")
        st.line_chart(comparison_df)
    else: st.error("No stock data found for the selected ticker and date range.")

