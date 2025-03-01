import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_ai_recommendations(predicted_prices):
    try:
        # Create a prompt for Gemini
        prompt = f"""
        The following are predicted stock prices for the next {len(predicted_prices)} days:
        {predicted_prices}

        Based on these predictions, provide actionable recommendations (e.g., buy, sell, or hold) and a brief explanation.
        """

        # Use Gemini to generate recommendations
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Failed to generate recommendations: {str(e)}"

# Function to fetch stock data
def fetch_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# Preprocess data for LSTM
def preprocess_data(data, look_back):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

# Build LSTM model
def build_lstm_model(input_shape, lstm_units, dense_units, dropout_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot predictions vs actuals
def plot_predictions(actual, predicted, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted Prices'))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')
    return fig



# Streamlit App
def main():
    st.title("Stock Price Prediction using LSTM")
    
    # Sidebar for user input
    st.sidebar.header("User Input")
    ticker = st.sidebar.text_input("Stock Ticker", value="")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    look_back = st.sidebar.slider("Look-back Period", min_value=1, max_value=100, value=60)
    predict_days = st.sidebar.slider("Prediction Days", min_value=1, max_value=30, value=7)

    # Hyperparameter tuning options
    st.sidebar.header("Model Hyperparameters")
    lstm_units = st.sidebar.slider("LSTM Units", min_value=10, max_value=200, value=50, step=10)
    dense_units = st.sidebar.slider("Dense Units", min_value=1, max_value=100, value=25, step=5)
    dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)

    # Start button to prevent automatic execution
    if st.sidebar.button("Run Prediction"):
        if ticker:
            try:
                company_info = yf.Ticker(ticker).info
                st.subheader(f"Company Description: {company_info.get('longName', 'N/A')}")
                st.write(company_info.get('longBusinessSummary', 'No description available.'))
                st.write("**Sector:**", company_info.get('sector', 'N/A'))
                st.write("**Industry:**", company_info.get('industry', 'N/A'))
                st.write("**Website:**", company_info.get('website', 'N/A'))
            except Exception as e:
                st.error(f"Failed to fetch company description: {str(e)}")
        if ticker:
            try:
                data = fetch_stock_data(ticker, start_date, end_date)
                if data.empty:
                    st.error("No data found for the given ticker and date range!")
                    return
                
                # Ensure column names are flattened (remove MultiIndex if present)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)  # Flatten to single-level columns

                st.header("Visualization Dashboard")
                st.write(f"Latest Stock Data for {ticker}")
                st.write(data.tail())

                # Visualization Dashboard
                st.write("Opening Price Over Time")
                fig1 = px.area(data, x=data.index, y='Open', title="Opening Price Over Time")
                st.plotly_chart(fig1)

                st.write("Closing Price Over Time")
                fig2 = px.line(data, x=data.index, y='Close', title="Closing Price Over Time")
                st.plotly_chart(fig2)

                st.write("Trading Volume Over Time")
                fig3 = px.bar(data, x=data.index, y='Volume', title="Trading Volume Over Time")
                st.plotly_chart(fig3)

                # Moving Averages
                st.write("Moving Averages (50-day & 200-day)")
                data['50MA'] = data['Close'].rolling(window=50).mean()
                data['200MA'] = data['Close'].rolling(window=200).mean()
                fig4 = px.line(data, x=data.index, y=['Close', '50MA', '200MA'], title="Moving Averages (50-day & 200-day)")
                st.plotly_chart(fig4)

                # Daily Returns
                st.write("Daily Returns Over Time")
                data['Daily Returns'] = data['Close'].pct_change()
                fig5 = px.line(data, x=data.index, y='Daily Returns', title="Daily Returns Over Time")
                st.plotly_chart(fig5)

                # Bollinger Bands
                st.write("Bollinger Bands (20-day)")
                data['20SMA'] = data['Close'].rolling(window=20).mean()
                data['Upper Band'] = data['20SMA'] + (data['Close'].rolling(window=20).std() * 2)
                data['Lower Band'] = data['20SMA'] - (data['Close'].rolling(window=20).std() * 2)
                fig6 = px.line(data, x=data.index, y=['Close', '20SMA', 'Upper Band', 'Lower Band'], title="Bollinger Bands (20-day)")
                st.plotly_chart(fig6)

                # Candlestick Chart
                st.write("Candlestick Chart")
                fig7 = go.Figure(data=[go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Candlestick'
                )])
                fig7.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig7)

                # Histogram of Daily Returns
                st.write("Histogram of Daily Returns")
                fig8 = px.histogram(data, x='Daily Returns', nbins=50, title="Distribution of Daily Returns")
                st.plotly_chart(fig8)

                # Cumulative Returns
                st.write("Cumulative Returns Over Time")
                data['Cumulative Returns'] = (1 + data['Daily Returns']).cumprod()
                fig9 = px.line(data, x=data.index, y='Cumulative Returns', title="Cumulative Returns Over Time")
                st.plotly_chart(fig9)

                # Key Statistics
                st.write("Key Statistics")
                stats = {
                    "Maximum Opening Price":data['Open'].max(),
                    "Minimum Opening Price":data['Open'].min(),
                    "Average Opening Price":data['Open'].mean(),
                    "Maximum Close Price": data['Close'].max(),
                    "Minimum Close Price": data['Close'].min(),
                    "Average Close Price": data['Close'].mean(),
                    "Maximum Volume": data['Volume'].max(),
                    "Minimum Volume": data['Volume'].min(),
                    "Mean Volume":data['Volume'].mean()
                }
                st.write(stats)

                pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

                with pricing_data:
                    st.header('Price Movements')
                    data2 = data
                    data2['% Change'] = data['Close'] / data['Close'].shift(1) - 1
                    data2.dropna(inplace=True)
                    st.write(data2)
                    annual_return = data2['% Change'].mean() * 252 * 100
                    st.write('Annual Return is ', annual_return, '%')
                    stdev = np.std(data2['% Change']) * np.sqrt(252)
                    st.write('Standard Deviation is ', stdev * 100, '%')
                    st.write('Risk Adj. Return is ', annual_return / (stdev * 100))

                with fundamental_data:
                    key = '701G7G76FIB6IA3M'
                    fd = FundamentalData(key, output_format='pandas')

                    st.subheader('Balance Sheet')
                    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
                    
                    if not balance_sheet.empty and len(balance_sheet.T) > 2:
                        bs = balance_sheet.T.iloc[2:]
                        bs.columns = balance_sheet.T.iloc[0].values
                        st.write(bs)
                    else:
                        st.error("Balance Sheet data unavailable.")

                    st.subheader('Income Statement')
                    income_statement = fd.get_income_statement_annual(ticker)[0]
                    
                    if not income_statement.empty and len(income_statement.T) > 2:
                        is1 = income_statement.T.iloc[2:]
                        is1.columns = income_statement.T.iloc[0].values
                        st.write(is1)
                    else:
                        st.error("Income Statement data unavailable.")

                    st.subheader('Cash Flow Statement')
                    cash_flow = fd.get_cash_flow_annual(ticker)[0]
                    
                    if not cash_flow.empty and len(cash_flow.T) > 2:
                        cf = cash_flow.T.iloc[2:]
                        cf.columns = cash_flow.T.iloc[0].values
                        st.write(cf)
                    else:
                        st.error("Cash Flow Statement data unavailable.")

                with news:
                    st.header(f'News of {ticker}')
                    sn = StockNews(ticker, save_news=False)
                    df_news = sn.read_rss()
                    for i in range(10):
                        st.subheader(f'News {i + 1}')
                        st.write(df_news['published'][i])
                        st.write(df_news['title'][i])
                        st.write(df_news['summary'][i])
                        title_sentiment = df_news['sentiment_title'][i]
                        st.write(f'Title Sentiment {title_sentiment}')
                        news_sentiment = df_news['sentiment_summary'][i]
                        st.write(f'News Sentiment {news_sentiment}')

                # Preprocess and prepare LSTM inputs
                st.header("Preparing Data")
                if len(data) > look_back:
                    data_close = data['Close'].values.reshape(-1, 1)
                    X, y, scaler = preprocess_data(data_close, look_back)
                    X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM input shape

                    # Train-test split
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    # Build and train the LSTM model
                    st.header("Training LSTM Model")
                    model = build_lstm_model((X_train.shape[1], 1), lstm_units, dense_units, dropout_rate)
                    with st.spinner("Training the model..."):
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                    # Test predictions
                    st.header("Evaluating Model")
                    y_pred = model.predict(X_test)
                    y_pred_rescaled = scaler.inverse_transform(y_pred)
                    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

                    # Display predictions
                    st.plotly_chart(plot_predictions(y_test_rescaled.flatten(), y_pred_rescaled.flatten(),
                                                     title="Actual vs Predicted Prices"))

                    # Future predictions
                    st.header("Future Predictions")
                    last_sequence = X_test[-1]  # Last sequence from the test set
                    future_preds = []
                    for _ in range(predict_days):
                        # Predict the next value
                        next_pred = model.predict(last_sequence.reshape(1, look_back, 1))[0]
                        # Append the prediction to the sequence
                        next_pred_reshaped = next_pred.reshape(1, 1)  # Reshape to (1, 1)
                        last_sequence = np.concatenate((last_sequence[1:], next_pred_reshaped), axis=0)  # Maintain shape (look_back, 1)
                        future_preds.append(next_pred_reshaped)

                    # Rescale the predictions back to the original scale
                    future_preds_rescaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

                    # Display predictions
                    st.write(f"Predicted Prices for Next {predict_days} Days:")
                    future_dates = [end_date + pd.Timedelta(days=i) for i in range(1, predict_days + 1)]
                    future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_preds_rescaled.flatten()})
                    st.write(future_data)

                    st.header("AI Recommendations")
                    recommendations = get_ai_recommendations(future_preds_rescaled.flatten())
                    st.write(recommendations)

                    # Download predictions as CSV
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=future_data.to_csv(index=False),
                        file_name=f"{ticker}_predictions.csv",
                        mime='text/csv'
                    )
                else:
                    st.error("Not enough data for the selected look-back period!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter a valid stock ticker!")

if __name__ == "__main__":
    main()


