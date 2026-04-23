import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta

# Top 10 companies by market cap (as of recent years)
tickers = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet (Google)
    "AMZN",  # Amazon
    "META",  # Meta (Facebook)
    "TSLA",  # Tesla
    "NVDA",  # Nvidia
    "JPM",   # JPMorgan Chase
    "V",     # Visa
    "UNH"    # UnitedHealth
]

start_date = '2015-01-01'
end_date = '2025-05-20'

# Download OHLCV data for all tickers
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)

# Prepare empty dictionary to hold dataframes per ticker
indicator_dfs = {}

for ticker in tickers:
    # Extract OHLCV
    df = data[ticker].copy()
    
    # Compute indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD_diff'] = ta.trend.MACD(df['Close']).macd_diff()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_upper'] = bollinger.bollinger_hband()
    df['BB_lower'] = bollinger.bollinger_lband()
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    
    # Keep only relevant columns and rename to include ticker for clarity
    cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD_diff', 'ATR', 'BB_upper', 'BB_lower', 'SMA_20', 'EMA_20']
    df = df[cols_to_keep]
    df.columns = [f"{ticker}_{col}" for col in df.columns]
    
    indicator_dfs[ticker] = df

# Combine all tickers' data into one DataFrame aligned by date
combined_df = pd.concat(indicator_dfs.values(), axis=1).dropna()

# Save to CSV if needed
combined_df.to_csv("stock_data_with_indicators.csv")

print(combined_df.head())
