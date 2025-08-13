# features.py
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def required_bars(rsi_period=14, sma_short=10, sma_long=50, vol_window=20, buffer=10):
    """
    Calculate the number of historical bars needed for all features.
    Returns the max lookback window + buffer.
    """
    max_window = max(rsi_period, sma_short, sma_long, vol_window, 5)  # include 5 for return_5
    return max_window + buffer

def make_features(df, rsi_period=14, sma_short=10, sma_long=50, vol_window=20):
    df = df.copy()

    # Ensure numeric columns for those that exist
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing Close
    df.dropna(subset=['Close'], inplace=True)

    # Ensure datetime column
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    elif 'Date' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("No 'Datetime' or 'Date' column found in data.")

    # Ensure Ticker
    if 'Ticker' not in df.columns:
        df['Ticker'] = 'UNKNOWN'

    # Sort for rolling calculations
    df.sort_values(['Ticker', 'Datetime'], inplace=True)

    out = []
    for t, g in df.groupby('Ticker'):
        g = g.copy()
        g['return_1'] = g['Close'].pct_change(1)
        g['return_5'] = g['Close'].pct_change(5)

        # RSI
        rsi = RSIIndicator(close=g['Close'], window=rsi_period)
        g['rsi'] = rsi.rsi()

        # SMA ratios
        sma_s = SMAIndicator(close=g['Close'], window=sma_short).sma_indicator()
        sma_l = SMAIndicator(close=g['Close'], window=sma_long).sma_indicator()
        g['sma_ratio'] = sma_s / sma_l

        # Volume average
        g['vol_20'] = g['Volume'].rolling(vol_window).mean()

        # Target (binary: next day up)
        g['target'] = (g['Close'].shift(-1) > g['Close']).astype(int)

        out.append(g)

    return pd.concat(out, ignore_index=True)
