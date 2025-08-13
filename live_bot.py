# live_bot.py
import os
import time
import joblib
import pandas as pd
from datetime import datetime, timezone
from alpaca_trade_api.rest import REST, TimeFrame
from features import make_features, required_bars
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Load environment variables
load_dotenv()

# Fetch credentials
API_KEY = os.getenv('APCA_API_KEY_ID')
API_SECRET = os.getenv('APCA_API_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

if not API_KEY or not API_SECRET:
    raise ValueError("Alpaca API credentials not found. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")

# Initialize Alpaca API
api = REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Global settings
MODEL = None
FEATURES = ['return_1', 'return_5', 'rsi', 'sma_ratio', 'vol_20']
POSITION_FRACTION = 0.1
PROB_OPEN = 0.55
PROB_CLOSE = 0.45
SLEEP_SECONDS = 60 * 5  # 5 minutes

# Dynamic lookback calculation
LOOKBACK_DAYS = required_bars(rsi_period=14, sma_short=10, sma_long=50, vol_window=20, buffer=10)

def fetch_recent(ticker, lookback_days=LOOKBACK_DAYS):
    """Fetch recent daily bars for a ticker, ensure enough history."""
    barset = api.get_bars(ticker, TimeFrame.Day, limit=lookback_days).df

    if 'symbol' in barset.columns:
        barset = barset[barset['symbol'] == ticker]

    if len(barset) < lookback_days:
        raise ValueError(
            f"Alpaca returned only {len(barset)} bars, but {lookback_days} are required. "
            "Try increasing lookback_days or check if the ticker has enough history."
        )

    df = barset[['open', 'high', 'low', 'close', 'volume']].reset_index()
    df.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Ticker'] = ticker
    return df

def run_live(ticker):
    """Main live trading loop."""
    global MODEL
    while True:
        try:
            print(datetime.now(timezone.utc), f"Fetching recent {LOOKBACK_DAYS} bars...")
            df = fetch_recent(ticker)

            df_feat = make_features(df)

            # Debug before dropna
            print(f"Features shape before dropna: {df_feat.shape}")
            print("NaN counts per column:\n", df_feat.isna().sum())
            print(df_feat.tail(5))

            df_feat.dropna(inplace=True)

            if df_feat.empty:
                print("No valid feature rows, skipping iteration.")
                time.sleep(SLEEP_SECONDS)
                continue

            cur_row = df_feat.iloc[-1]
            x = cur_row[FEATURES].values.reshape(1, -1)
            prob = MODEL.predict_proba(x)[0, 1]
            print("Predicted probability of price increase:", prob)

            # Account info
            account = api.get_account()
            equity = float(account.equity)
            position_size = equity * POSITION_FRACTION
            last_price = float(cur_row['Close'])
            qty = max(1, int(position_size / last_price))

            # Check existing position
            try:
                pos = api.get_position(ticker)
                has_position = True
            except:
                has_position = False

            # Trading decisions
            if prob > PROB_OPEN and not has_position:
                print(f"Placing BUY order for {qty} shares of {ticker}.")
                try:
                    api.submit_order(symbol=ticker, qty=qty, side='buy',
                                     type='market', time_in_force='day')
                except Exception as e:
                    print("Error placing BUY order:", e)

            elif prob < PROB_CLOSE and has_position:
                print(f"Closing position of {pos.qty} shares of {ticker}.")
                try:
                    api.submit_order(symbol=ticker, qty=pos.qty, side='sell',
                                     type='market', time_in_force='day')
                except Exception as e:
                    print("Error placing SELL order:", e)
            else:
                print("No trade this iteration.")

        except Exception as e:
            print("Error in live trading loop:", e)

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--model', default='model.pkl')
    args = parser.parse_args()

    MODEL = joblib.load(args.model)
    run_live(args.ticker)
