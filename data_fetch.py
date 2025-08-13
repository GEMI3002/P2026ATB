import argparse
import pandas as pd
import os
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--tickers", required=True, help="Comma-separated list of tickers")
parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
parser.add_argument("--outfile_prefix", required=True, help="Prefix for output CSV files")
parser.add_argument("--batch_size", type=int, default=50, help="Number of tickers per CSV file")
parser.add_argument("--final_outfile", required=True, help="Filename for combined CSV")
args = parser.parse_args()

tickers = args.tickers.split(",")
start_date = args.start
end_date = args.end
outfile_prefix = args.outfile_prefix
batch_size = args.batch_size
final_outfile = args.final_outfile

# Alpaca API credentials
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not API_SECRET:
    raise ValueError("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")

api = REST(API_KEY, API_SECRET, base_url=BASE_URL)

# Rate limiter
rate_lock = Lock()
last_request_time = 0
current_rate_per_sec = 5

def rate_limited_request():
    global last_request_time
    with rate_lock:
        now = time.time()
        elapsed = now - last_request_time
        min_interval = 1 / current_rate_per_sec
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_request_time = time.time()

def fetch_bars_adaptive(ticker, start, end, max_retries=5):
    global current_rate_per_sec
    attempt = 0
    wait = 1
    while attempt < max_retries:
        try:
            rate_limited_request()
            bars = api.get_bars(ticker, TimeFrame.Day, start=start.isoformat(), end=end.isoformat()).df
            current_rate_per_sec = min(current_rate_per_sec * 1.05, 20)
            return bars
        except Exception as e:
            attempt += 1
            current_rate_per_sec = max(current_rate_per_sec * 0.5, 1)
            wait *= 2
            print(f"Warning: {ticker} fetch failed (attempt {attempt}/{max_retries}). Retrying in {wait}s. Rate now: {current_rate_per_sec:.2f}/s")
            time.sleep(wait)
    print(f"Error: {ticker} failed after {max_retries} attempts")
    return pd.DataFrame()

def fetch_ticker_data(ticker, start, end, global_pbar):
    current_start = datetime.fromisoformat(start)
    final_end = datetime.fromisoformat(end)
    frames = []
    ticker_pbar = tqdm(total=(final_end - current_start).days + 1, desc=f"{ticker}", unit="day", leave=False)
    while current_start < final_end:
        chunk_end = min(current_start + timedelta(days=365*2), final_end)
        bars = fetch_bars_adaptive(ticker, current_start, chunk_end)
        if not bars.empty:
            bars = bars.reset_index()
            bars.rename(columns={'timestamp': 'Datetime'}, inplace=True)
            bars['Ticker'] = ticker
            frames.append(bars)
        days_fetched = (chunk_end - current_start).days + 1
        ticker_pbar.update(days_fetched)
        global_pbar.update(days_fetched)
        current_start = chunk_end + timedelta(days=1)
    ticker_pbar.close()
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def fetch_batch(ticker_batch, start, end, batch_index):
    all_data = []
    total_days_batch = sum((datetime.fromisoformat(end) - datetime.fromisoformat(start)).days + 1 for _ in ticker_batch)
    with tqdm(total=total_days_batch, desc=f"Batch {batch_index+1}", unit="day") as global_pbar:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_ticker_data, t, start, end, global_pbar): t for t in ticker_batch}
