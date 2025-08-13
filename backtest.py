import argparse
import pandas as pd
import joblib
from backtesting import Backtest, Strategy
from features import make_features
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

FEATURES = ['return_1', 'return_5', 'rsi', 'sma_ratio', 'vol_20']

class MLStrategy(Strategy):
    model = None  # Will be injected

    def init(self):
        self.features = FEATURES

    def next(self):
        if len(self.data) < 50:
            return
        last_row = {f: self.data.df[f].iloc[-1] for f in self.features}
        X = pd.DataFrame([last_row])
        pred = self.model.predict(X)[0]
        if pred == 1 and not self.position:
            self.buy()
        elif pred == 0 and self.position:
            self.position.close()

def run_backtest(data_file, model_file, ticker, cash=10000):
    df = pd.read_csv(data_file)

    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Close'], inplace=True)

    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    elif 'Date' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("No 'Datetime' or 'Date' column found in data.")

    if 'Ticker' not in df.columns:
        df['Ticker'] = ticker

    df = df[df['Ticker'] == ticker].copy()
    df = make_features(df)
    df.dropna(inplace=True)

    df_bt = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'] + FEATURES].copy()
    df_bt.set_index('Datetime', inplace=True)

    # Load model and inject into strategy
    model = joblib.load(model_file)

    # --- SAFETY CHECK ---
    missing_features = [f for f in FEATURES if f not in df_bt.columns]
    if missing_features:
        raise ValueError(f"The following required features are missing from the data: {missing_features}")
    # Optional: Check if model was trained with all FEATURES
    try:
        model_features = model.get_booster().feature_names
        if sorted(FEATURES) != sorted(model_features):
            print(f"Warning: Model feature names differ from expected FEATURES.\nExpected: {FEATURES}\nModel: {model_features}")
    except AttributeError:
        # Not all models support get_booster(), ignore
        pass
    # --------------------

    MLStrategy.model = model

    bt = Backtest(df_bt, MLStrategy, cash=cash, commission=0.001, trade_on_close=False, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--ticker', required=True)
    args = p.parse_args()

    run_backtest(args.data, args.model, args.ticker)
