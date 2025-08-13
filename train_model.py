# train_model.py
# Usage: python train_model.py --data data.csv --model model.pkl

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
from xgboost import XGBClassifier
from features import make_features
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

FEATURES = ['return_1', 'return_5', 'rsi', 'sma_ratio', 'vol_20']

def prepare(df):
    df = make_features(df)
    df.dropna(inplace=True)
    X = df[FEATURES]
    y = df['target']
    return X, y, df

def train(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        m = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            max_depth=4
        )
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        scores.append(roc_auc_score(yte, p))
    print("CV AUC scores:", scores)

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=200,
        max_depth=4
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--model', default='model.pkl')
    args = p.parse_args()

    df = pd.read_csv(args.data)

    # Ensure numeric price/volume columns
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['Close'], inplace=True)

    # Handle datetime
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
    elif 'Date' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'])
    else:
        raise ValueError("No 'Datetime' or 'Date' column found in data.")

    if 'Ticker' not in df.columns:
        df['Ticker'] = 'UNKNOWN'

    X, y, _ = prepare(df)
    model = train(X, y)
    joblib.dump(model, args.model)
    print("Model saved to", args.model)
