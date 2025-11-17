"""
All-in-One Forex/CFD/Crypto Signal Bot (Streamlit UI)
Supports data from OANDA (FX, XAU) and Binance (crypto like EUR/USDT, BTC/USDT) if python-binance is installed.

Files: single runnable app. Put your .env with OANDA_TOKEN, OANDA_ACCOUNT, BINANCE_API_KEY, BINANCE_API_SECRET
Usage: streamlit run all_in_one_streamlit_bot.py

Note: This is a demo scaffold — always test in demo accounts and backtest before using real money.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import time
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ML / indicators
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange

# Optional libraries
try:
    import oandapyV20
    import oandapyV20.endpoints.instruments as instruments
    OANDA_AVAILABLE = True
except Exception:
    OANDA_AVAILABLE = False

try:
    from binance import Client as BinanceClient
    BINANCE_AVAILABLE = True
except Exception:
    BINANCE_AVAILABLE = False

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except Exception:
    VBT_AVAILABLE = False

# Credentials
OANDA_TOKEN = os.getenv("OANDA_TOKEN")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT")
BINANCE_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET = os.getenv("BINANCE_API_SECRET")

# --------------------- Utility: fetch candles ---------------------
@st.cache_data(ttl=30)
def fetch_oanda_candles(pair="EUR_USD", timeframe="M5", count=500):
    if not OANDA_AVAILABLE or not OANDA_TOKEN:
        raise RuntimeError("OANDA not available or OANDA_TOKEN missing in .env")
    api = oandapyV20.API(access_token=OANDA_TOKEN, environment="practice")
    params = {"granularity": timeframe, "count": count, "price": "M"}
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    api.request(r)
    rows = []
    for c in r.response.get("candles", []):
        if not c.get("complete", False):
            continue
        t = pd.to_datetime(c["time"]) 
        o = float(c["mid"]["o"]) 
        h = float(c["mid"]["h"]) 
        l = float(c["mid"]["l"]) 
        cc = float(c["mid"]["c"]) 
        rows.append([t, o, h, l, cc])
    df = pd.DataFrame(rows, columns=["time","open","high","low","close"]).set_index("time")
    return df

@st.cache_data(ttl=30)
def fetch_binance_candles(symbol="BTCUSDT", interval='5m', limit=500):
    if not BINANCE_AVAILABLE or not BINANCE_KEY:
        raise RuntimeError("Binance client not available or keys missing in .env")
    client = BinanceClient(BINANCE_KEY, BINANCE_SECRET)
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    rows = []
    for k in klines:
        t = pd.to_datetime(k[0], unit='ms')
        o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
        rows.append([t,o,h,l,c])
    df = pd.DataFrame(rows, columns=["time","open","high","low","close"]).set_index("time")
    return df

# --------------------- Feature engineering ---------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    macd = MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
    df['macd'] = macd.macd(); df['macd_signal'] = macd.macd_signal(); df['macd_hist'] = macd.macd_diff()
    bb = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband(); df['bb_low'] = bb.bollinger_lband(); df['bb_width'] = df['bb_high'] - df['bb_low']
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()
    df['return_1'] = df['close'].pct_change(1); df['return_5'] = df['close'].pct_change(5); df['return_15'] = df['close'].pct_change(15)
    df['future_close'] = df['close'].shift(-1)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

# --------------------- Model training ---------------------
def train_lightgbm(df_feat: pd.DataFrame):
    features = [
        'rsi','macd','macd_signal','macd_hist','bb_high','bb_low','bb_width','atr','return_1','return_5','return_15'
    ]
    X = df_feat[features]; y = df_feat['target']
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    accs = []
    for tr, val in tscv.split(X):
        model = lgb.LGBMClassifier(objective='binary', n_estimators=300)
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[val])
        acc = accuracy_score(y.iloc[val], preds)
        accs.append(acc)
        models.append(model)
    return models, np.mean(accs)

# --------------------- Predict / Signal ---------------------
def predict_signal(models, df_feat):
    features = ['rsi','macd','macd_signal','macd_hist','bb_high','bb_low','bb_width','atr','return_1','return_5','return_15']
    X = df_feat[features]
    probs = np.mean([m.predict_proba(X)[:,1] for m in models], axis=0)
    df_feat['pred_prob'] = probs
    last_p = probs[-1]
    if last_p > 0.6:
        sig = 'BUY (ALTA)'
    elif last_p < 0.4:
        sig = 'SELL (BAIXA)'
    else:
        sig = 'NEUTRO'
    return last_p, sig, df_feat

# --------------------- Backtest (optional) ---------------------
def backtest_vectorbt(df_feat):
    if not VBT_AVAILABLE:
        raise RuntimeError('vectorbt not installed')
    close = df_feat['close']
    entries = df_feat['pred_prob'] > 0.6
    exits = df_feat['pred_prob'] < 0.4
    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=1000, fees=0.0004, slippage=0.0002)
    return pf

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title='Bot Forex/CFD/Crypto — All-in-One', layout='wide')
st.title('Bot Forex/CFD/Crypto — All-in-One (demo)')

col1, col2 = st.columns([2,1])
with col1:
    st.markdown('### Seleção de instrumento e timeframe')
    instrument = st.selectbox('Instrumento', ['EUR_USD','XAU_USD','GBP_USD','USD_JPY','BTCUSDT','ETHUSDT'])
    timeframe = st.selectbox('Timeframe', ['M1','M5','M15','H1'])
    count = st.slider('Quantidade de candles', min_value=200, max_value=3000, value=800, step=100)

with col2:
    st.markdown('### Credenciais')
    st.write(f'OANDA: {"OK" if OANDA_TOKEN else "MISSING
