# streamlit_app.py
# Full Stonki.ai demo app: Rolling LSTM backtests + indicators + Gemini LLM recommender + human feedback
# WARNING: This is demo/research code. Do NOT run live money until full validation & risk controls are added.

import os
import time
import json
import sqlite3
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib

# TensorFlow / Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# technical indicators
import ta

# Gemini (Google) generative API (example)
# You told me to use gemini-2.5-pro and provided an API key. In production, DO NOT hardcode keys.
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="Stonki.ai — Rolling LSTM + Indicators + Gemini LLM", layout="wide")
st.title("📈 Stonki.ai — Rolling LSTM Backtests + Indicators + LLM (Human-in-the-loop)")

# -----------------------
# Sidebar: Interactive param sliders & toggles
# -----------------------
st.sidebar.header("Ticker & Date Range")
TICKER = st.sidebar.text_input("Ticker", value="AAPL").upper()
START = st.sidebar.date_input("Start date", datetime.now().date() - timedelta(days=365*2))
END = st.sidebar.date_input("End date", datetime.now().date())

st.sidebar.header("Indicator parameters")
SMA_SHORT = st.sidebar.slider("SMA short window", 5, 50, 20, 1)
SMA_LONG = st.sidebar.slider("SMA long window", 10, 200, 50, 1)
RSI_WINDOW = st.sidebar.slider("RSI window", 5, 30, 14, 1)
BB_WINDOW = st.sidebar.slider("BollingerBands window", 10, 30, 20, 1)
ATR_WINDOW = st.sidebar.slider("ATR window", 7, 30, 14, 1)
MACD_FAST = st.sidebar.number_input("MACD fast", value=12, min_value=3, max_value=50)
MACD_SLOW = st.sidebar.number_input("MACD slow", value=26, min_value=5, max_value=100)
MACD_SIGNAL = st.sidebar.number_input("MACD signal", value=9, min_value=3, max_value=50)

st.sidebar.header("LSTM / Rolling backtest parameters")
LSTM_WINDOW = st.sidebar.slider("LSTM train window (days)", 30, 500, 120, 10)
RETRAIN_FREQ = st.sidebar.slider("Retrain every N days (rolling)", 1, 10, 1, 1)
LSTM_EPOCHS = st.sidebar.slider("LSTM epochs (per retrain)", 1, 50, 6, 1)
LSTM_BATCH = st.sidebar.selectbox("LSTM batch size", options=[8,16,32,64,128], index=2)
FORECAST_HORIZON = st.sidebar.slider("Forecast horizon (days, iterative)", 1, 14, 1, 1)

st.sidebar.header("Backtest & Strategy")
CAPITAL = st.sidebar.number_input("Backtest capital ($)", min_value=100, value=10000, step=100)
POLICY = st.sidebar.selectbox("Backtest policy", ["SMA only", "LSTM only", "Combined (SMA+LSTM)"])
ALLOW_FRACTIONAL = st.sidebar.checkbox("Allow fractional shares in simulation", value=True)

st.sidebar.header("Chart toggles")
SHOW_SMA = st.sidebar.checkbox("Show SMA lines", value=True)
SHOW_RSI = st.sidebar.checkbox("Show RSI", value=False)
SHOW_MACD = st.sidebar.checkbox("Show MACD", value=False)
SHOW_BB = st.sidebar.checkbox("Show Bollinger Bands", value=False)
SHOW_ATR = st.sidebar.checkbox("Show ATR", value=False)
SHOW_VOLUME = st.sidebar.checkbox("Show Volume", value=False)
SHOW_LSTM_FORECAST = st.sidebar.checkbox("Show LSTM forecast", value=True)

st.sidebar.header("LLM & News")
# NOTE: For security, set these in environment or Streamlit secrets. I include the key variable only because you provided it.
# In production remove hard-coded keys and use environment variables/Streamlit secrets.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAiaswXxN3ngfEwMRXckBmEoZHO151jRv0"  # <-- Remove this hardcode in production
NEWS_API_KEY = os.getenv("NEWS_API_KEY") or None

LLM_ENABLED = st.sidebar.checkbox("Enable Gemini LLM calls", value=bool(GEMINI_AVAILABLE))
NEWS_ENABLED = st.sidebar.checkbox("Fetch news (NewsAPI)", value=False)

st.sidebar.markdown("**Run**")
RUN_BTN = st.sidebar.button("Run full analysis (can be slow)")

# -----------------------
# Feedback DB & preference model
# -----------------------
DB_PATH = "stonki_feedback.db"
PREF_MODEL_PATH = "preference_model.joblib"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        ts TEXT,
        symbol TEXT,
        date TEXT,
        price REAL,
        rsi REAL,
        macd REAL,
        bb_high REAL,
        bb_low REAL,
        sma_short REAL,
        sma_long REAL,
        lstm_confidence REAL,
        llm_rec TEXT,
        llm_confidence REAL,
        user_label INTEGER
    )
    """)
    conn.commit()
    conn.close()

def insert_feedback(row):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO feedback (ts,symbol,date,price,rsi,macd,bb_high,bb_low,sma_short,sma_long,lstm_confidence,llm_rec,llm_confidence,user_label)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (row['ts'], row['symbol'], row['date'], row['price'], row['rsi'], row['macd'], row['bb_high'], row['bb_low'], row['sma_short'], row['sma_long'], row['lstm_confidence'], row['llm_rec'], row['llm_confidence'], row['user_label']))
    conn.commit()
    conn.close()

def load_feedback_df(limit=500):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def train_preference_model():
    df = load_feedback_df(10000)
    if df.empty or df.shape[0] < 30:
        return None
    X = df[['rsi','macd','bb_high','bb_low','sma_short','sma_long','lstm_confidence']].fillna(0)
    y = df['user_label']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    joblib.dump(model, PREF_MODEL_PATH)
    return model

def load_preference_model():
    if os.path.exists(PREF_MODEL_PATH):
        return joblib.load(PREF_MODEL_PATH)
    return None

# -----------------------
# Data fetch & indicators
# -----------------------
@st.cache_data(ttl=3600)
def fetch_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df.index = pd.to_datetime(df.index)
    return df

def add_technical_indicators(df):
    df = df.copy()
    # SMA
    df[f"SMA_{SMA_SHORT}"] = df['Close'].rolling(SMA_SHORT).mean()
    df[f"SMA_{SMA_LONG}"] = df['Close'].rolling(SMA_LONG).mean()
    # RSI
    df['rsi'] = ta.momentum.rsi(df['Close'], window=RSI_WINDOW)
    # MACD
    macd = ta.trend.MACD(df['Close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'], window=BB_WINDOW, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # ATR
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=ATR_WINDOW)
    # Volume MA
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    return df

# -----------------------
# LSTM helpers for rolling retrain
# -----------------------
def build_seq(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)

def make_lstm_model(window):
    model = Sequential([
        LSTM(64, input_shape=(window,1), return_sequences=False),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_once(series, window, epochs, batch):
    # series = numpy 1D array of close prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1,1)).flatten()
    X, y = build_seq(scaled, window)
    if len(X) < 10:
        return None, None, None
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = make_lstm_model(window)
    es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch, verbose=0, callbacks=[es])
    return model, scaler, window

def predict_next_day(model, scaler, last_window):
    scaled = scaler.transform(last_window.reshape(-1,1)).flatten()
    x = scaled.reshape((1, scaled.shape[0], 1))
    pred_scaled = model.predict(x, verbose=0)[0,0]
    pred = scaler.inverse_transform(np.array([[pred_scaled]]))[0,0]
    return pred, pred_scaled

# -----------------------
# Rolling LSTM backtest (realistic but slow)
# -----------------------
def rolling_lstm_signals(df, window, retrain_every, epochs, batch):
    prices = df['Close'].values
    n = len(prices)
    preds = np.full(n, np.nan)
    pred_scaled_arr = np.full(n, np.nan)
    signals = np.zeros(n, dtype=int)
    warmup = window + 5
    model = None
    used_window = window
    for t in range(warmup, n):
        train_end = t - 1
        train_start = max(0, train_end - window)
        # retrain on schedule (first train when we hit the warmup)
        if (t - warmup) % retrain_every == 0 or model is None:
            try:
                model, scaler, used_window = train_lstm_once(prices[train_start:train_end+1], window=min(window, train_end-train_start+1), epochs=epochs, batch=batch)
            except Exception:
                model = None
        if model is not None:
            last_window = prices[train_end - used_window + 1: train_end+1]
            if len(last_window) == used_window:
                pred, pred_scaled = predict_next_day(model, scaler, last_window)
                preds[t] = pred
                pred_scaled_arr[t] = pred_scaled
                prev_price = prices[train_end]
                signals[t] = 1 if pred > prev_price else -1 if pred < prev_price else 0
    df_out = df.copy()
    df_out['lstm_pred'] = preds
    df_out['lstm_signal'] = signals
    df_out['lstm_confidence'] = np.abs(pred_scaled_arr)
    return df_out

# -----------------------
# SMA signals & backtest engine
# -----------------------
def sma_signals(df):
    df = df.copy()
    df['sma_signal'] = 0
    df.loc[df[f"SMA_{SMA_SHORT}"] > df[f"SMA_{SMA_LONG}"], 'sma_signal'] = 1
    df.loc[df[f"SMA_{SMA_SHORT}"] < df[f"SMA_{SMA_LONG}"], 'sma_signal'] = -1
    df['sma_change'] = df['sma_signal'].diff().fillna(0)
    return df

def backtest(df, policy, capital):
    cash = capital
    position = 0.0
    holding = False
    trades = []
    portfolio = []
    for idx, row in df.iterrows():
        price = row['Close']
        sma_sig = int(row.get('sma_signal', 0))
        lstm_sig = int(row.get('lstm_signal', 0))
        if policy == "SMA only":
            sig = sma_sig
        elif policy == "LSTM only":
            sig = lstm_sig
        else:
            sig = sma_sig if sma_sig == lstm_sig else 0
        if sig == 1 and not holding:
            # buy with all cash
            if ALLOW_FRACTIONAL:
                position = cash / price
                cash = 0.0
            else:
                # buy integer shares
                shares = int(cash // price)
                position = shares
                cash -= shares * price
            holding = True
            trades.append({'date': idx, 'type': 'BUY', 'price': price, 'shares': position})
        elif sig == -1 and holding:
            cash += position * price
            trades.append({'date': idx, 'type': 'SELL', 'price': price, 'shares': position})
            position = 0.0
            holding = False
        pv = cash + position * price
        portfolio.append({'date': idx, 'pv': pv})
    final = cash + position * df['Close'].iloc[-1]
    profit = final - capital
    pv_df = pd.DataFrame(portfolio).set_index('date')
    trades_df = pd.DataFrame(trades)
    return trades_df, pv_df, final, profit

# -----------------------
# News retrieval & LLM wrapper (Gemini)
# -----------------------
def fetch_news_snippets(symbol, days=3, max_articles=3):
    if not NEWS_ENABLED or not NEWS_API_KEY:
        return []
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
            "pageSize": max_articles,
            "sortBy": "relevancy",
            "language": "en",
            "apiKey": NEWS_API_KEY
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        items = r.json().get("articles", [])
        snippets = []
        for it in items:
            snippets.append({"title": it.get("title"), "source": it.get("source", {}).get("name"), "excerpt": (it.get("description") or "")[:240]})
        return snippets
    except Exception:
        return []

def gemini_llm_recommendation(evidence):
    # If no gemini library available or disabled, fallback to deterministic rule
    if not LLM_ENABLED or not GEMINI_AVAILABLE:
        # fallback deterministic rule
        rec = "HOLD"
        conf = 50
        if evidence.get('sma_short') and evidence.get('sma_long') and evidence.get('rsi') is not None:
            if evidence['sma_short'] > evidence['sma_long'] and evidence['rsi'] < 70:
                rec, conf = "BUY", 65
            elif evidence['sma_short'] < evidence['sma_long'] and evidence['rsi'] > 30:
                rec, conf = "SELL", 55
            else:
                rec, conf = "HOLD", 50
        return {"recommendation": rec, "confidence": conf, "rationale": "Fallback deterministic rule (Gemini disabled or not available)", "evidence_used": ["sma_short","sma_long","rsi"]}
    # Use gemini-2.5-pro via google.generativeai wrapper (example)
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Compose prompt carefully; pass structured evidence
        prompt = f"""
You are a concise quantitative trading assistant. Using the evidence JSON below, return a JSON object with keys:
recommendation (BUY/SELL/HOLD), confidence (0-100), rationale (2-4 short sentences), evidence_used (list of keys used).

Evidence:
{json.dumps(evidence, default=str, indent=2)}
"""
        # The API surface may differ; below is illustrative and may need adapting to the installed genai package version.
        model = genai.get_model("gemini-2.5-pro")
        response = model.generate(input=prompt, temperature=0.0, max_output_tokens=400)
        text = response.result[0].content[0].text if hasattr(response, 'result') else getattr(response, 'text', None)
        if not text:
            # fallback
            return {"recommendation":"HOLD","confidence":40,"rationale":"No text returned from Gemini","evidence_used":[]}
        # try parse JSON from text
        try:
            out = json.loads(text)
        except Exception:
            # extract JSON block
            s = text.find("{")
            e = text.rfind("}") + 1
            out = json.loads(text[s:e])
        return out
    except Exception as e:
        # return fallback if error
        return {"recommendation":"HOLD","confidence":40,"rationale":f"Gemini call failed: {e}", "evidence_used": []}

# -----------------------
# Main UI flow
# -----------------------
init_db()

if RUN_BTN:
    with st.spinner("Fetching data..."):
        df = fetch_data(TICKER, START.isoformat(), END.isoformat())
    if df.empty:
        st.error("No data found for that ticker & date range.")
        st.stop()

    st.subheader(f"{TICKER} — Raw & Indicator Data (most recent rows)")
    df = add_technical_indicators(df)
    st.dataframe(df[['Close', f"SMA_{SMA_SHORT}", f"SMA_{SMA_LONG}", 'rsi', 'macd', 'bb_high', 'bb_low', 'atr', 'volume_ma']].tail(10).style.format({
        'Close':'${:,.2f}', f"SMA_{SMA_SHORT}":'${:,.2f}', f"SMA_{SMA_LONG}":'${:,.2f}', 'rsi':'{:.2f}', 'macd':'{:.4f}'
    }))

    # SMA signals (historic)
    df = sma_signals(df)

    st.info("Running rolling LSTM backtest (retraining). This may take time depending on data length & LSTM window.")
    t0 = time.time()
    df_with_lstm = rolling_lstm_signals(df, window=int(LSTM_WINDOW), retrain_every=int(RETRAIN_FREQ), epochs=int(LSTM_EPOCHS), batch=int(LSTM_BATCH))
    t1 = time.time()
    st.write(f"Rolling LSTM pass finished in {t1-t0:.1f}s")

    # Chart: price + chosen overlays + markers
    st.subheader("Price chart with indicators & signals")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['Close'], name='Close', line=dict(width=1)))
    if SHOW_SMA:
        fig.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm[f"SMA_{SMA_SHORT}"], name=f"SMA_{SMA_SHORT}", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm[f"SMA_{SMA_LONG}"], name=f"SMA_{SMA_LONG}", line=dict(width=1)))
    if SHOW_BB:
        fig.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['bb_high'], name='BB High', line=dict(width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['bb_low'], name='BB Low', line=dict(width=1, dash='dash')))
    # SMA markers
    buys = df_with_lstm[df_with_lstm['sma_change'] > 0]
    sells = df_with_lstm[df_with_lstm['sma_change'] < 0]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='SMA BUY', marker_symbol='triangle-up', marker_size=10, marker_color='green'))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='SMA SELL', marker_symbol='triangle-down', marker_size=10, marker_color='red'))
    # LSTM markers
    lstm_buys = df_with_lstm[df_with_lstm['lstm_signal'] == 1]
    lstm_sells = df_with_lstm[df_with_lstm['lstm_signal'] == -1]
    if not lstm_buys.empty:
        fig.add_trace(go.Scatter(x=lstm_buys.index, y=lstm_buys['Close'], mode='markers', name='LSTM BUY', marker_symbol='star', marker_size=8, marker_color='darkgreen'))
    if not lstm_sells.empty:
        fig.add_trace(go.Scatter(x=lstm_sells.index, y=lstm_sells['Close'], mode='markers', name='LSTM SELL', marker_symbol='star', marker_size=8, marker_color='darkred'))

    # Optionally add indicator subplots below using separate charts
    fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # Indicator subplots (RSI, MACD, ATR, Volume)
    col1, col2 = st.columns(2)
    if SHOW_RSI:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['rsi'], name='RSI'))
        fig_rsi.update_layout(title="RSI", yaxis_title="RSI", height=250)
        col1.plotly_chart(fig_rsi, use_container_width=True)
    if SHOW_MACD:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['macd'], name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['macd_signal'], name='MACD signal'))
        fig_macd.update_layout(title="MACD", height=250)
        col1.plotly_chart(fig_macd, use_container_width=True)
    if SHOW_ATR:
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['atr'], name='ATR'))
        fig_atr.update_layout(title="ATR", height=250)
        col2.plotly_chart(fig_atr, use_container_width=True)
    if SHOW_VOLUME:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df_with_lstm.index, y=df_with_lstm['Volume'], name='Volume'))
        fig_vol.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['volume_ma'], name='Volume MA'))
        fig_vol.update_layout(title="Volume", height=250)
        col2.plotly_chart(fig_vol, use_container_width=True)

    # Forecast: show recent LSTM preds (where available)
    fc = df_with_lstm[['lstm_pred']].dropna().tail(50)
    if not fc.empty and SHOW_LSTM_FORECAST:
        st.subheader("Recent rolling LSTM one-day-ahead predictions (historic rolling predictions)")
        fc_display = fc.copy()
        fc_display['lstm_pred'] = fc_display['lstm_pred'].map(lambda x: f"${x:,.2f}")
        st.dataframe(fc_display)

        # Historic + predicted (overlay predicted where available)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_with_lstm.index, y=df_with_lstm['Close'], name='Historic Close'))
        fig2.add_trace(go.Scatter(x=fc.index, y=df_with_lstm.loc[fc.index, 'lstm_pred'], name='LSTM predicted', line=dict(dash='dash')))
        fig2.update_layout(title="Historic Close + Rolling LSTM Predictions", height=450)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No rolling LSTM predictions available (check params or data length).")

    # Backtest simulation
    st.subheader("Backtest Simulation & P&L")
    trades_df, pv_df, final_value, profit = backtest(df_with_lstm, policy=POLICY, capital=float(CAPITAL))
    st.metric("Starting capital", f"${CAPITAL:,.2f}")
    st.metric("Final portfolio value", f"${final_value:,.2f}")
    st.metric("Net profit / (loss)", f"${profit:,.2f} ({(profit/float(CAPITAL))*100:.2f}%)")
    if not pv_df.empty:
        pv_df = pv_df.reset_index()
        figpv = go.Figure()
        figpv.add_trace(go.Scatter(x=pv_df['date'], y=pv_df['pv'], name='Portfolio Value'))
        figpv.update_layout(title="Portfolio Value Over Time", height=450)
        st.plotly_chart(figpv, use_container_width=True)
    st.subheader("Trades from Backtest")
    if trades_df.empty:
        st.write("No trades executed under selected policy.")
    else:
        trades_display = trades_df.copy()
        trades_display['price'] = trades_display['price'].map(lambda x: f"${x:,.2f}")
        trades_display['shares'] = trades_display['shares'].map(lambda x: f"{x:,.6f}")
        st.dataframe(trades_display)

    # LLM-based recommendation for most recent date
    st.subheader("LLM-based recommendation for the most recent date (human-in-the-loop)")
    last_row = df_with_lstm.iloc[-1]
    evidence = {
        "symbol": TICKER,
        "date": str(df_with_lstm.index[-1].date()),
        "price": float(last_row['Close']),
        "sma_short": float(last_row.get(f"SMA_{SMA_SHORT}", np.nan)) if not np.isnan(last_row.get(f"SMA_{SMA_SHORT}", np.nan)) else None,
        "sma_long": float(last_row.get(f"SMA_{SMA_LONG}", np.nan)) if not np.isnan(last_row.get(f"SMA_{SMA_LONG}", np.nan)) else None,
        "rsi": float(last_row.get('rsi', np.nan)) if not np.isnan(last_row.get('rsi', np.nan)) else None,
        "macd": float(last_row.get('macd', np.nan)) if not np.isnan(last_row.get('macd', np.nan)) else None,
        "bb_high": float(last_row.get('bb_high', np.nan)) if not np.isnan(last_row.get('bb_high', np.nan)) else None,
        "bb_low": float(last_row.get('bb_low', np.nan)) if not np.isnan(last_row.get('bb_low', np.nan)) else None,
        "lstm_pred": float(last_row['lstm_pred']) if not np.isnan(last_row['lstm_pred']) else None,
        "lstm_confidence": float(last_row['lstm_confidence']) if not np.isnan(last_row['lstm_confidence']) else None,
        "sma_signal": int(last_row.get('sma_signal', 0)),
        "lstm_signal": int(last_row.get('lstm_signal', 0)),
        "recent_news": fetch_news_snippets(TICKER, days=3, max_articles=3) if NEWS_ENABLED else []
    }
    st.json(evidence, expanded=False)

    # Ask Gemini (or fallback) for recommendation
    llm_out = gemini_llm_recommendation(evidence)
    st.markdown("**LLM recommendation**")
    st.write(f"Recommendation: **{llm_out.get('recommendation')}**")
    st.write(f"Confidence: **{llm_out.get('confidence')}**")
    st.write("Rationale:")
    st.write(llm_out.get('rationale'))
    st.write("Evidence used:", llm_out.get('evidence_used'))

    # Accept / Reject
    colA, colB, colC = st.columns(3)
    if colA.button("ACCEPT recommendation"):
        fb = {
            'ts': str(pd.Timestamp.utcnow()),
            'symbol': TICKER,
            'date': evidence['date'],
            'price': evidence['price'],
            'rsi': evidence['rsi'] or 0,
            'macd': evidence['macd'] or 0,
            'bb_high': evidence['bb_high'] or 0,
            'bb_low': evidence['bb_low'] or 0,
            'sma_short': evidence['sma_short'] or 0,
            'sma_long': evidence['sma_long'] or 0,
            'lstm_confidence': evidence['lstm_confidence'] or 0,
            'llm_rec': llm_out.get('recommendation'),
            'llm_confidence': float(llm_out.get('confidence') or 0),
            'user_label': 1
        }
        insert_feedback(fb)
        st.success("Logged ACCEPT. You can retrain preference model using the button below.")
    if colB.button("REJECT recommendation"):
        fb = {
            'ts': str(pd.Timestamp.utcnow()),
            'symbol': TICKER,
            'date': evidence['date'],
            'price': evidence['price'],
            'rsi': evidence['rsi'] or 0,
            'macd': evidence['macd'] or 0,
            'bb_high': evidence['bb_high'] or 0,
            'bb_low': evidence['bb_low'] or 0,
            'sma_short': evidence['sma_short'] or 0,
            'sma_long': evidence['sma_long'] or 0,
            'lstm_confidence': evidence['lstm_confidence'] or 0,
            'llm_rec': llm_out.get('recommendation'),
            'llm_confidence': float(llm_out.get('confidence') or 0),
            'user_label': 0
        }
        insert_feedback(fb)
        st.error("Logged REJECT.")

    if colC.button("Retrain preference model (requires >=30 labels)"):
        model = train_preference_model()
        if model is None:
            st.warning("Not enough feedback data yet. Need ~30+ labeled rows to train reliably.")
        else:
            st.success("Trained preference model and saved locally.")

    st.subheader("Feedback log (most recent)")
    fb_df = load_feedback_df()
    if fb_df.empty:
        st.write("No feedback recorded yet.")
    else:
        st.dataframe(fb_df.head(200))

    # Show preference model's estimate for current evidence if available
    pref_model = load_preference_model()
    if pref_model is not None:
        feat = np.array([[evidence['rsi'] or 0, evidence['macd'] or 0, evidence['bb_high'] or 0, evidence['bb_low'] or 0, evidence['sma_short'] or 0, evidence['sma_long'] or 0, evidence['lstm_confidence'] or 0]])
        p_accept = pref_model.predict_proba(feat)[0,1]
        st.write(f"Preference model estimate probability you'd accept this recommendation: **{p_accept:.2f}**")

    # Final notes and caveats
    st.markdown("""
    ### Notes & caveats
    - **Rolling LSTM** retrains the model in-sample for each historical date to create realistic simulated predictions. This is accurate but computationally heavy. Use smaller windows or higher retrain intervals (`Retrain every N days`) to speed up.
    - Execution assumptions in backtest: buys use full capital (fractional shares optional). There is **no** slippage, commissions, or bid/ask spread modeled.
    - **LLM (Gemini)**: the app calls Gemini if enabled. LLM outputs should always be audited — show the evidence used and do not auto-execute trades.
    - **Security**: remove the hard-coded GEMINI_API_KEY from code and put it in environment variables or Streamlit secrets. Treat API keys like passwords.
    - This project is **research/educational** only — do not trade real funds without rigorous validation and risk controls.
    """)

else:
    st.info("Adjust settings in the sidebar and click 'Run full analysis (can be slow)'.")
