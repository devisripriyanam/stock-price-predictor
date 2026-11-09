import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecasting import (
    LAST_CLOSE, RANDOM_FOREST, SNAPSHOT_TEST_START, VALIDATION_PERIOD,
    prepare_prices, train_lstm, train_random_forest, walk_forward_benchmark,
)

st.set_page_config(page_title="Stock AI | Predictor", page_icon="📈", layout="wide")

DB_PATH = "stock_predictor.db"
SNAPSHOT = Path(__file__).with_name("data") / "reliance_daily_2016_2025.csv"
SNAPSHOT_LABEL = "Reliance dataset (2016–2025)"
LIVE_LABEL = "Yahoo Finance ticker"
LSTM = "LSTM (adjusted close)"

def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, ticker TEXT,
            created_at TEXT, last_close REAL, predicted_close REAL, model TEXT
        )""")

def password_hash(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def authenticate():
    init_db()
    if "user" in st.session_state:
        return True
    st.title("🔐 Stock AI")
    st.write("Create a local demo account or sign in. This login is for project demonstration only.")
    mode = st.radio("Choose", ["Sign in", "Create account"], horizontal=True)
    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password")
    if st.button(mode, type="primary"):
        if not username or not password:
            st.warning("Enter both username and password.")
        else:
            with closing(sqlite3.connect(DB_PATH)) as conn, conn:
                if mode == "Create account":
                    try:
                        conn.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)",
                                     (username, password_hash(password)))
                        conn.commit()
                        st.session_state.user = username
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("That username already exists.")
                else:
                    row = conn.execute("SELECT password_hash FROM users WHERE username=?",
                                       (username,)).fetchone()
                    if row and row[0] == password_hash(password):
                        st.session_state.user = username
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")
    st.caption("Demo note: use a non-sensitive password. This local prototype is not a production authentication system.")
    return False

def load_stock(symbol, period):
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,29}", symbol):
        raise ValueError("Enter one Yahoo Finance ticker, such as RELIANCE.NS or AAPL.")

    query = urlencode({"range": period, "interval": "1d", "events": "history"})
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{query}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=12) as response:
            results = json.load(response)["chart"]["result"]
    except HTTPError as error:
        if error.code in (404, 422):
            raise ValueError(f"No daily history was found for {symbol}.") from error
        raise
    if not results or not results[0].get("timestamp"):
        raise ValueError(f"No daily history was found for {symbol}.")

    result = results[0]
    metadata = result["meta"]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(metadata["exchangeTimezoneName"])
    indicators = result["indicators"]
    bars = indicators["quote"][0]
    data = pd.DataFrame({"Date": dates, "Close": bars["close"], "Volume": bars["volume"]})
    if indicators.get("adjclose"):
        data["Adj Close"] = indicators["adjclose"][0]["adjclose"]

    now = pd.Timestamp.now(tz=dates.tz)
    end = metadata.get("currentTradingPeriod", {}).get("regular", {}).get("end")
    if end:
        session_close = pd.Timestamp(end, unit="s", tz=dates.tz)
        if session_close.normalize() == now.normalize() and now >= session_close:
            return prepare_prices(data)
    return prepare_prices(data.loc[dates.normalize() < now.normalize()])

def save_prediction(username, ticker, close, pred, model_name):
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute("""INSERT INTO prediction_history
        (username, ticker, created_at, last_close, predicted_close, model)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (username, ticker, date.today().isoformat(), float(close), float(pred), model_name))
        conn.commit()

def get_history(username):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(
            "SELECT ticker, created_at, last_close, predicted_close, model FROM prediction_history WHERE username=? ORDER BY id DESC LIMIT 100",
            conn, params=(username,))

if not authenticate():
    st.stop()

username = st.session_state.user
with st.sidebar:
    st.markdown("## ⚙️ Model settings")
    source = st.selectbox("Data source", [SNAPSHOT_LABEL, LIVE_LABEL])
    snapshot = source == SNAPSHOT_LABEL
    if snapshot:
        ticker, period = "RELIANCE.NS", None
        st.caption("Offline dataset with 2025 reserved for testing.")
    else:
        ticker = st.text_input("Ticker symbol", "RELIANCE.NS").strip().upper()
        period = st.selectbox("Historical period", ["1y", "2y", "5y", "10y"], index=2)
    model_choice = st.selectbox("Prediction model", [RANDOM_FOREST, LSTM])
    lstm_settings = None
    if model_choice == LSTM:
        lookback = st.slider("LSTM lookback window (trading days)", 20, 100, 60, 10)
        epochs = st.slider("Training epochs", 5, 50, 20, 5)
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
        lstm_settings = (lookback, epochs, batch_size)
    run = st.button("🚀 Train & predict", type="primary", width="stretch")
    st.divider()
    if st.button("Sign out", width="stretch"):
        del st.session_state["user"]
        st.session_state.pop("forecast_result", None)
        st.session_state.pop("benchmark", None)
        st.rerun()

st.title("📈 Stock AI — Price Prediction")
st.caption("Next-session adjusted close · Chronological evaluation · Last-close benchmark")
st.caption("Machine learning coursework demonstration.")

if snapshot and SNAPSHOT.is_file():
    st.download_button("Download Reliance dataset (CSV)", SNAPSHOT.read_bytes(),
                       file_name=SNAPSHOT.name, mime="text/csv", on_click="ignore")
    with st.expander("Walk-forward model benchmark"):
        st.caption("Each model trains on sessions before the validation year. The 2025 data is reserved for the final test.")
        if st.button("Run walk-forward benchmark"):
            st.session_state.pop("benchmark", None)
            try:
                with st.spinner("Scoring models on 2023 and 2024..."):
                    validation_prices = prepare_prices(pd.read_csv(SNAPSHOT))
                    st.session_state.benchmark = walk_forward_benchmark(validation_prices)
            except (OSError, ValueError) as error:
                st.error(f"Could not run the benchmark. {error}")

        benchmark = st.session_state.get("benchmark")
        if benchmark is not None:
            pooled = benchmark.loc[benchmark["Period"] == VALIDATION_PERIOD].sort_values("MAE")
            best_model = pooled.loc[pooled["Model"] != LAST_CLOSE, "Model"].iloc[0]
            st.write(f"Lowest overall MAE in {VALIDATION_PERIOD} — **{pooled['Model'].iloc[0]}**")
            st.caption(f"Lowest MAE among trained models — {best_model}. Errors are in INR.")
            st.dataframe(pooled.drop(columns="Period").round(3), width="stretch", hide_index=True)
            st.dataframe(benchmark.loc[benchmark["Period"] != VALIDATION_PERIOD].round(3),
                         width="stretch", hide_index=True)
            st.download_button("Download benchmark scores (CSV)", benchmark.to_csv(index=False),
                               file_name="reliance_walk_forward_scores.csv", mime="text/csv", on_click="ignore")

settings = (username, source, ticker, period, model_choice, lstm_settings)
if run:
    st.session_state.pop("forecast_result", None)
    try:
        with st.spinner(f"Loading {ticker} data..."):
            prices = prepare_prices(pd.read_csv(SNAPSHOT)) if snapshot else load_stock(ticker, period)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    except (OSError, KeyError):
        if snapshot:
            st.error("Could not read the bundled Reliance dataset. Check the project's data folder.")
        else:
            st.error("Could not load Yahoo Finance data. Try again or choose the bundled Reliance dataset.")
        st.stop()

    test_start = SNAPSHOT_TEST_START if snapshot else None
    try:
        with st.spinner(f"Training {model_choice}..."):
            if lstm_settings:
                forecast = train_lstm(prices, *lstm_settings, test_start=test_start)
            else:
                forecast = train_random_forest(prices, test_start=test_start)
    except ImportError:
        if not lstm_settings:
            raise
        st.error("LSTM requires TensorFlow. See the optional LSTM setup in the README.")
        st.stop()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    history_model = f"{model_choice} · {SNAPSHOT_LABEL}" if snapshot else model_choice
    save_prediction(username, ticker, prices["Close"].iloc[-1], forecast.next_close, history_model)
    st.session_state.forecast_result = (settings, prices, forecast)

saved = st.session_state.get("forecast_result")
if saved is not None and saved[0] == settings:
    _, prices, forecast = saved
    test = forecast.test
    metrics = forecast.metrics()
    latest_close = prices["Close"].iloc[-1]
    origin = prices["Date"].iloc[-1].strftime("%d %b %Y")
    delta = forecast.next_close - latest_close
    percent = delta / latest_close * 100

    st.subheader(ticker)
    st.caption(f"Forecast origin: {origin} · Estimate for the following trading session")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest adjusted close", f"{latest_close:,.2f}")
    m2.metric("Next-session estimate", f"{forecast.next_close:,.2f}", f"{delta:+,.2f} ({percent:+.2f}%)")
    m3.metric("Model test MAE", f"{metrics.at[0, 'MAE']:,.2f}")
    m4.metric("Last-close test MAE", f"{metrics.at[1, 'MAE']:,.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧪 Model evaluation", "🧠 Methodology", "🗂️ History"])
    with tab1:
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=prices["Date"], y=prices["Close"], name="Historical close", mode="lines"))
        chart.update_layout(height=380, title="Historical adjusted close", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(chart, width="stretch")
        compare = go.Figure()
        for column in ("Actual", "Predicted", "Baseline"):
            compare.add_trace(go.Scatter(x=test["Date"], y=test[column], name=column, mode="lines"))
        compare.update_layout(height=420, title="One-session forecasts on held-out data",
                              xaxis_title="Target trading date", yaxis_title="Adjusted close")
        st.plotly_chart(compare, width="stretch")
    with tab2:
        st.subheader("Performance on held-out data")
        st.write(f"{len(test)} forecasts from {test['Date'].iloc[0]:%d %b %Y} through {test['Date'].iloc[-1]:%d %b %Y}.")
        st.caption("MAE and RMSE use the ticker’s price units. Lower values indicate smaller errors.")
        st.dataframe(metrics.replace({"Selected model": model_choice}).round(3), width="stretch", hide_index=True)
        st.download_button("Download held-out forecasts (CSV)", test.to_csv(index=False),
                           file_name=f"{ticker.lower()}_{'lstm' if lstm_settings else 'random_forest'}_results.csv",
                           mime="text/csv", on_click="ignore")
    with tab3:
        st.markdown("""
        1. Read daily adjusted closing prices and volume from Yahoo Finance or the bundled snapshot.
        2. Random Forest uses lagged returns, rolling volatility, volume change and distances from moving averages.
           It predicts the next session’s return, then multiplies the current close by `1 + predicted return`.
        3. The optional LSTM predicts the next adjusted close from a sequence of earlier closes.
           Its evaluation scaler is fitted on training prices before validation and testing.
        4. Score one-session forecasts in chronological order. The baseline carries the previous session’s close forward.
        5. Refit the selected model on all available history for the forecast following the latest observed session.
        """)
    with tab4:
        st.dataframe(get_history(username), width="stretch", hide_index=True)
else:
    st.info("Click **Train & predict** to evaluate the selected model and forecast the next session.")
    if not snapshot:
        st.caption("Example tickers: RELIANCE.NS, TCS.NS, INFY.NS, AAPL, MSFT.")
