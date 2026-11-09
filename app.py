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

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")

DB_PATH = "stock_predictor.db"
SNAPSHOT = Path(__file__).with_name("data") / "reliance_daily_2016_2025.csv"
DEMO_MARKET = "Reliance demo · India (offline)"
CUSTOM_TICKER = "Enter a ticker"
MARKET_STOCKS = {
    "India": {"RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services"},
    "United States": {"AAPL": "Apple", "MSFT": "Microsoft"},
    "United Kingdom": {"SHEL.L": "Shell", "AZN.L": "AstraZeneca"},
    "Japan": {"7203.T": "Toyota", "6758.T": "Sony"},
    "Canada": {"RY.TO": "Royal Bank of Canada", "TD.TO": "Toronto-Dominion Bank"},
    "Australia": {"BHP.AX": "BHP", "CBA.AX": "Commonwealth Bank"},
}
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

def sign_in():
    st.write("Sign in to a local demo account to save future forecasts.")
    mode = st.radio("Account action", ["Sign in", "Create account"], horizontal=True)
    with st.form("demo_account"):
        username = st.text_input("Username").strip()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(mode, type="primary")
    st.caption("Use a demo password you do not use elsewhere.")
    if not submitted:
        return
    if not username or not password:
        st.warning("Enter both username and password.")
        return

    init_db()
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        if mode == "Create account":
            try:
                conn.execute("INSERT INTO users(username, password_hash) VALUES (?, ?)",
                             (username, password_hash(password)))
            except sqlite3.IntegrityError:
                st.error("That username already exists.")
                return
        else:
            row = conn.execute("SELECT password_hash FROM users WHERE username=?", (username,)).fetchone()
            if not row or row[0] != password_hash(password):
                st.error("Incorrect username or password.")
                return
    st.session_state.user = username
    st.rerun()

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
    currency = metadata.get("currency") or ""
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
            return prepare_prices(data), currency
    return prepare_prices(data.loc[dates.normalize() < now.normalize()]), currency

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



def render_forecast(username):
    with st.container(border=True):
        st.subheader("Choose a stock")
        market = st.selectbox("Stock market", [DEMO_MARKET, *MARKET_STOCKS, CUSTOM_TICKER])
        snapshot = market == DEMO_MARKET
        if snapshot:
            ticker = "RELIANCE.NS"
            stock_name = "Reliance Industries"
            st.caption("Practice with saved Reliance prices from 2016–2025.")
        elif market == CUSTOM_TICKER:
            ticker = st.text_input("Yahoo Finance ticker", placeholder="For example, AAPL or TCS.NS").strip().upper()
            stock_name = None
        else:
            stocks = MARKET_STOCKS[market]
            ticker = st.selectbox("Stock", list(stocks), format_func=stocks.get, key=f"stock_{market}")
            stock_name = stocks[ticker]

        period = None
        lstm_settings = None
        with st.expander("Advanced settings"):
            if not snapshot:
                period = st.selectbox("Historical period", ["1y", "2y", "5y", "10y"], index=2)
            model_choice = st.selectbox("Prediction model", [RANDOM_FOREST, LSTM])
            if model_choice == LSTM:
                lookback = st.slider("Past sessions to use", 20, 100, 60, 10)
                epochs = st.slider("Training passes", 5, 50, 20, 5)
                batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
                lstm_settings = (lookback, epochs, batch_size)
            else:
                st.caption("Random Forest averages forecasts from many decision trees.")
        run = st.button("Run forecast", type="primary")

    settings = (username, market, ticker, period, model_choice, lstm_settings)
    if run:
        st.session_state.pop("forecast_result", None)
        try:
            with st.spinner(f"Loading daily prices for {ticker}…"):
                if snapshot:
                    prices, currency = prepare_prices(pd.read_csv(SNAPSHOT)), "INR"
                else:
                    prices, currency = load_stock(ticker, period)
        except ValueError as error:
            st.error(str(error))
            return
        except (OSError, KeyError):
            if snapshot:
                st.error("Could not open the Reliance CSV. Check the project’s data folder.")
            else:
                st.error("Could not load Yahoo Finance prices. Try again or use the bundled Reliance dataset.")
            return

        test_start = SNAPSHOT_TEST_START if snapshot else None
        try:
            with st.spinner(f"Training {model_choice} and checking its forecasts…"):
                if lstm_settings:
                    forecast = train_lstm(prices, *lstm_settings, test_start=test_start)
                else:
                    forecast = train_random_forest(prices, test_start=test_start)
        except ImportError:
            if not lstm_settings:
                raise
            st.error("LSTM requires TensorFlow. See the optional LSTM setup in the README.")
            return
        except ValueError as error:
            st.error(str(error))
            return

        if username:
            history_model = f"{model_choice} · {DEMO_MARKET}" if snapshot else model_choice
            save_prediction(username, ticker, prices["Close"].iloc[-1], forecast.next_close, history_model)
        st.session_state.forecast_result = (settings, prices, forecast, currency)

    saved = st.session_state.get("forecast_result")
    if saved is not None and saved[0] == settings:
        _, prices, forecast, currency = saved
        render_result(prices, forecast, ticker, stock_name, model_choice, currency)
    else:
        st.caption("Run a forecast to see the estimated close and how the model performed on past data.")


def render_result(prices, forecast, ticker, stock_name, model_choice, currency):
    test = forecast.test
    metrics = forecast.metrics()
    latest = prices["Close"].iloc[-1]
    origin = prices["Date"].iloc[-1]
    percent = (forecast.next_close / latest - 1) * 100
    price_prefix = "₹" if currency == "INR" else f"{currency} " if currency else ""
    price_axis = f"Adjusted close ({currency})" if currency else "Adjusted close"

    st.subheader("Forecast result")
    if stock_name:
        st.write(stock_name)
    st.caption(f"{ticker} · {model_choice} · Uses prices through {origin:%d %b %Y}")
    if currency in ("GBp", "GBX"):
        st.caption("This listing is quoted in British pence.")
    estimate, previous = st.columns(2)
    estimate.metric("Estimated next close", f"{price_prefix}{forecast.next_close:,.2f}",
                    f"{percent:+.2f}% vs. last close", delta_color="off", border=True)
    previous.metric("Last adjusted close", f"{price_prefix}{latest:,.2f}", border=True)

    st.subheader("How accurate was it?")
    model_mae, baseline_mae = metrics["MAE"]
    st.write(f"The model was off by **{price_prefix}{model_mae:,.2f}** on average. Carrying the last close forward "
             f"was off by **{price_prefix}{baseline_mae:,.2f}** on the same dates.")
    st.caption(f"{len(test)} test forecasts from {test['Date'].iloc[0]:%d %b %Y} to {test['Date'].iloc[-1]:%d %b %Y}.")

    comparison = go.Figure()
    for column, name, dash in (("Actual", "Actual close", "solid"), ("Predicted", "Estimated close", "dash")):
        comparison.add_trace(go.Scatter(x=test["Date"], y=test[column], name=name, mode="lines", line=dict(dash=dash)))
    comparison.update_layout(height=320, margin=dict(l=0, r=0, t=24, b=0), hovermode="x unified",
                             legend=dict(orientation="h", y=1.08), yaxis_title=price_axis)
    st.plotly_chart(comparison, width="stretch", config={"displayModeBar": False})

    with st.expander("Test details and downloads"):
        st.caption("MAE measures the average size of an error. RMSE gives bigger errors more weight. "
                   "Both use the stock’s price units.")
        st.dataframe(metrics.replace({"Selected model": model_choice}).round(3), hide_index=True)
        st.download_button("Download test forecasts (CSV)", test.assign(Currency=currency).to_csv(index=False),
                           file_name=f"{ticker.lower()}_{'random_forest' if model_choice == RANDOM_FOREST else 'lstm'}_results.csv",
                           mime="text/csv", on_click="ignore")
        st.line_chart(prices, x="Date", y="Close", y_label=price_axis, height=240)


def render_comparison():
    st.subheader("Compare models")
    st.write("See how three trained models compare with a simple guess on the same Reliance trading days.")
    with st.expander("How the comparison works"):
        st.write("The simple guess carries the previous day’s close forward. For each year in 2023 and 2024, "
                 "the models train on earlier dates and forecast the sessions in that year. "
                 "The 2025 data is reserved for the separate forecast test.")
        st.write("Random Forest averages many trees. Ridge gives each input a weight. Gradient boosting "
                 "builds trees one at a time to improve earlier predictions.")
    if not SNAPSHOT.is_file():
        st.error("The comparison needs the Reliance CSV in the project’s data folder.")
        return

    if st.button("Compare models on Reliance", type="primary"):
        st.session_state.pop("benchmark", None)
        try:
            with st.spinner("Comparing forecasts from 2023 and 2024…"):
                prices = prepare_prices(pd.read_csv(SNAPSHOT))
                st.session_state.benchmark = walk_forward_benchmark(prices)
        except (OSError, ValueError) as error:
            st.error(f"Could not run the comparison. {error}")

    benchmark = st.session_state.get("benchmark")
    if benchmark is not None:
        pooled = benchmark.loc[benchmark["Period"] == VALIDATION_PERIOD].sort_values("MAE")
        best_model = pooled.loc[pooled["Model"] != LAST_CLOSE, "Model"].iloc[0]
        st.caption(f"{VALIDATION_PERIOD} · Lowest average error: {pooled['Model'].iloc[0]} · "
                   f"Best trained model: {best_model}")
        st.dataframe(pooled.drop(columns="Period").round(3), hide_index=True, column_config={
            "MAE": "Average error (₹)", "RMSE": "RMSE (₹)",
        })
        with st.expander("Results by year"):
            st.dataframe(benchmark.loc[benchmark["Period"] != VALIDATION_PERIOD].round(3), hide_index=True)
        st.download_button("Download comparison scores (CSV)", benchmark.to_csv(index=False),
                           file_name="reliance_walk_forward_scores.csv", mime="text/csv", on_click="ignore")
    st.download_button("Download Reliance dataset (CSV)", SNAPSHOT.read_bytes(),
                       file_name=SNAPSHOT.name, mime="text/csv", on_click="ignore")


def render_history(username):
    st.subheader("Saved forecasts")
    if not username:
        sign_in()
        return
    st.text(f"Signed in as {username}")
    if st.button("Sign out"):
        del st.session_state["user"]
        st.session_state.pop("forecast_result", None)
        st.session_state.pop("benchmark", None)
        st.rerun()
    history = get_history(username)
    if history.empty:
        st.caption("Run a forecast while signed in to save it here.")
    else:
        st.dataframe(history.rename(columns={
            "ticker": "Ticker", "created_at": "Saved on", "last_close": "Last close",
            "predicted_close": "Estimated close", "model": "Model",
        }), hide_index=True)


st.caption("MACHINE LEARNING COURSEWORK")
st.title("Stock Price Predictor")
st.write("Estimate the next trading session’s closing price and see how the forecast performed on past data.")

username = st.session_state.get("user")
forecast_tab, comparison_tab, history_tab = st.tabs(["Forecast", "Model comparison", "History"])
with forecast_tab:
    render_forecast(username)
with comparison_tab:
    render_comparison()
with history_tab:
    render_history(username)
