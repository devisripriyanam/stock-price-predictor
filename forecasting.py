from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler


RETURN_LAGS = (1, 2, 3)
AVERAGE_WINDOWS = (5, 10, 20)
FEATURE_COLUMNS = [
    "Return", "Volatility", "Volume change",
    *(f"Return lag {days}" for days in RETURN_LAGS),
    *(f"Distance to MA{days}" for days in AVERAGE_WINDOWS),
]


@dataclass
class Forecast:
    test: pd.DataFrame
    next_close: float

    def metrics(self):
        actual = self.test["Actual"]
        return pd.DataFrame([
            {
                "Forecast": name,
                "MAE": mean_absolute_error(actual, self.test[column]),
                "RMSE": root_mean_squared_error(actual, self.test[column]),
                "R²": r2_score(actual, self.test[column]),
            }
            for name, column in [("Selected model", "Predicted"), ("Last close", "Baseline")]
        ])


def prepare_prices(data):
    if data.empty or not {"Date", "Close", "Volume"}.issubset(data.columns):
        raise ValueError("No daily prices were returned. Check the ticker or use the bundled dataset.")

    close_column = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = pd.DataFrame({
        "Date": pd.to_datetime(data["Date"]).dt.tz_localize(None).dt.normalize(),
        "Close": pd.to_numeric(data[close_column], errors="coerce"),
        "Volume": pd.to_numeric(data["Volume"], errors="coerce"),
    })
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.loc[prices["Date"].notna() & prices["Close"].gt(0)]
    return prices.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def return_features(prices):
    close = prices["Close"]
    returns = close.pct_change()
    volume = prices["Volume"].where(prices["Volume"].gt(0))
    features = pd.DataFrame({
        "Return": returns,
        "Volatility": returns.rolling(10).std(),
        "Volume change": volume.pct_change(fill_method=None).fillna(0),
    })
    for days in RETURN_LAGS:
        features[f"Return lag {days}"] = returns.shift(days)
    for days in AVERAGE_WINDOWS:
        features[f"Distance to MA{days}"] = close / close.rolling(days).mean() - 1
    return features


def test_start_index(prices, test_start, min_training):
    split = prices["Date"].searchsorted(pd.Timestamp(test_start)) if test_start else int(len(prices) * 0.8)
    if split < min_training or len(prices) - split < 2:
        raise ValueError("There is too little daily history for this model. Choose a longer period or a shorter lookback.")
    return split


def test_results(prices, split, predicted):
    return pd.DataFrame({
        "Date": prices["Date"].iloc[split:].to_numpy(),
        "Actual": prices["Close"].iloc[split:].to_numpy(),
        "Predicted": predicted,
        "Baseline": prices["Close"].iloc[split - 1:-1].to_numpy(),
    })


def train_random_forest(prices, test_start=None):
    split = test_start_index(prices, test_start, min_training=80)
    features = return_features(prices)[FEATURE_COLUMNS]
    close = prices["Close"]
    target = close.shift(-1) / close - 1
    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1,
    )

    # Feature row t predicts the label on row t + 1. The first 19 rows warm up MA20.
    first = max(AVERAGE_WINDOWS) - 1
    model.fit(features.iloc[first:split - 1], target.iloc[first:split - 1])
    test_return = model.predict(features.iloc[split - 1:-1])
    test_prices = close.iloc[split - 1:-1].to_numpy() * (1 + test_return)

    # Refit through the most recent observed label for the next-session estimate.
    model.fit(features.iloc[first:-1], target.iloc[first:-1])
    next_return = model.predict(features.iloc[-1:])[0]
    return Forecast(test_results(prices, split, test_prices), float(close.iloc[-1] * (1 + next_return)))


def train_lstm(prices, lookback, epochs, batch_size, test_start=None):
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.utils import set_random_seed

    def make_model():
        model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(64, return_sequences=True), Dropout(0.2),
            LSTM(32), Dropout(0.2), Dense(16, activation="relu"), Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    split = test_start_index(prices, test_start, min_training=lookback + 30)
    val_start = split - max(1, (split - lookback) // 10)
    close = prices["Close"].to_numpy().reshape(-1, 1)

    def sequences(scaled):
        return np.asarray([scaled[i - lookback:i] for i in range(lookback, len(scaled))])

    set_random_seed(42)
    scaler = MinMaxScaler().fit(close[:val_start])
    scaled = scaler.transform(close)
    inputs = sequences(scaled)
    model = make_model()
    history = model.fit(
        inputs[:val_start - lookback], scaled[lookback:val_start],
        validation_data=(inputs[val_start - lookback:split - lookback], scaled[val_start:split]),
        epochs=epochs, batch_size=batch_size, shuffle=False, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )
    test_prices = scaler.inverse_transform(model.predict(inputs[split - lookback:], verbose=0)).ravel()

    best_epochs = int(np.argmin(history.history["val_loss"])) + 1
    scaled = scaler.fit_transform(close)
    model = make_model()
    model.fit(sequences(scaled), scaled[lookback:], epochs=best_epochs,
              batch_size=batch_size, shuffle=False, verbose=0)
    next_scaled = model.predict(scaled[-lookback:][None, :, :], verbose=0)
    next_close = float(scaler.inverse_transform(next_scaled)[0, 0])
    return Forecast(test_results(prices, split, test_prices), next_close)
