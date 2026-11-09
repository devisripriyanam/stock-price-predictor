# Advanced Stock Price Predictor

A Streamlit project that estimates the next trading session’s adjusted close and compares the estimate with a last-close baseline. The default model is a Random Forest regressor trained on daily returns.

## Run on Windows

Open PowerShell in the project folder and run these commands with Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS or Linux, use `python3` to create the environment and `.venv/bin/python` to install and run the app. Open the local URL shown in the terminal and create a demo account. The app saves prediction history in a local SQLite file.

## Dataset

The bundled `data/reliance_daily_2016_2025.csv` contains 2,470 daily records for Reliance Industries (`RELIANCE.NS`), from 1 January 2016 through 31 December 2025. It includes date, open, high, low, close, adjusted close and volume. The default dataset works offline and can be downloaded from the app. The alternative Yahoo Finance source retrieves daily data for a ticker you select. [Historical prices](https://finance.yahoo.com/quote/RELIANCE.NS/history/) · [Source data](https://query1.finance.yahoo.com/v8/finance/chart/RELIANCE.NS?period1=1451586600&period2=1767205800&interval=1d&events=history)

The bundled dataset's forecast starts from the 31 December 2025 close. In Yahoo Finance mode, the app includes the current exchange-local day when Yahoo provides a completed regular session for that day.

## Model and evaluation

1. Compute the current return, three lagged daily returns, 10-session volatility, volume change and distances from the 5-, 10- and 20-session moving averages. Each row uses information available through that session’s close. An unavailable volume change uses zero so the price record stays in the dataset.
2. Fit the forest to `next adjusted close / current adjusted close - 1`. Convert the predicted return to a price with `current adjusted close * (1 + predicted return)`.
3. Compare against the previous session’s close on the same target dates. The bundled dataset holds out all 249 observations in 2025. Yahoo Finance mode holds out the latest 20% of observations. The evaluation model stays fixed throughout the test period and receives the information available at each forecast origin.
4. Report MAE, RMSE and R² and export the dated forecasts. Refit the selected model on all known labels for the final estimate using the latest observed session.

| Model | 2025 MAE (INR) | 2025 RMSE (INR) |
| --- | ---: | ---: |
| Random Forest on returns | 12.18 | 16.37 |
| Last-close baseline | 12.17 | 16.45 |

The baseline has slightly lower MAE on this holdout, and the forest has slightly lower RMSE. The exercise measures historical one-session forecast errors for coursework.

### Walk-forward model comparison

With the bundled Reliance dataset selected, open **Walk-forward model benchmark** and click **Run walk-forward benchmark**. It compares Random Forest, Ridge regression, and histogram gradient boosting on the same return features, alongside the last-close baseline.

The first fold trains through 2022 and scores 2023. The second expands training through 2023 and scores 2024. The dashboard reports MAE and RMSE for each year and across all 491 validation targets, ranks the methods by pooled MAE, and exports the scores as a CSV. Use **Train & predict** separately to evaluate the selected forecast model on the 2025 holdout.

On the bundled data, last close has the lowest pooled MAE at ₹11.967. Random Forest leads the trained models at ₹12.002.

### Optional LSTM comparison

With Python 3.13 installed on Windows, create a separate environment for the optional LSTM and select **LSTM (adjusted close)** in the sidebar. The main app and Random Forest run with the standard requirements. [TensorFlow installation and supported versions](https://www.tensorflow.org/install/pip)

```powershell
py -3.13 -m venv .venv-lstm
.\.venv-lstm\Scripts\python.exe -m pip install -r requirements.txt tensorflow
.\.venv-lstm\Scripts\python.exe -m streamlit run app.py
```
