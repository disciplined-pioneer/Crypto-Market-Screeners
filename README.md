# Crypto Market Screeners

## Project Description

This project provides a set of tools for analyzing the cryptocurrency market. Its main purpose is to assist traders and analysts in finding key market signals using screeners. Each module in the project implements its approach to data analysis, offering powerful tools for identifying market opportunities.

---

## Project Structure

```
├─── data
│   ├─── ATR
│   │   └─── volatility_results.json
│   ├─── Sentiment
│   │   ├─── sentiment_*.csv
│   └─── market_ratio
├─── atr_screener.py
├─── data_processing.py
├─── isol_forest_screener.py
├─── market_ratio.py
├─── sentiment.py
└─── volume_spikes.py
```

- **data/** — Directory containing the results of the screeners and preprocessed data.
- **atr_screener.py** — Module for analyzing volatility based on ATR (Average True Range).
- **data_processing.py** — Module for fetching and preprocessing cryptocurrency futures data.
- **isol_forest_screener.py** — Tool for detecting anomalies in market data using Isolation Forest.
- **market_ratio.py** — Screener for analyzing market metrics.
- **sentiment.py** — Tool for assessing market sentiment.
- **volume_spikes.py** — Module for detecting trading volume spikes.

---

## Screeners and Their Functions

### 1. ATR Screener (atr_screener.py)

#### Purpose:
The ATR-based screener evaluates the volatility of cryptocurrency pairs. It identifies periods of high and low volatility, which can be useful for determining market conditions.

#### Key Functions:
- Calculate ATR for a specified period.
- Identify periods of high/low volatility.
- Visualize ATR and market volatility.

#### Usage:
1. Specify a list of cryptocurrency pairs in the code.
2. Run the script. Results are saved to `data/ATR/volatility_results.json`.
3. Analyze which pairs exhibit high volatility.

#### Example Output:
- **High Volatility:** BTCUSDT, ETHUSDT
- **Low Volatility:** ADAUSDT, SOLUSDT

---

### 2. Data Processing (data_processing.py)

#### Purpose:
Provides a set of tools for loading, processing, and aggregating price, volume, delta, and open interest data.

#### Key Functions:
- Load historical price and volume data via the Binance API.
- Calculate delta and open interest.
- Combine all data into a convenient format for analysis.

---

### 3. Isolation Forest Screener (isol_forest_screener.py)

#### Purpose:
Identify anomalies in market data, such as trading volume spikes.

#### Key Functions:
- Detect anomalies in volume and price data.
- Use the Isolation Forest algorithm.

#### Usage:
1. Define the list of cryptocurrency pairs and time frames.
2. Use the `find_anomalies()` function to detect anomalies in volumes.
3. Interpret the results to identify market opportunities.

---

### 4. Market Ratio Screener (market_ratio.py)

#### Purpose:
Analyze market ratios, such as the buy/sell volume ratio.

#### Key Functions:
- Calculate market metrics.
- Identify supply and demand imbalances.

#### Usage:
1. Specify analysis parameters.
2. Run the script and analyze the results.

---

### 5. Sentiment Screener (sentiment.py)

#### Purpose:
Evaluate market sentiment based on data from social networks or other sources.

#### Key Functions:
- Analyze text and compute sentiment metrics.
- Output results for each cryptocurrency pair.

#### Usage:
1. Load sentiment data in CSV format into the `data/Sentiment/` directory.
2. Run the analysis and interpret the results.

---

### 6. Volume Spikes Screener (volume_spikes.py)

#### Purpose:
Detect trading volume spikes that may signal market changes.

#### Key Functions:
- Search for significant volume changes.
- Display the time frames of anomalies.

#### Usage:
1. Set volume parameters and time frames.
2. Run the script and analyze periods of abnormal activity.

---

## Example Usage

1. Ensure all dependencies are installed (Python 3.8+, libraries from `requirements.txt`).
2. Configure Binance API parameters in the `data_processing.py` file.
3. Run scripts to analyze the data of interest.

## Running the Project and Activating Dependencies

```
git clone https://github.com/disciplined-pioneer/Crypto-Market-Screeners.git
```

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Useful Links

- [Binance API Documentation](https://binance-docs.github.io/apidocs/futures/en/)
- [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Seaborn for Visualization](https://seaborn.pydata.org/)

## License
The project is distributed under the MIT license. Free use and modification are permitted.
