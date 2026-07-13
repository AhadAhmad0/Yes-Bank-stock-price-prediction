# Yes Bank Stock Closing Price Prediction

Exploratory Data Analysis and regression modelling on 15 years of Yes Bank monthly stock price data (2005–2020), covering the bank's steady growth period and the dramatic 2018 stock crash triggered by the fraud case involving co-founder Rana Kapoor.

**Repo:** [AhadAhmad0/Yes-Bank-stock-price-prediction](https://github.com/AhadAhmad0/Yes-Bank-stock-price-prediction)

---

## Problem Statement

Analyze Yes Bank's historical monthly stock price behaviour (Open, High, Low, Close), quantify the impact of the 2018 governance crisis on price and volatility, and build a regression model to accurately predict the monthly closing price.

## Dataset

- **Source:** Monthly Yes Bank stock prices, July 2005 – present
- **Records:** 185 rows, 5 columns (`Date`, `Open`, `High`, `Low`, `Close`)
- **File:** `data_YesBank_StockPrices.csv`
- No missing values, no duplicates. `Date` arrives as a string (`Mon-YY` format) and is parsed to datetime during wrangling.

---

## Notebook 1 — Exploratory Data Analysis
`Yes_Bank_EDA_Completed.ipynb`

**What it covers:**
- Data cleaning: datetime parsing, chronological sort, null/duplicate checks
- Feature engineering: monthly return, trading range (High − Low), 3-month & 12-month rolling averages, pre/post-crisis flag (split at March 2018)
- 15 visualizations: price trend, OHLC comparison, rolling averages, return distribution, yearly box plots, volatility over time, crisis-period comparison, zoomed 2018–2020 crash view, correlation heatmap, pair plot, and more

**Key insights:**
- Stock rose from ~₹12 (2005) to an all-time high of ~₹393 (2018), then lost over 90% of its value within ~2 years
- Open, High, Low, and Close are correlated **>0.99** with each other — near-perfect multicollinearity
- Return volatility increased sharply post-2018, with a pronounced fat left tail in the return distribution
- A 3-month/12-month moving-average bearish crossover preceded the 2018 crash

---

## Notebook 2 — ML Regression
`Yes_Bank_ML_Completed.ipynb`

**Approach:**
- Feature selection: only `Open` retained as raw price input (avoiding multicollinearity from High/Low/Close), combined with engineered `Range`, `Monthly_Return`, and `Crisis_Period`
- Scaling: `StandardScaler`
- Train/test split: **chronological 80/20** (not random — this is time-series data, a random split would leak future information into training)
- 3 hypothesis tests conducted (Welch's t-test, Levene's test, Pearson correlation test) to statistically confirm the pre/post-crisis regime shift
- 3 models trained and tuned via `GridSearchCV`: Linear Regression, Random Forest, Gradient Boosting

**Hypothesis test results:**

| Test | What it checked | Result |
|---|---|---|
| Welch's t-test | Mean monthly return, pre vs post crisis | Significant (p < 0.05) — means differ |
| Levene's test | Return variance, pre vs post crisis | Significant (p < 0.001) — volatility differs |
| Pearson correlation | Open vs Close relationship | r = 0.978, highly significant |

**Model comparison:**

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~32.4 | — | 0.934 |
| Random Forest (tuned) | — | — | 0.923 |
| **Gradient Boosting (tuned)** | **~28.6** | **~13.3** | **0.949** |

**Final model:** Gradient Boosting (tuned) — best RMSE, MAE, and R² of the three, capturing non-linear interactions between `Range`, `Crisis_Period`, and `Open` that the linear model misses.

Model and scaler are saved via `joblib` (`yes_bank_close_price_model.joblib`, `yes_bank_scaler.joblib`) and reloaded for a sanity-check prediction on held-out data.

---

## Limitation

This model predicts `Close` using **same-month** `Open` price — it's a strong explanatory model, not a forward-looking forecaster. It cannot predict next month's price before that month's Open is known, and no quantitative model based on price history alone could have anticipated a governance-driven shock like the 2018 fraud case. Real forecasting would require lagged features, and any deployment should be paired with qualitative risk monitoring.

---

## Repo Structure

```
Yes-Bank-stock-price-prediction/
├── README.md
├── data/
│   └── data_YesBank_StockPrices.csv
├── notebooks/
│   ├── Yes_Bank_EDA_Completed.ipynb
│   └── Yes_Bank_ML_Completed.ipynb
├── models/
│   ├── yes_bank_close_price_model.joblib
│   └── yes_bank_scaler.joblib
└── requirements.txt
```

## How to Run

```bash
git clone https://github.com/AhadAhmad0/Yes-Bank-stock-price-prediction.git
cd Yes-Bank-stock-price-prediction
pip install -r requirements.txt
jupyter notebook notebooks/Yes_Bank_EDA_Completed.ipynb
```

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, SciPy, scikit-learn, joblib

---

**Author:** Ahad Ahmad — [GitHub](https://github.com/AhadAhmad0) · [LinkedIn](https://linkedin.com/in/ahadahmad7/)
