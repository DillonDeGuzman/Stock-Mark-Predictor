import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# Load and clean data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")
sp500.index

# Plot historical closing prices
sp500.plot.line(y="Close", use_index=True)
plt.show()

# Remove unnecessary columns
del sp500["Dividends"]
del sp500["Stock Splits"]

# Add target column: 1 if next day > today, else 0
sp500["Tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

# Use only recent data
sp500 = sp500.loc["1990-01-01":].copy()

# Train-test split
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Model training
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(train[predictors], train["Target"])

# Predictions and evaluation
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
print("Precision Score:", precision_score(test["Target"], preds))

# Plot predictions vs actual
combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()
plt.show()

# Prediction function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
print(predictions["Predictions"].value_counts())
print("Backtest Precision:", precision_score(predictions["Target"], predictions["Predictions"]))
print("Class Balance:", predictions["Target"].value_counts() / predictions.shape[0])

# Add rolling average features
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()

# Improved model with thresholds
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(sp500, model, new_predictors)
print(predictions["Predictions"].value_counts())
print("Improved Precision:", precision_score(predictions["Target"], predictions["Predictions"]))
