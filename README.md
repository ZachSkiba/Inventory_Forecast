# Credit Card Transactions Predictive Analytics
Predicting monthly user spending using time series forecasting.

## Dataset
Sparkov Synthetic Credit Card Transactions
Download fraudTrain.csv and fraudTest.csv from kaggle.com/datasets/kartik2112/fraud-detection
Place both files in data/raw/

## Setup
pip install -r requirements.txt

## Notebooks
- 01_eda.ipynb � data loading, cleaning, EDA, seasonal decomposition
- 02_sarima.ipynb � SARIMA model, forecast, evaluation
- 03_prophet.ipynb � Prophet model, forecast, comparison with SARIMA