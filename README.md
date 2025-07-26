# 🛒 Sales Forecasting Project

## 📌 Overview
This project focuses on building a robust machine learning pipeline to **predict daily item-level sales** for a retail chain. By leveraging historical sales, store metadata, transactions, oil prices, and holidays/events, we aim to forecast future sales and uncover patterns that influence retail performance.

---

## 📁 Datasets

| File Name              | Description |
|------------------------|-------------|
| `train.csv`            | Main training data containing date-wise item sales per store, now enriched with state info |
| `test.csv`             | Test data used for evaluating model predictions |
| `stores.csv`           | Metadata for each store (e.g., state, type, cluster) |
| `transactions.csv`     | Daily transaction count per store |
| `holidays_events.csv`  | Local/national holidays and special events (could impact sales) |
| `oil.csv`              | Daily oil prices, used as a macroeconomic feature |
| `sample_submission.csv`| Sample format for the prediction output |

---

## 🧑‍💻 Project Workflow

1. **Data Loading & Cleaning**
   - Merging datasets based on `store_nbr` and `date`
   - Handling missing values
   - Creating additional time-based features (month, weekday, etc.)

2. **Feature Engineering**
   - Incorporating holiday/event impact
   - Including oil price fluctuations
   - Analyzing transaction trends
   - State-wise and item-wise aggregations

3. **Visualization**
   - Sales trend analysis
   - State-wise sales distribution (pie/bar charts)
   - Seasonal patterns and anomalies

4. **Modeling**
   - Regression models (e.g., XGBoost, Random Forest)
   - Time-series forecasting (optional)
   - Evaluation using metrics like RMSE/MAE

---

## 📊 Example Visualization

```python
# Pie chart of sales by state
state_sales = train_df.groupby('state')['sales'].sum()
plt.figure(figsize=(10, 10))
plt.pie(state_sales, labels=state_sales.index, autopct='%1.1f%%', startangle=90)
plt.title("Sales Distribution by State")
plt.show()
