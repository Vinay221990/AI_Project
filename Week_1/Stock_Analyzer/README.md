# Stock Analyzer

## How to run

From the project folder:

```bash
# 1. Create a virtual environment (first time only)
python3 -m venv .venv

# 2. Install dependencies (first time only)
.venv/bin/pip install -r requirements.txt

# 3. Start the app
.venv/bin/streamlit run app.py
```

Streamlit opens the app in your browser, usually at http://localhost:8501. If it doesn't open, paste that address into your browser. Stop the app with `Ctrl+C`.

On Windows, use `.venv\Scripts\pip` and `.venv\Scripts\streamlit` instead of `.venv/bin/...`.

---

## About the project

A demo stock portfolio analyzer built with [Streamlit](https://streamlit.io) and Plotly. It reads a CSV of trade lots and turns it into an interactive dashboard of realized and unrealized profit and loss. It runs locally and all the data is dummy data.

### Files

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app |
| `stock_trades.csv` | Dummy dataset: 126 trade lots across 46 companies |
| `requirements.txt` | Python dependencies (streamlit, plotly, pandas) |
| `.streamlit/config.toml` | Dark theme settings |

### Dataset columns

| Column | Meaning |
|--------|---------|
| Stock Name, Ticker, Sector | The company, its ticker symbol and its sector |
| Stock Units Bought | Units bought in this lot |
| Purchase Date | Date of purchase |
| Cost Price Per Unit | Price paid per unit |
| Units Sold | Units sold from this lot (blank if none) |
| Sold Date, Selling Price | Date and price per unit of the sale (blank if unsold) |
| Profit / Loss | Realized P/L: `Units Sold × (Selling Price − Cost Price)` |
| Current Price | Dummy current market price of the stock |

A lot can be fully sold, partly sold or unsold. Units not yet sold count as an open position and are valued at the Current Price to give unrealized P/L.

### What the app shows

**Sidebar filters:** stocks, sectors, purchase date range, "profitable trades only" and "open positions only". Every tab responds to them.

**Overview**
- KPI cards: realized and unrealized P/L, win rate, average holding period, total invested, open position value, best and worst trade
- Cumulative realized P/L over time
- Portfolio treemap (size = capital invested, color = return %)
- Open positions bar chart (green for profit, red for loss)
- P/L by stock

**Stock Explorer** (pick one stock)
- KPI cards: total, realized and unrealized P/L, win rate, portfolio rank, average cost basis and more
- Lot-by-lot P/L waterfall
- Price ladder showing every price the stock traded at
- Comparison against its sector and the whole portfolio
- Auto-generated quick facts and a lots table

**Trades**
- Monthly bar chart of profit, loss and open trades
- Full trades table with colored P/L
- Download of the filtered rows as CSV

**Insights**
- Top 5 and bottom 5 stocks
- Auto-generated text insights
- What-if slider: how total P/L changes if open positions move by a given percentage

### Using your own data

Use the **Upload your own trades CSV** box at the top of the sidebar. The whole app (every chart, KPI and filter) rebuilds from your file. Remove the file to go back to the sample data.

- The file needs the same columns as `stock_trades.csv`. The **Download sample CSV template** button in the sidebar gives you a ready-made example.
- `Profit / Loss` is optional. The app recalculates it from the units, cost and selling price.
- If the file is invalid (missing columns, bad dates, units sold above units bought), the sidebar shows what is wrong and keeps using the sample data.
- Dates should be in `YYYY-MM-DD` format.
