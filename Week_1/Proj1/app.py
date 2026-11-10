from datetime import date
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")

GREEN, RED, BLUE = "#2ecc71", "#e74c3c", "#4da3ff"
TEMPLATE = "plotly_dark"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {background:#1b2030; border:1px solid #2a3147; border-radius:12px; padding:14px 16px;}
    [data-testid="stMetricLabel"] p {opacity:.7;}
    .insight {background:#1b2030; border-left:4px solid #4da3ff; border-radius:8px; padding:12px 16px; margin-bottom:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
REQUIRED = ["Stock Name", "Ticker", "Sector", "Stock Units Bought", "Purchase Date", "Cost Price Per Unit",
            "Units Sold", "Sold Date", "Selling Price", "Current Price"]
DEFAULT_CSV = "stock_trades.csv"


@st.cache_data
def load(data: bytes) -> pd.DataFrame:
    """Parse CSV bytes into the analysis frame. Raises ValueError with a friendly message if invalid."""
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        raise ValueError(f"Could not read the file as CSV ({e}).")
    df.columns = df.columns.str.strip()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))
    if df.empty:
        raise ValueError("The file has no rows.")
    for c in ["Purchase Date", "Sold Date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["Stock Units Bought", "Cost Price Per Unit", "Units Sold", "Selling Price", "Current Price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[["Stock Name", "Stock Units Bought", "Purchase Date", "Cost Price Per Unit", "Current Price"]].isna().any(axis=1)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} row(s) have a missing or invalid Stock Name, Units Bought, "
                         f"Purchase Date, Cost Price or Current Price (e.g. row {int(bad.idxmax()) + 2}).")
    df["Sector"] = df["Sector"].fillna("Unknown")
    df["Ticker"] = df["Ticker"].fillna(df["Stock Name"])
    sold = df["Units Sold"].fillna(0)
    if (sold > 0).any() and df.loc[sold > 0, ["Sold Date", "Selling Price"]].isna().any().any():
        raise ValueError("Rows with Units Sold must also have a Sold Date and Selling Price.")
    if (sold > df["Stock Units Bought"]).any():
        raise ValueError("Units Sold cannot exceed Stock Units Bought.")
    df = df.rename(columns={
        "Stock Name": "Stock", "Stock Units Bought": "Bought", "Cost Price Per Unit": "Cost",
        "Units Sold": "Sold", "Selling Price": "SellPrice", "Profit / Loss": "Realized",
        "Current Price": "Current",
    })
    df["Sold"] = df["Sold"].fillna(0)
    df["Realized"] = df["Sold"] * (df["SellPrice"].fillna(0) - df["Cost"]).where(df["Sold"] > 0, 0)
    df["Remaining"] = df["Bought"] - df["Sold"]
    df["Invested"] = df["Bought"] * df["Cost"]
    df["IsClosed"] = df["Sold"] > 0
    df["IsOpen"] = df["Remaining"] > 0
    df["ReturnPct"] = (df["SellPrice"] / df["Cost"] - 1) * 100
    df["HoldDays"] = (df["Sold Date"] - df["Purchase Date"]).dt.days
    df["Unrealized"] = df["Remaining"] * (df["Current"] - df["Cost"])
    df["OpenValue"] = df["Remaining"] * df["Current"]
    df["TotalPL"] = df["Realized"] + df["Unrealized"]
    df["LotId"] = df.index + 1
    return df


def money(v: float) -> str:
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def signed(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):,.0f}"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📂 Data")
uploaded = st.sidebar.file_uploader("Upload your own trades CSV", type="csv",
                                    help="Must use the same columns as the sample file.")
default_bytes = open(DEFAULT_CSV, "rb").read()
source_name = "sample data (stock_trades.csv)"
raw = None
if uploaded is not None:
    try:
        raw = load(uploaded.getvalue())
        source_name = uploaded.name
    except ValueError as e:
        st.sidebar.error(f"{e} Showing the sample data instead.")
if raw is None:
    raw = load(default_bytes)
st.sidebar.download_button("⬇️ Download sample CSV template", default_bytes, "stock_trades_sample.csv", "text/csv")
st.sidebar.caption(f"Using: **{source_name}** · {len(raw)} lots")
st.sidebar.divider()

st.sidebar.title("📈 Filters")
stocks = st.sidebar.multiselect("Stocks", sorted(raw["Stock"].unique()))
sectors = st.sidebar.multiselect("Sectors", sorted(raw["Sector"].unique()))
dmin, dmax = raw["Purchase Date"].min().date(), raw["Purchase Date"].max().date()
drange = st.sidebar.date_input("Purchase date range", (dmin, dmax), min_value=dmin, max_value=dmax)
profitable_only = st.sidebar.toggle("Profitable trades only")
open_only = st.sidebar.toggle("Open positions only")
st.sidebar.caption("Leave Stocks/Sectors empty to include all.")

df = raw.copy()
if stocks:
    df = df[df["Stock"].isin(stocks)]
if sectors:
    df = df[df["Sector"].isin(sectors)]
if isinstance(drange, tuple) and len(drange) == 2:
    df = df[(df["Purchase Date"].dt.date >= drange[0]) & (df["Purchase Date"].dt.date <= drange[1])]
if profitable_only:
    df = df[df["Realized"] > 0]
if open_only:
    df = df[df["IsOpen"]]

st.title("Stock Analyzer")
st.caption("Realized and unrealized performance across your trade lots (demo data).")

if df.empty:
    st.warning("No trades match the current filters.")
    st.stop()

closed = df[df["IsClosed"]]
opened = df[df["IsOpen"]]

tab_overview, tab_stock, tab_trades, tab_insights = st.tabs(
    ["📊 Overview", "🔎 Stock Explorer", "🧾 Trades", "💡 Insights"]
)

# ---------------------------------------------------------------- overview
with tab_overview:
    realized = closed["Realized"].sum()
    unrealized = opened["Unrealized"].sum()
    win_rate = (closed["Realized"] > 0).mean() * 100 if len(closed) else 0
    avg_hold = closed["HoldDays"].mean() if len(closed) else 0

    c = st.columns(4)
    c[0].metric("Realized P/L", money(realized), signed(realized), delta_color="normal")
    c[1].metric("Unrealized P/L", money(unrealized), signed(unrealized), delta_color="normal")
    c[2].metric("Win rate", f"{win_rate:.0f}%", f"{len(closed)} closed lots", delta_color="off")
    c[3].metric("Avg holding period", f"{avg_hold:.0f} days")

    c = st.columns(4)
    c[0].metric("Total invested", money(df["Invested"].sum()))
    c[1].metric("Open position value", money(opened["OpenValue"].sum()))
    if len(closed):
        best, worst = closed.loc[closed["Realized"].idxmax()], closed.loc[closed["Realized"].idxmin()]
        c[2].metric(f"Best trade · {best['Ticker']}", money(best["Realized"]))
        c[3].metric(f"Worst trade · {worst['Ticker']}", money(worst["Realized"]))

    st.subheader("Cumulative realized P/L")
    if len(closed):
        cum = closed.sort_values("Sold Date").copy()
        cum["Cumulative"] = cum["Realized"].cumsum()
        fig = go.Figure(go.Scatter(
            x=cum["Sold Date"], y=cum["Cumulative"], mode="lines+markers", fill="tozeroy",
            line=dict(color=BLUE, width=2), marker=dict(size=5),
            customdata=cum[["Stock", "Realized"]],
            hovertemplate="<b>%{customdata[0]}</b><br>Trade: $%{customdata[1]:,.0f}<br>Cumulative: $%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(template=TEMPLATE, height=360, margin=dict(t=10), yaxis_title="USD", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades in this selection.")

    left, right = st.columns(2)
    with left:
        st.subheader("Portfolio treemap")
        tm = df.groupby(["Sector", "Stock"], as_index=False).agg(
            Invested=("Invested", "sum"), TotalPL=("TotalPL", "sum"))
        tm["Return %"] = tm["TotalPL"] / tm["Invested"] * 100
        fig = px.treemap(tm, path=[px.Constant("Portfolio"), "Sector", "Stock"], values="Invested",
                         color="Return %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                         range_color=(-40, 40))
        fig.update_layout(template=TEMPLATE, height=520, margin=dict(t=10, l=0, r=0, b=0))
        st.caption("Size = capital invested · Color = total return % (realized + unrealized)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Open positions")
        op = opened.groupby("Stock", as_index=False).agg(
            OpenValue=("OpenValue", "sum"), Unrealized=("Unrealized", "sum")).sort_values("OpenValue")
        if len(op):
            fig = px.bar(op, x="OpenValue", y="Stock", orientation="h",
                         color=op["Unrealized"] > 0, color_discrete_map={True: GREEN, False: RED},
                         hover_data={"Unrealized": ":,.0f"})
            fig.update_layout(template=TEMPLATE, height=520, showlegend=False, margin=dict(t=10),
                              xaxis_title="Open position value (USD)", yaxis_title=None)
            st.caption("Bar length = current value · Color = unrealized P/L (green profit, red loss)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No open positions in this selection.")

    st.subheader("P/L by stock")
    by_stock = closed.groupby("Stock", as_index=False)["Realized"].sum().sort_values("Realized", ascending=False)
    if len(by_stock):
        fig = px.bar(by_stock, x="Stock", y="Realized",
                     color=by_stock["Realized"] > 0, color_discrete_map={True: GREEN, False: RED})
        fig.update_layout(template=TEMPLATE, height=420, showlegend=False, margin=dict(t=10),
                          yaxis_title="Realized P/L (USD)", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------- stock explorer
with tab_stock:
    choice = st.selectbox("Choose a stock", sorted(df["Stock"].unique()))
    s = df[df["Stock"] == choice].sort_values("Purchase Date")
    sc = s[s["IsClosed"]]
    so = s[s["IsOpen"]]
    info = s.iloc[0]
    cur = info["Current"]
    st.markdown(f"### {choice} · `{info['Ticker']}` · {info['Sector']}")

    # --- derived facts
    invested = s["Invested"].sum()
    total_pl = s["TotalPL"].sum()
    ret_pct = total_pl / invested * 100
    avg_cost = (s["Bought"] * s["Cost"]).sum() / s["Bought"].sum()
    wins = (sc["Realized"] > 0).mean() * 100 if len(sc) else None
    rank_series = df.groupby("Stock")["TotalPL"].sum().rank(ascending=False, method="min")
    rank, n_stocks = int(rank_series[choice]), len(rank_series)
    capital_share = invested / df["Invested"].sum() * 100
    sector_df = df[df["Sector"] == info["Sector"]]
    sector_ret = sector_df["TotalPL"].sum() / sector_df["Invested"].sum() * 100
    port_ret = df["TotalPL"].sum() / df["Invested"].sum() * 100

    c = st.columns(5)
    c[0].metric("Total P/L", money(total_pl), f"{ret_pct:+.1f}% on invested")
    c[1].metric("Realized P/L", money(s["Realized"].sum()))
    c[2].metric("Unrealized P/L", money(s["Unrealized"].sum()))
    c[3].metric("Win rate", f"{wins:.0f}%" if wins is not None else "n/a", f"{len(sc)} closed lots", delta_color="off")
    c[4].metric("Portfolio rank", f"#{rank} of {n_stocks}", "by total P/L", delta_color="off")

    c = st.columns(5)
    c[0].metric("Current price", f"${cur:,.2f}", f"{(cur / avg_cost - 1) * 100:+.1f}% vs avg cost")
    c[1].metric("Avg cost basis", f"${avg_cost:,.2f}")
    c[2].metric("Units held", f"{int(s['Remaining'].sum()):,}", f"of {int(s['Bought'].sum()):,} bought", delta_color="off")
    c[3].metric("Capital invested", money(invested), f"{capital_share:.1f}% of portfolio", delta_color="off")
    c[4].metric("Avg holding period", f"{sc['HoldDays'].mean():.0f} days" if len(sc) else "n/a")

    left, right = st.columns(2)
    with left:
        st.subheader("P/L lot by lot")
        lots = s.copy()
        lots["Label"] = "Lot " + lots["LotId"].astype(str) + "<br>" + lots["Purchase Date"].dt.strftime("%b %y")
        fig = go.Figure(go.Waterfall(
            x=list(lots["Label"]) + ["Total"], y=list(lots["TotalPL"]) + [0],
            measure=["relative"] * len(lots) + ["total"],
            increasing=dict(marker=dict(color=GREEN)), decreasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color="gray", dash="dot")),
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>"))
        fig.update_layout(template=TEMPLATE, height=380, margin=dict(t=10), yaxis_title="Total P/L per lot (USD)",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each bar is one purchase lot (realized + unrealized); the blue bar is the running total.")
    with right:
        st.subheader("Price ladder")
        marks = {"Lowest buy": (s["Cost"].min(), BLUE), "Avg cost": (avg_cost, "#a78bfa"),
                 "Highest buy": (s["Cost"].max(), BLUE), "Current": (cur, "#f5a623")}
        if len(sc):
            marks["Best sell"] = (sc["SellPrice"].max(), GREEN)
            marks["Worst sell"] = (sc["SellPrice"].min(), RED)
        items = sorted(marks.items(), key=lambda kv: kv[1][0])
        fig = go.Figure()
        lo, hi = items[0][1][0], items[-1][1][0]
        fig.add_shape(type="line", x0=lo, x1=hi, y0=0, y1=0, line=dict(color="gray", width=3))
        for i, (name, (price, col)) in enumerate(items):
            fig.add_trace(go.Scatter(
                x=[price], y=[0], mode="markers+text", showlegend=False,
                marker=dict(size=18, color=col, line=dict(color="white", width=1)),
                text=[f"<b>{name}</b><br>${price:,.2f}"], textposition="top center" if i % 2 == 0 else "bottom center",
                hovertemplate=f"{name}: ${price:,.2f}<extra></extra>"))
        pad = (hi - lo) * 0.12 or 1
        fig.update_layout(template=TEMPLATE, height=380, margin=dict(t=10),
                          xaxis=dict(range=[lo - pad, hi + pad], title="Price (USD)"),
                          yaxis=dict(visible=False, range=[-1.2, 1.2]))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Every price this stock has traded at in your lots, on one line.")

    left, right = st.columns(2)
    with left:
        st.subheader("How it compares")
        cmp = pd.DataFrame({"Group": [choice, f"{info['Sector']} sector", "Whole portfolio"],
                            "Return %": [ret_pct, sector_ret, port_ret]})
        fig = px.bar(cmp, x="Group", y="Return %", color="Group", text=cmp["Return %"].map("{:+.1f}%".format),
                     color_discrete_sequence=[BLUE, "#a78bfa", "#8892a6"])
        fig.update_layout(template=TEMPLATE, height=320, showlegend=False, margin=dict(t=10), xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Where the money went")
        parts = pd.DataFrame({
            "Bucket": ["Realized proceeds", "Open position value"],
            "USD": [(sc["Sold"] * sc["SellPrice"]).sum(), so["OpenValue"].sum()]})
        fig = px.pie(parts, names="Bucket", values="USD", hole=0.55,
                     color_discrete_sequence=[GREEN, BLUE])
        fig.update_layout(template=TEMPLATE, height=320, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quick facts")
    facts = [
        f"First bought on <b>{s['Purchase Date'].min():%d %b %Y}</b> — "
        f"{(pd.Timestamp(date.today()) - s['Purchase Date'].min()).days} days ago, across <b>{len(s)}</b> lot(s).",
        f"Biggest lot by capital: <b>Lot {int(s.loc[s['Invested'].idxmax(), 'LotId'])}</b> "
        f"({money(s['Invested'].max())}).",
    ]
    best_lot, worst_lot = s.loc[s["TotalPL"].idxmax()], s.loc[s["TotalPL"].idxmin()]
    facts.append(f"Best lot: <b>Lot {int(best_lot['LotId'])}</b> ({signed(best_lot['TotalPL'])}). "
                 f"Weakest lot: <b>Lot {int(worst_lot['LotId'])}</b> ({signed(worst_lot['TotalPL'])}).")
    if len(sc):
        last = sc["Sold Date"].max()
        facts.append(f"Last sale was on <b>{last:%d %b %Y}</b>, {(pd.Timestamp(date.today()) - last).days} days ago. "
                     f"Best single sale return: <b>{sc['ReturnPct'].max():+.1f}%</b>.")
    if len(so):
        open_cost = (so["Remaining"] * so["Cost"]).sum() / so["Remaining"].sum()
        gap = (open_cost / cur - 1) * 100
        facts.append(
            f"Open units cost <b>${open_cost:,.2f}</b> on average. "
            + ("They are in profit." if cur >= open_cost else f"The price must rise <b>{gap:.1f}%</b> to break even."))
    else:
        facts.append("No open units — this position is fully closed.")
    facts.append(f"This stock is <b>{capital_share:.1f}%</b> of your invested capital and is "
                 f"{'beating' if ret_pct > sector_ret else 'trailing'} its sector by <b>{abs(ret_pct - sector_ret):.1f}</b> pts.")
    for f in facts:
        st.markdown(f"<div class='insight'>{f}</div>", unsafe_allow_html=True)

    st.subheader("Lots")
    lots_tbl = s[["LotId", "Bought", "Purchase Date", "Cost", "Sold", "Sold Date", "SellPrice", "Realized",
                  "Remaining", "Unrealized"]]
    st.dataframe(lots_tbl, hide_index=True, width="stretch")

# ---------------------------------------------------------------- trades
with tab_trades:
    st.subheader("Trades per month")
    tl = df.copy()
    tl["End"] = tl["Sold Date"].fillna(pd.Timestamp(date.today()))
    tl["Status"] = tl.apply(
        lambda r: "Open" if not r["IsClosed"] else ("Profit" if r["Realized"] > 0 else "Loss"), axis=1)
    tl["Label"] = tl["Ticker"] + " #" + tl["LotId"].astype(str)
    # closed lots are counted in the month they were sold, open lots in the month they were bought
    tl["Month"] = tl["Sold Date"].fillna(tl["Purchase Date"]).dt.to_period("M").dt.to_timestamp()
    monthly = (tl.groupby(["Month", "Status"]).agg(Trades=("LotId", "count"), Amount=("Realized", "sum"),
                                                   Invested=("Invested", "sum")).reset_index())
    monthly["Amount"] = monthly.apply(lambda r: r["Invested"] if r["Status"] == "Open" else r["Amount"], axis=1)
    fig = px.bar(monthly, x="Month", y="Trades", color="Status", barmode="group",
                 category_orders={"Status": ["Profit", "Loss", "Open"]},
                 color_discrete_map={"Profit": GREEN, "Loss": RED, "Open": BLUE},
                 custom_data=["Amount"])
    fig.update_traces(hovertemplate="%{x|%b %Y}<br>%{y} trades<br>$%{customdata[0]:,.0f}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.update_layout(template=TEMPLATE, height=420, margin=dict(t=10), yaxis_title="Number of trades",
                      xaxis_title=None, legend_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Profit and loss trades are counted in the month they were sold; open trades in the month they were bought. "
               "Hover for the dollar amount (realized P/L, or capital invested for open trades).")

    st.subheader("All trades")
    show = tl[["LotId", "Stock", "Ticker", "Sector", "Bought", "Purchase Date", "Cost", "Sold", "Sold Date",
               "SellPrice", "Realized", "Remaining", "Current", "Unrealized", "Status"]]
    colour = lambda v: f"color: {GREEN}" if v > 0 else (f"color: {RED}" if v < 0 else "")
    styled = (show.style.map(colour, subset=["Realized", "Unrealized"])
              .format({"Cost": "${:,.2f}", "SellPrice": "${:,.2f}", "Current": "${:,.2f}",
                       "Realized": "${:,.2f}", "Unrealized": "${:,.2f}",
                       "Purchase Date": "{:%Y-%m-%d}", "Sold Date": lambda v: "" if pd.isna(v) else f"{v:%Y-%m-%d}"}))
    st.dataframe(styled, hide_index=True, width="stretch", height=450)
    st.download_button("⬇️ Download filtered CSV", show.to_csv(index=False), "filtered_trades.csv", "text/csv")

# ---------------------------------------------------------------- insights
with tab_insights:
    stock_pl = df.groupby("Stock")["Realized"].sum().sort_values(ascending=False)
    left, right = st.columns(2)
    with left:
        st.subheader("🏆 Top 5 stocks")
        for i, (n, v) in enumerate(stock_pl.head(5).items()):
            st.metric(f"{['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i]} {n}", money(v))
    with right:
        st.subheader("📉 Bottom 5 stocks")
        for n, v in stock_pl.tail(5).iloc[::-1].items():
            st.metric(f"🔻 {n}", money(v))

    st.subheader("Auto-generated insights")
    notes = []
    if len(closed):
        sec = closed.groupby("Sector")["Realized"].sum().sort_values()
        notes.append(f"Your best sector is <b>{sec.index[-1]}</b> ({signed(sec.iloc[-1])} realized); "
                     f"the weakest is <b>{sec.index[0]}</b> ({signed(sec.iloc[0])}).")
        lh = closed.loc[closed["HoldDays"].idxmax()]
        notes.append(f"Longest closed hold: <b>{lh['Stock']}</b> for {int(lh['HoldDays'])} days "
                     f"({avg_hold:.0f} days is your average).")
        w, l = closed[closed["Realized"] > 0], closed[closed["Realized"] <= 0]
        if len(w) and len(l):
            notes.append(f"Winners average {w['ReturnPct'].mean():+.1f}% vs. losers at {l['ReturnPct'].mean():+.1f}% "
                         f"— winners were held {w['HoldDays'].mean():.0f} days vs. {l['HoldDays'].mean():.0f} for losers.")
        notes.append(f"You win {win_rate:.0f}% of closed lots across {closed['Stock'].nunique()} stocks.")
    if len(opened):
        top = opened.groupby("Stock")["Unrealized"].sum().sort_values()
        notes.append(f"Biggest open winner: <b>{top.index[-1]}</b> ({signed(top.iloc[-1])} unrealized). "
                     f"Biggest open drag: <b>{top.index[0]}</b> ({signed(top.iloc[0])}).")
    for n in notes:
        st.markdown(f"<div class='insight'>{n}</div>", unsafe_allow_html=True)

    st.subheader("What-if: market move on open positions")
    move = st.slider("Open positions move by", -40, 60, 0, format="%d%%")
    hypo = (opened["Remaining"] * (opened["Current"] * (1 + move / 100) - opened["Cost"])).sum()
    c = st.columns(3)
    c[0].metric("Realized P/L", money(realized))
    c[1].metric(f"Unrealized at {move:+d}%", money(hypo))
    c[2].metric("Total P/L", money(realized + hypo), signed(hypo - unrealized) + " vs. today")
