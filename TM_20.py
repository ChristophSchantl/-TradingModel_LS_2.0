import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, date
from deap import base, creator, tools, algorithms
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------
# 1) GA-Optimierung + Post-Selection + finaler Backtest
# ---------------------------------------
def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    # --- Daten laden ---
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    if data is None or data.empty or 'Close' not in data:
        raise ValueError("Keine Kursdaten für diesen Ticker/Zeitraum.")
    data['Close'] = data['Close'].interpolate()
    data.dropna(subset=['Close'], inplace=True)
    prices = data['Close']
    if len(prices) < 2:
        raise ValueError("Zu wenige Datenpunkte.")

    # --- Fitness: nur Sharpe-Ratio ---
    def evaluate_sharpe(ind):
        s, l = map(int, ind)
        # ungültige Parameter
        if s >= l or s <= 0:
            return (-np.inf,)
        # Rolling MAs
        ma_s = prices.rolling(s).mean()
        ma_l = prices.rolling(l).mean()
        # DataFrame via concat (kein Scalar-Fehler)
        df = pd.concat([
            prices.rename('Price'),
            ma_s.rename('MA_s'),
            ma_l.rename('MA_l')
        ], axis=1).dropna()
        if df.empty:
            return (0.0,)
        wealth = [start_capital]
        pos = 0
        entry_price = 0.0
        for i in range(1, len(df)):
            p = df['Price'].iat[i]
            # Kaufsignal
            if df['MA_s'].iat[i] > df['MA_l'].iat[i] and df['MA_s'].iat[i-1] <= df['MA_l'].iat[i-1]:
                if pos == -1:
                    pnl = (entry_price - p) / entry_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry_price = 1, p
                continue
            # Verkaufssignal
            if df['MA_s'].iat[i] < df['MA_l'].iat[i] and df['MA_s'].iat[i-1] >= df['MA_l'].iat[i-1]:
                if pos == 1:
                    pnl = (p - entry_price) / entry_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry_price = -1, p
                continue
            wealth.append(wealth[-1])
        rets = pd.Series(wealth).pct_change().dropna()
        if rets.std() == 0:
            return (0.0,)
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
        return (sharpe,)

    # --- Fitness+Count: Sharpe & Trade-Anzahl ---
    def evaluate_sharpe_count(ind):
        s, l = map(int, ind)
        if s >= l or s <= 0:
            return -np.inf, 0
        ma_s = prices.rolling(s).mean()
        ma_l = prices.rolling(l).mean()
        df = pd.concat([
            prices.rename('Price'),
            ma_s.rename('MA_s'),
            ma_l.rename('MA_l')
        ], axis=1).dropna()
        if df.empty:
            return 0.0, 0
        wealth = [start_capital]
        pos = 0
        entry_price = 0.0
        trades = 0
        for i in range(1, len(df)):
            p = df['Price'].iat[i]
            # Long-Entry
            if df['MA_s'].iat[i] > df['MA_l'].iat[i] and df['MA_s'].iat[i-1] <= df['MA_l'].iat[i-1]:
                trades += (1 if pos != 1 else 0)
                if pos == -1:
                    pnl = (entry_price - p) / entry_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry_price = 1, p
                continue
            # Short-Entry
            if df['MA_s'].iat[i] < df['MA_l'].iat[i] and df['MA_s'].iat[i-1] >= df['MA_l'].iat[i-1]:
                trades += (1 if pos != -1 else 0)
                if pos == 1:
                    pnl = (p - entry_price) / entry_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry_price = -1, p
                continue
            wealth.append(wealth[-1])
        # letzte Position schließen
        if pos != 0:
            trades += 1
            p = df['Price'].iat[-1]
            pnl = ((entry_price - p) / entry_price if pos==-1 else (p - entry_price) / entry_price) * wealth[-1]
            wealth[-1] += pnl
        rets = pd.Series(wealth).pct_change().dropna()
        sharpe = 0.0 if rets.std()==0 else rets.mean()/rets.std()*np.sqrt(252)
        return sharpe, trades

    # --- DEAP Setup ---
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_s", random.randint, 5, 50)
    toolbox.register("attr_l", random.randint, 10, 200)
    def init_ind():
        s = toolbox.attr_s()
        l = toolbox.attr_l()
        # sicherstellen s<l
        return creator.Individual([min(s, l-1), max(s+1, l)])
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_sharpe)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt,
                     low=[5,10], up=[50,200], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=30)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(1)

    # GA laufen lassen
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.5, mutpb=0.2,
        ngen=15,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    # --- Post-Selection: Sharpe ≥99% & min Trades ---
    best_sh = hof[0].fitness.values[0]
    eps = 0.01 * best_sh
    cands = [ind for ind in pop if ind.fitness.values[0] >= best_sh - eps]
    best_ind = min(cands, key=lambda ind: evaluate_sharpe_count(ind)[1])

    bs, bl = map(int, best_ind)
    if bs >= bl:
        bs, bl = bl-1, bs+1

    # --- Finaler Backtest mit (bs, bl) ---
    df_bt = pd.DataFrame({'Close': prices})
    df_bt['MA_s'] = df_bt['Close'].rolling(bs).mean()
    df_bt['MA_l'] = df_bt['Close'].rolling(bl).mean()
    df_bt.dropna(inplace=True)

    position = 0
    entry_price = 0.0
    capital = start_capital
    cum_pnl = 0.0
    trades = []
    wealth_hist = []
    pos_hist = []

    for i in range(len(df_bt)):
        price = df_bt['Close'].iat[i]
        date_i = df_bt.index[i]
        # Equity
        if position == 1:
            equity = capital * price / entry_price
        elif position == -1:
            equity = capital * (2 - price/entry_price)
        else:
            equity = capital
        wealth_hist.append(equity)
        pos_hist.append(position)

        ms_t = df_bt['MA_s'].iat[i]
        ml_t = df_bt['MA_l'].iat[i]
        ms_y = df_bt['MA_s'].iat[i-1] if i>0 else ms_t
        ml_y = df_bt['MA_l'].iat[i-1] if i>0 else ml_t

        # Entry Long
        if ms_t > ml_t and ms_y <= ml_y and position==0:
            position = 1
            entry_price = price
            trades.append({'Typ':'Entry Long','Datum':date_i,'Preis':price})
        # Exit Long
        if ms_t < ml_t and ms_y >= ml_y and position==1:
            pnl = (price - entry_price)/entry_price * capital
            cum_pnl += pnl
            capital += pnl
            position = 0
            trades.append({'Typ':'Exit Long','Datum':date_i,'Preis':price,'P&L':pnl})
        # Entry Short
        if ms_t < ml_t and ms_y >= ml_y and position==0:
            position = -1
            entry_price = price
            trades.append({'Typ':'Entry Short','Datum':date_i,'Preis':price})
        # Exit Short
        if ms_t > ml_t and ms_y <= ml_y and position==-1:
            pnl = (entry_price - price)/entry_price * capital
            cum_pnl += pnl
            capital += pnl
            position = 0
            trades.append({'Typ':'Exit Short','Datum':date_i,'Preis':price,'P&L':pnl})

    # letzte Position schließen
    if position != 0:
        price = df_bt['Close'].iat[-1]
        pnl = ((entry_price - price)/entry_price if position==-1 else (price - entry_price)/entry_price) * capital
        capital += pnl
        cum_pnl += pnl
        trades.append({'Typ':'Final Exit','Datum':df_bt.index[-1],'Preis':price,'P&L':pnl})
        wealth_hist[-1] = capital

    trades_df = pd.DataFrame(trades)
    strat_ret = (capital - start_capital) / start_capital * 100
    bh_ret    = (prices.iloc[-1]/prices.iloc[0] - 1) * 100

    df_plot   = pd.DataFrame({'Close':df_bt['Close'],'Position':pos_hist}, index=df_bt.index)
    df_wealth = pd.DataFrame({'Datum':df_bt.index,'Wealth':wealth_hist})

    return {
        "best_individual": (bs, bl),
        "logbook": logbook,
        "trades_df": trades_df,
        "strategy_return": strat_ret,
        "buy_and_hold_return": bh_ret,
        "df_plot": df_plot,
        "df_wealth": df_wealth
    }


# ---------------------------------------
# 2) Streamlit-UI
# ---------------------------------------
st.title("✨ MA-Crossover GA-Optimierung")

st.markdown("""
Wähle einen Ticker, ein Startdatum und das Startkapital.  
Der GA sucht die MA-Fenster mit maximaler Sharpe-Ratio und danach minimaler Trade-Anzahl.
""")

ticker = st.text_input("Ticker (z.B. AAPL):", "AAPL").strip().upper()
start_dt = st.date_input("Startdatum:", date(2022,1,1), max_value=date.today())
capital  = st.number_input("Startkapital (€):", min_value=1000.0, value=10000.0, step=500.0)

if st.button("🔄 Optimierung starten"):
    with st.spinner("Optimiere…"):
        try:
            res = optimize_and_run(ticker, start_dt.strftime("%Y-%m-%d"), capital)
        except Exception as e:
            st.error(f"Fehler: {e}")
            st.stop()

    # 0️⃣ Optimierungsergebnisse
    bs, bl = res["best_individual"]
    st.subheader("0. Optimierungsergebnisse")
    st.markdown(f"- **Short MA:** {bs} Tage  \n- **Long MA:** {bl} Tage")
    sharpe_gen = res["logbook"].select("max")
    df_log = pd.DataFrame({"Max Sharpe": sharpe_gen})
    st.line_chart(df_log, use_container_width=True)

    # 1️⃣ Performance vs. Buy & Hold
    strat_ret = res["strategy_return"]
    bh_ret    = res["buy_and_hold_return"]
    st.subheader("1. Performance-Vergleich")
    fig1, ax1 = plt.subplots()
    bars = ax1.bar(["Strategie","Buy & Hold"], [strat_ret, bh_ret],
                   color=["#2ca02c","#555555"], alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2, h, f"{h:.2f}%", ha='center', va='bottom')
    ax1.set_ylabel("Rendite (%)"); ax1.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig1)

    # 2️⃣ Equity-Kurve mit Phasen
    st.subheader("2. Equity-Kurve & Phasen")
    df_plot   = res["df_plot"]
    df_wealth = res["df_wealth"]
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax_w = ax2.twinx()
    ax2.plot(df_plot.index, df_plot["Close"], label="Preis", color="black")
    ax_w.plot(df_wealth["Datum"], df_wealth["Wealth"],
             label="Wealth", color="green")
    # Phasen-Overlay
    pos = df_plot["Position"].values
    dates = df_plot.index
    cur, start = pos[0], dates[0]
    for i in range(1,len(pos)):
        if pos[i] != cur:
            end = dates[i-1]
            col = "green" if cur==1 else "red"
            ax2.axvspan(start, end, color=col, alpha=0.2)
            cur, start = pos[i], dates[i]
    # letzte Phase
    col = "green" if cur==1 else "red"
    ax2.axvspan(start, dates[-1], color=col, alpha=0.2)
    ax2.set_xlabel("Datum"); ax2.set_ylabel("Preis")
    ax_w.set_ylabel("Wealth (€)")
    l1,l2 = ax2.get_legend_handles_labels()
    l3,l4 = ax_w.get_legend_handles_labels()
    ax2.legend(l1+l3, l2+l4, loc="upper left")
    st.pyplot(fig2)

    # 3️⃣ Trade-Tabelle
    st.subheader("3. Trades")
    td = res["trades_df"]
    if td.empty:
        st.write("Keine Trades ausgeführt.")
    else:
        # Datum schön formatieren
        if pd.api.types.is_datetime64_any_dtype(td["Datum"]):
            td["Datum"] = td["Datum"].dt.strftime("%Y-%m-%d")
        st.dataframe(td, use_container_width=True)
