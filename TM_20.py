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

# ──────────────────────────────────────────────────────────────
# 1) Optimierungs- und Backtest-Funktion
# ──────────────────────────────────────────────────────────────
def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    # 1. Daten laden
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
    if data is None or data.empty:
        raise ValueError("Keine Kursdaten für diesen Ticker/Zeitraum.")
    
    # 2. Robust: Price-Serie ermitteln (Close, dann Adj Close, sonst erste num-Spalte)
    if 'Close' in data.columns:
        prices = data['Close']
    elif 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        num_cols = data.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            raise ValueError("Keine numerische Spalte für Preise gefunden.")
        prices = data[num_cols[0]]
    prices = prices.interpolate().dropna()
    if len(prices) < 2:
        raise ValueError("Zu wenige Datenpunkte.")
    
    # 3. Fitness: nur Sharpe-Ratio
    def evaluate_sharpe(ind):
        s, l = map(int, ind)
        if s >= l or s <= 0:
            return (-np.inf,)
        ma_s = prices.rolling(s).mean()
        ma_l = prices.rolling(l).mean()
        df = pd.concat([prices.rename('P'), ma_s.rename('MA_s'), ma_l.rename('MA_l')], axis=1).dropna()
        wealth = [start_capital]; pos = 0; entry = 0.0
        for i in range(1, len(df)):
            p = df['P'].iat[i]
            # Long-Entry
            if df['MA_s'].iat[i] > df['MA_l'].iat[i] and df['MA_s'].iat[i-1] <= df['MA_l'].iat[i-1]:
                if pos == -1:
                    pnl = (entry - p)/entry * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry = 1, p
                continue
            # Short-Entry
            if df['MA_s'].iat[i] < df['MA_l'].iat[i] and df['MA_s'].iat[i-1] >= df['MA_l'].iat[i-1]:
                if pos == 1:
                    pnl = (p - entry)/entry * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry = -1, p
                continue
            wealth.append(wealth[-1])
        rets = pd.Series(wealth).pct_change().dropna()
        if rets.std() == 0:
            return (0.0,)
        return (rets.mean()/rets.std()*np.sqrt(252),)
    
    # 4. Fitness+Count: Sharpe & Trades
    def evaluate_sharpe_count(ind):
        s, l = map(int, ind)
        if s >= l or s <= 0:
            return -np.inf, 0
        ma_s = prices.rolling(s).mean()
        ma_l = prices.rolling(l).mean()
        df = pd.concat([prices.rename('P'), ma_s.rename('MA_s'), ma_l.rename('MA_l')], axis=1).dropna()
        wealth = [start_capital]; pos = 0; entry = 0.0; trades = 0
        for i in range(1, len(df)):
            p = df['P'].iat[i]
            if df['MA_s'].iat[i] > df['MA_l'].iat[i] and df['MA_s'].iat[i-1] <= df['MA_l'].iat[i-1]:
                trades += (1 if pos != 1 else 0)
                if pos == -1:
                    pnl = (entry - p)/entry * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry = 1, p
                continue
            if df['MA_s'].iat[i] < df['MA_l'].iat[i] and df['MA_s'].iat[i-1] >= df['MA_l'].iat[i-1]:
                trades += (1 if pos != -1 else 0)
                if pos == 1:
                    pnl = (p - entry)/entry * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                pos, entry = -1, p
                continue
            wealth.append(wealth[-1])
        if pos != 0:
            trades += 1
            p = df['P'].iat[-1]
            pnl = ((entry - p)/entry if pos==-1 else (p - entry)/entry) * wealth[-1]
            wealth[-1] += pnl
        rets = pd.Series(wealth).pct_change().dropna()
        sharpe = 0.0 if rets.std()==0 else rets.mean()/rets.std()*np.sqrt(252)
        return sharpe, trades

    # 5. DEAP Setup
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_s", random.randint, 5, 50)
    toolbox.register("attr_l", random.randint, 10, 200)
    def init_ind():
        s = toolbox.attr_s(); l = toolbox.attr_l()
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

    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.5, mutpb=0.2,
        ngen=15,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    # 6. Post-Selection: Sharpe ≥99% & min Trades
    best_sh = hof[0].fitness.values[0]
    eps = 0.01 * best_sh
    cands = [ind for ind in pop if ind.fitness.values[0] >= best_sh - eps]
    best_ind = min(cands, key=lambda ind: evaluate_sharpe_count(ind)[1])
    bs, bl = map(int, best_ind)
    if bs >= bl:
        bs, bl = bl-1, bs+1

    # 7. Finaler Backtest mit (bs,bl)
    df_bt = pd.DataFrame({'Close': prices})
    df_bt['MA_s'] = df_bt['Close'].rolling(bs).mean()
    df_bt['MA_l'] = df_bt['Close'].rolling(bl).mean()
    df_bt.dropna(inplace=True)

    position = 0; entry = 0.0; capital = start_capital; cum_pnl = 0.0
    trades = []; wealth_hist = []; pos_hist = []

    for i in range(len(df_bt)):
        p = df_bt['Close'].iat[i]; dt = df_bt.index[i]
        eq = (capital * p/entry) if position==1 else (capital*(2-p/entry)) if position==-1 else capital
        wealth_hist.append(eq); pos_hist.append(position)

        ms_t = df_bt['MA_s'].iat[i]; ml_t = df_bt['MA_l'].iat[i]
        ms_y = df_bt['MA_s'].iat[i-1] if i>0 else ms_t
        ml_y = df_bt['MA_l'].iat[i-1] if i>0 else ml_t

        # Entry/Exit
        if ms_t>ml_t and ms_y<=ml_y and position==0:
            position, entry = 1, p; trades.append({'Typ':'Entry Long','Datum':dt,'Preis':p})
        if ms_t<ml_t and ms_y>=ml_y and position==1:
            pnl = (p-entry)/entry*capital; cum_pnl+=pnl; capital+=pnl; position=0
            trades.append({'Typ':'Exit Long','Datum':dt,'Preis':p,'P&L':pnl})
        if ms_t<ml_t and ms_y>=ml_y and position==0:
            position, entry = -1, p; trades.append({'Typ':'Entry Short','Datum':dt,'Preis':p})
        if ms_t>ml_t and ms_y<=ml_y and position==-1:
            pnl = (entry-p)/entry*capital; cum_pnl+=pnl; capital+=pnl; position=0
            trades.append({'Typ':'Exit Short','Datum':dt,'Preis':p,'P&L':pnl})

    # letzte Position
    if position!=0:
        p = df_bt['Close'].iat[-1]
        pnl = ((entry-p)/entry if position==-1 else (p-entry)/entry)*capital
        capital+=pnl; cum_pnl+=pnl; trades.append({'Typ':'Final Exit','Datum':df_bt.index[-1],'Preis':p,'P&L':pnl})
        wealth_hist[-1] = capital

    trades_df = pd.DataFrame(trades)
    strat_ret = (capital - start_capital)/start_capital*100
    bh_ret    = (prices.iloc[-1]/prices.iloc[0] -1)*100

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


# ──────────────────────────────────────────────────────────────
# 2) Streamlit-UI
# ──────────────────────────────────────────────────────────────
st.title("✨ MA-Crossover GA-Optimierung")

st.markdown("""
Wähle einen **Ticker**, ein **Startdatum** und das **Startkapital**.  
Der GA findet das **Short/Long MA-Fenster** mit maximaler Sharpe-Ratio und wählt anschließend unter den Top-Kandidaten das mit den wenigsten Trades aus.
""")

ticker  = st.text_input("Ticker (z.B. AAPL):", "AAPL").strip().upper()
start_dt= st.date_input("Startdatum:", date(2022,1,1), max_value=date.today())
capital = st.number_input("Startkapital (€):", min_value=1000.0, value=10000.0, step=500.0)

if st.button("🔄 Optimierung starten"):
    with st.spinner("Optimiere MA-Fenster…"):
        try:
            res = optimize_and_run(ticker, start_dt.strftime("%Y-%m-%d"), capital)
        except Exception as e:
            st.error(str(e))
            st.stop()

    # 0️⃣ Optimierungsergebnisse
    bs, bl = res["best_individual"]
    st.subheader("0. Optimierungsergebnisse")
    st.markdown(f"- **Short MA:** {bs} Tage  \n- **Long MA:** {bl} Tage")
    df_log = pd.DataFrame({"Max Sharpe": res["logbook"].select("max")})
    st.line_chart(df_log, use_container_width=True)

    # 1️⃣ Performance-Vergleich
    st.subheader("1. Performance vs. Buy & Hold")
    strat_ret = res["strategy_return"]; bh_ret = res["buy_and_hold_return"]
    fig, ax = plt.subplots()
    bars = ax.bar(["Strategie","Buy & Hold"], [strat_ret, bh_ret],
                  color=["#2ca02c","#555555"], alpha=0.7)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{bar.get_height():.2f}%",
                ha='center', va='bottom')
    ax.set_ylabel("Rendite (%)"); ax.grid(axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)

    # 2️⃣ Equity-Kurve & Phasen
    st.subheader("2. Equity-Kurve & Phasen")
    df_plot = res["df_plot"]; df_wealth = res["df_wealth"]
    fig2, ax2 = plt.subplots(figsize=(10,5))
    axw = ax2.twinx()
    ax2.plot(df_plot.index, df_plot["Close"], color="black", label="Preis")
    axw.plot(df_wealth["Datum"], df_wealth["Wealth"], color="green", label="Wealth")
    pos = df_plot["Position"].values; dates = df_plot.index
    cur, start = pos[0], dates[0]
    for i in range(1,len(pos)):
        if pos[i]!=cur:
            ax2.axvspan(start, dates[i-1], color="green" if cur==1 else "red", alpha=0.2)
            cur, start = pos[i], dates[i]
    ax2.axvspan(start, dates[-1], color="green" if cur==1 else "red", alpha=0.2)
    ax2.set_xlabel("Datum"); ax2.set_ylabel("Preis")
    axw.set_ylabel("Wealth")
    h1,l1 = ax2.get_legend_handles_labels(); h2,l2 = axw.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, loc="upper left")
    st.pyplot(fig2)

    # 3️⃣ Trade-Tabelle
    st.subheader("3. Trades")
    td = res["trades_df"]
    if td.empty:
        st.write("Keine Trades ausgeführt.")
    else:
        # Datum formatieren
        if pd.api.types.is_datetime64_any_dtype(td["Datum"]):
            td["Datum"] = td["Datum"].dt.strftime("%Y-%m-%d")
        st.dataframe(td, use_container_width=True)
