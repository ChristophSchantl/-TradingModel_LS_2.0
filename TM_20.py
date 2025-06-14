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
# 1) Funktion: GA‐Optimierung + Post‐Selection + Backtest
# ---------------------------------------
def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    # Daten laden
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    data['Close'] = data['Close'].interpolate()
    data.dropna(inplace=True)
    prices = data['Close']

    # Fitness: nur Sharpe‐Ratio
    def evaluate_sharpe(ind):
        s, l = int(ind[0]), int(ind[1])
        if s >= l or s <= 0 or l <= 0:
            return (-np.inf,)
        ma_s = prices.rolling(window=s).mean()
        ma_l = prices.rolling(window=l).mean()
        df = pd.DataFrame({'Price': prices, 'MA_s': ma_s, 'MA_l': ma_l}).dropna()
        if df.empty:
            return (0.0,)
        wealth = [start_capital]
        pos = 0
        trade_price = 0.0
        for i in range(1, len(df)):
            p = df['Price'].iat[i]
            if df['MA_s'].iat[i] > df['MA_l'].iat[i] and df['MA_s'].iat[i-1] <= df['MA_l'].iat[i-1]:
                if pos == -1:
                    pnl = (trade_price - p) / trade_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                if pos != 1:
                    pos, trade_price = 1, p
                else:
                    wealth.append(wealth[-1])
                continue
            if df['MA_s'].iat[i] < df['MA_l'].iat[i] and df['MA_s'].iat[i-1] >= df['MA_l'].iat[i-1]:
                if pos == 1:
                    pnl = (p - trade_price) / trade_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                if pos != -1:
                    pos, trade_price = -1, p
                else:
                    wealth.append(wealth[-1])
                continue
            wealth.append(wealth[-1])
        rets = pd.Series(wealth).pct_change().dropna()
        if rets.std() == 0:
            return (-np.inf,)
        sharpe = (rets.mean()) / rets.std() * np.sqrt(252)
        return (sharpe,)

    # Fitness+Count: Sharpe + Anzahl Trades
    def evaluate_sharpe_count(ind):
        s, l = int(ind[0]), int(ind[1])
        if s >= l or s <= 0 or l <= 0:
            return -np.inf, 0
        ma_s = prices.rolling(window=s).mean()
        ma_l = prices.rolling(window=l).mean()
        df = pd.DataFrame({'Price': prices, 'MA_s': ma_s, 'MA_l': ma_l}).dropna()
        if df.empty:
            return 0.0, 0
        wealth = [start_capital]
        pos = 0
        trade_price = 0.0
        trades = 0
        for i in range(1, len(df)):
            p = df['Price'].iat[i]
            # Long‐Signal
            if df['MA_s'].iat[i] > df['MA_l'].iat[i] and df['MA_s'].iat[i-1] <= df['MA_l'].iat[i-1]:
                if pos == -1:
                    trades += 1
                    pnl = (trade_price - p) / trade_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                if pos != 1:
                    trades += 1
                    pos, trade_price = 1, p
                else:
                    wealth.append(wealth[-1])
                continue
            # Short‐Signal
            if df['MA_s'].iat[i] < df['MA_l'].iat[i] and df['MA_s'].iat[i-1] >= df['MA_l'].iat[i-1]:
                if pos == 1:
                    trades += 1
                    pnl = (p - trade_price) / trade_price * wealth[-1]
                    wealth.append(wealth[-1] + pnl)
                if pos != -1:
                    trades += 1
                    pos, trade_price = -1, p
                else:
                    wealth.append(wealth[-1])
                continue
            wealth.append(wealth[-1])
        # letzte Position schließen
        if pos != 0:
            trades += 1
            p = df['Price'].iat[-1]
            if pos == 1:
                pnl = (p - trade_price) / trade_price * wealth[-1]
            else:
                pnl = (trade_price - p) / trade_price * wealth[-1]
            wealth[-1] += pnl
        rets = pd.Series(wealth).pct_change().dropna()
        sharpe = 0.0 if rets.std() == 0 else (rets.mean()) / rets.std() * np.sqrt(252)
        return sharpe, trades

    # DEAP‐Setup (Single‐Objective Sharpe)
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 5, 200)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_int, toolbox.attr_int), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_sharpe)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    pop = toolbox.population(n=30)
    population, logbook = algorithms.eaSimple(pop, toolbox,
                                              cxpb=0.5, mutpb=0.2,
                                              ngen=15,
                                              stats=stats,
                                              halloffame=hof,
                                              verbose=False)

    # Post‐Selection: innerhalb ±1% Sharpe die mit wenigsten Trades wählen
    best_sh = max(ind.fitness.values[0] for ind in population)
    eps = 0.01 * best_sh
    candidates = [ind for ind in population if ind.fitness.values[0] >= best_sh - eps]
    best_ind = min(candidates, key=lambda ind: evaluate_sharpe_count(ind)[1])

    # finale Fenster
    bs, bl = int(round(best_ind[0])), int(round(best_ind[1]))
    if bs >= bl:
        bs, bl = bl - 1, bs + 1

    # finaler Backtest
    data_vis = data.copy()
    data_vis['MA_s'] = data_vis['Close'].rolling(bs).mean()
    data_vis['MA_l'] = data_vis['Close'].rolling(bl).mean()
    df = data_vis.dropna()

    trades = []
    pos = 0
    entry_price = 0.0
    wealth = start_capital
    cum_pnl = 0.0
    positionswert = 0.0
    wealth_hist = []
    pos_hist = []

    for i in range(len(df)):
        price = df['Close'].iat[i]
        date_i = df.index[i]
        # Equity
        if pos == 1:
            units = positionswert / entry_price
            equity_val = units * price
        elif pos == -1:
            units = positionswert / entry_price
            equity_val = positionswert + (entry_price - price) * units
        else:
            equity_val = wealth
        wealth_hist.append(equity_val)
        pos_hist.append(pos)

        ms_t, ml_t = df['MA_s'].iat[i], df['MA_l'].iat[i]
        ms_y = df['MA_s'].iat[i-1] if i>0 else 0
        ml_y = df['MA_l'].iat[i-1] if i>0 else 0

        # Kauf
        if ms_t > ml_t and ms_y <= ml_y and pos == 0:
            entry_price = price
            pos = 1
            positionswert = wealth
            wealth -= positionswert
            trades.append({'Typ':'Kauf','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':positionswert,
                           'Profit/Loss':None,'Kumulative P&L':cum_pnl})
        # Verkauf Long
        if ms_t < ml_t and ms_y >= ml_y and pos == 1:
            pos = 0
            gross = (price - entry_price)/entry_price * positionswert
            cum_pnl += gross
            wealth += positionswert + gross
            trades.append({'Typ':'Verkauf','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':None,
                           'Profit/Loss':gross,'Kumulative P&L':cum_pnl})
        # Short
        if ms_t < ml_t and ms_y >= ml_y and pos == 0:
            entry_price = price
            pos = -1
            positionswert = wealth
            wealth -= positionswert
            trades.append({'Typ':'Short','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':positionswert,
                           'Profit/Loss':None,'Kumulative P&L':cum_pnl})
        # Short‐Cover → Long
        if ms_t > ml_t and ms_y <= ml_y and pos == -1:
            gross = (entry_price - price)/entry_price * positionswert
            cum_pnl += gross
            wealth += positionswert + gross
            trades.append({'Typ':'Cover','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':None,
                           'Profit/Loss':gross,'Kumulative P&L':cum_pnl})
            entry_price = price
            pos = 1
            positionswert = wealth
            wealth -= positionswert
            trades.append({'Typ':'Kauf','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':positionswert,
                           'Profit/Loss':None,'Kumulative P&L':cum_pnl})

    # offene Position schließen
    if pos != 0:
        last_p = df['Close'].iat[-1]
        last_d = df.index[-1]
        if pos == 1:
            gross = (last_p - entry_price)/entry_price * positionswert
        else:
            gross = (entry_price - last_p)/entry_price * positionswert
        cum_pnl += gross
        wealth += positionswert + gross
        wealth_hist[-1] = wealth
        trades.append({'Typ':'Schließen','Datum':last_d,'Kurs':last_p,
                       'Spesen':0,'Positionswert':None,
                       'Profit/Loss':gross,'Kumulative P&L':cum_pnl})

    trades_df = pd.DataFrame(trades)
    strat_ret = (wealth - start_capital)/start_capital * 100
    bh_ret = (data_vis['Close'].iat[-1] - data_vis['Close'].iat[0]) / data_vis['Close'].iat[0] * 100

    df_plot = pd.DataFrame({'Close': df['Close'], 'Position': pos_hist}, index=df.index)
    df_wealth = pd.DataFrame({'Datum': df.index, 'Wealth': wealth_hist})

    return {
        "trades_df": trades_df,
        "strategy_return": float(strat_ret),
        "buy_and_hold_return": float(bh_ret),
        "total_trades": len(trades_df),
        "long_trades": trades_df['Typ'].str.contains("Kauf").sum(),
        "short_trades": trades_df['Typ'].str.contains("Short").sum(),
        "pos_count": (trades_df['Profit/Loss']>0).sum(),
        "neg_count": (trades_df['Profit/Loss']<0).sum(),
        "pos_pct": None,
        "neg_pct": None,
        "pos_pnl": trades_df.loc[trades_df['Profit/Loss']>0,'Profit/Loss'].sum(),
        "neg_pnl": trades_df.loc[trades_df['Profit/Loss']<0,'Profit/Loss'].sum(),
        "total_pnl": trades_df['Profit/Loss'].dropna().sum(),
        "pos_perf": None,
        "neg_perf": None,
        "df_plot": df_plot,
        "df_wealth": df_wealth,
        "best_individual": (bs, bl),
        "logbook": logbook
    }


# ---------------------------------------
# 2) Streamlit‐App UI
# ---------------------------------------
st.title("✨ AI Quant Model")

st.markdown("""
Gib einen **Aktien-Ticker**, ein **Startdatum** und das **Startkapital** ein.
Das Modell optimiert per GA ein MA-Crossover-System auf risikoadjustierte Performance (Sharpe)
und wählt unter nahezu gleich guten Lösungen (±1 % Sharpe) das mit den wenigsten Trades.
""")

ticker = st.text_input("Ticker (z.B. AAPL):", value="AAPL")
start_dt = st.date_input("Startdatum:", value=date(2020,1,1), max_value=date.today())
cap = st.number_input("Startkapital (€):", min_value=1000.0, value=10000.0, step=500.0, format="%.2f")

if st.button("🔄 Starte Optimierung"):
    if not ticker.strip():
        st.error("Bitte gib einen gültigen Ticker ein.")
    else:
        with st.spinner("Optimiere..."):
            results = optimize_and_run(ticker.strip().upper(),
                                       start_dt.strftime("%Y-%m-%d"),
                                       float(cap))

        # 0. Optimierungsresultate
        bs, bl = results["best_individual"]
        st.subheader("0️⃣ Optimierungsergebnisse")
        st.markdown(f"- **Bestes Short MA-Fenster:** {bs} Tage")
        st.markdown(f"- **Bestes Long MA-Fenster:**  {bl} Tage")
        log = results["logbook"].select("max")
        df_log = pd.DataFrame({"Generation": range(len(log)), "Max Sharpe": log}).set_index("Generation")
        st.line_chart(df_log, use_container_width=True)

        # 1. Performance‐Vergleich
        st.subheader("1️⃣ Strategie vs. Buy & Hold")
        strat_ret, bh_ret = results["strategy_return"], results["buy_and_hold_return"]
        fig1, ax1 = plt.subplots()
        bars = ax1.bar(["Strategie","Buy & Hold"], [strat_ret, bh_ret], color=["#2ca02c","#777777"])
        for bar in bars:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     f"{bar.get_height():.2f}%", ha="center", va="bottom")
        ax1.set_ylabel("Rendite (%)"); ax1.grid(axis="y", linestyle="--", alpha=0.5)
        st.pyplot(fig1)

        # 2. Kurs + Phasen
        st.subheader("2️⃣ Kursdiagramm mit Phasen")
        dfp = results["df_plot"]
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.plot(dfp.index, dfp["Close"], color="black", lw=1)
        phases = dfp["Position"].values; dates = dfp.index
        curr, start = phases[0], dates[0]
        for i in range(1,len(dates)):
            if phases[i]!=curr:
                end = dates[i-1]
                col = "green" if curr==1 else "red" if curr==-1 else None
                if col: ax2.axvspan(start,end,color=col,alpha=0.2)
                curr, start = phases[i], dates[i]
        col = "green" if curr==1 else "red" if curr==-1 else None
        if col: ax2.axvspan(start,dates[-1],color=col,alpha=0.2)
        ax2.set_xlabel("Datum"); ax2.set_ylabel("Preis"); ax2.grid(True,linestyle="--",alpha=0.4)
        st.pyplot(fig2)

        # 3. Trade‐Tabelle
        st.subheader("3️⃣ Trade‐Details")
        trades_df = results["trades_df"]
        if trades_df.empty:
            st.write("Keine Trades ausgeführt.")
        else:
            tbl = trades_df.copy()
            tbl['Datum'] = tbl['Datum'].dt.strftime("%Y-%m-%d")
            st.dataframe(tbl, use_container_width=True)

        # 4. Handels‐Statistiken
        st.subheader("4️⃣ Handelsstatistiken")
        total, lt, st_ = results["total_trades"], results["long_trades"], results["short_trades"]
        pc, nc = results["pos_count"], results["neg_count"]
        pos_pct = results["pos_pct"] or (pc/(pc+nc)*100 if pc+nc>0 else 0)
        neg_pct = results["neg_pct"] or (nc/(pc+nc)*100 if pc+nc>0 else 0)
        ppnl, npnl, tpnL = results["pos_pnl"], results["neg_pnl"], results["total_pnl"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Einträge gesamt", total)
            st.metric("Long‐Einträge", lt)
            st.metric("Short‐Einträge", st_)
        with col2:
            st.metric("Positive Trades", pc)
            st.metric("Negative Trades", nc)
            st.metric("Positive (%)", f"{pos_pct:.2f}%")
        with col3:
            st.metric("Gesamt‐P&L", f"{tpnL:.2f} EUR")
            st.metric("P&L positiv", f"{ppnl:.2f} EUR")
            st.metric("P&L negativ", f"{npnl:.2f} EUR")

        # 5. Anzahl Trades (Balken)
        st.subheader("5️⃣ Anzahl der Trades")
        fig5, ax5 = plt.subplots()
        ax5.bar(["Gesamt","Long","Short"], [total, lt, st_], color=["#4c72b0","#55a868","#c44e52"], alpha=0.8)
        ax5.set_ylabel("Anzahl"); ax5.grid(axis="y",linestyle="--",alpha=0.5)
        st.pyplot(fig5)

        # 6. Equity‐Kurve
        st.subheader("6️⃣ Equity‐Kurve")
        dfw = results["df_wealth"]
        fig6, ax6 = plt.subplots(figsize=(10,4))
        ax6.plot(dfw["Datum"], dfw["Wealth"], label="Strategie", color="#2ca02c", lw=1.3)
        ax6.set_xlabel("Datum"); ax6.set_ylabel("Vermögen (€)"); ax6.grid(True,linestyle="--",alpha=0.4)
        st.pyplot(fig6)

        # 7. Normalized Price vs. Wealth
        st.subheader("7️⃣ Normiertes Chart")
        dfw2 = dfw.set_index("Datum").reindex(dfp.index, method="ffill")
        price0, wealth0 = dfp["Close"].iloc[0], dfw2["Wealth"].iloc[0]
        dfp["PriceNorm"] = dfp["Close"]/price0
        dfw2["WealthNorm"] = dfw2["Wealth"]/wealth0
        fig7, ax7 = plt.subplots(figsize=(10,4))
        ax7.plot(dfp.index, dfp["PriceNorm"], label="Preis normiert", lw=1, alpha=0.5)
        ax7.plot(dfw2.index, dfw2["WealthNorm"], label="Wealth normiert", lw=1.5, alpha=0.8)
        curr, start = phases[0], dates[0]
        for i in range(1,len(dates)):
            if phases[i]!=curr:
                end = dates[i-1]
                col = "green" if curr==1 else "red" if curr==-1 else None
                if col: ax7.axvspan(start,end,color=col,alpha=0.1)
                curr, start = phases[i], dates[i]
        col = "green" if curr==1 else "red" if curr==-1 else None
        if col: ax7.axvspan(start,dates[-1],color=col,alpha=0.1)
        ax7.set_xlabel("Datum"); ax7.set_ylabel("Normierter Wert")
        ax7.legend(); ax7.grid(True,linestyle="--",alpha=0.4)
        st.pyplot(fig7)
