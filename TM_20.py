import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date
from deap import base, creator, tools, algorithms
import random
import warnings

warnings.filterwarnings("ignore")

def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    # 1) Lade und bereite Daten vor
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    data['Close'] = data['Close'].interpolate()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # 2a) Fitness-Funktion: nur Sharpe
    def evaluate_sharpe(ind):
        ma_s, ma_l = int(ind[0]), int(ind[1])
        if ma_s >= ma_l or ma_s <= 0 or ma_l <= 0:
            return (-np.inf,)
        df = data.copy()
        df['MA_s'] = df['Close'].rolling(ma_s).mean()
        df['MA_l'] = df['Close'].rolling(ma_l).mean()
        df.dropna(inplace=True)

        pos = 0
        trade_price = 0.0
        wealth_curve = [start_capital]

        for i in range(1, len(df)):
            price = df['Close'].iat[i]
            ms_t, ml_t = df['MA_s'].iat[i], df['MA_l'].iat[i]
            ms_y, ml_y = df['MA_s'].iat[i-1], df['MA_l'].iat[i-1]
            # Long-Entry
            if ms_t > ml_t and ms_y <= ml_y:
                if pos == -1:
                    pnl = (trade_price - price)/trade_price * wealth_curve[-1]
                    wealth_curve.append(wealth_curve[-1] + pnl)
                if pos != 1:
                    pos, trade_price = 1, price
                else:
                    wealth_curve.append(wealth_curve[-1])
                continue
            # Short-Entry
            if ms_t < ml_t and ms_y >= ml_y:
                if pos == 1:
                    pnl = (price - trade_price)/trade_price * wealth_curve[-1]
                    wealth_curve.append(wealth_curve[-1] + pnl)
                if pos != -1:
                    pos, trade_price = -1, price
                else:
                    wealth_curve.append(wealth_curve[-1])
                continue
            # hold
            wealth_curve.append(wealth_curve[-1])

        ret = pd.Series(wealth_curve).pct_change().dropna()
        if ret.std() == 0:
            return (-np.inf,)
        sharpe = (ret.mean() - 0.02/252) / ret.std() * np.sqrt(252)
        return (sharpe,)

    # 2b) Fitness+Count: Sharpe & Trade-Anzahl
    def evaluate_sharpe_count(ind):
        ma_s, ma_l = int(ind[0]), int(ind[1])
        if ma_s >= ma_l or ma_s <= 0 or ma_l <= 0:
            return -np.inf, 0
        df = data.copy()
        df['MA_s'] = df['Close'].rolling(ma_s).mean()
        df['MA_l'] = df['Close'].rolling(ma_l).mean()
        df.dropna(inplace=True)

        pos = 0
        trade_price = 0.0
        trade_count = 0
        wealth_curve = [start_capital]

        for i in range(1, len(df)):
            price = df['Close'].iat[i]
            ms_t, ml_t = df['MA_s'].iat[i], df['MA_l'].iat[i]
            ms_y, ml_y = df['MA_s'].iat[i-1], df['MA_l'].iat[i-1]
            # Long-Entry
            if ms_t > ml_t and ms_y <= ml_y:
                if pos == -1:
                    trade_count += 1
                    pnl = (trade_price - price)/trade_price * wealth_curve[-1]
                    wealth_curve.append(wealth_curve[-1] + pnl)
                if pos != 1:
                    trade_count += 1
                    pos, trade_price = 1, price
                else:
                    wealth_curve.append(wealth_curve[-1])
                continue
            # Short-Entry
            if ms_t < ml_t and ms_y >= ml_y:
                if pos == 1:
                    trade_count += 1
                    pnl = (price - trade_price)/trade_price * wealth_curve[-1]
                    wealth_curve.append(wealth_curve[-1] + pnl)
                if pos != -1:
                    trade_count += 1
                    pos, trade_price = -1, price
                else:
                    wealth_curve.append(wealth_curve[-1])
                continue
            # hold
            wealth_curve.append(wealth_curve[-1])

        # letzte Position schließen
        if pos != 0:
            trade_count += 1
            last_price = df['Close'].iat[-1]
            if pos == 1:
                pnl = (last_price - trade_price)/trade_price * wealth_curve[-1]
            else:
                pnl = (trade_price - last_price)/trade_price * wealth_curve[-1]
            wealth_curve[-1] += pnl

        ret = pd.Series(wealth_curve).pct_change().dropna()
        if ret.std() == 0:
            sharpe = -np.inf
        else:
            sharpe = (ret.mean() - 0.02/252) / ret.std() * np.sqrt(252)
        return sharpe, trade_count

    # 3) DEAP-Setup
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 5, 50)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_int, toolbox.attr_int), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_sharpe)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    pop = toolbox.population(n=20)
    population, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.5, mutpb=0.2,
        ngen=10, stats=stats,
        halloffame=hof,
        verbose=False
    )

    # 4) Post-Selection: Top 1% Sharpe und dann minimal Trades
    best_sh = max(ind.fitness.values[0] for ind in population)
    eps = 0.01 * best_sh
    candidates = [ind for ind in population if ind.fitness.values[0] >= best_sh - eps]
    best_ind = min(candidates, key=lambda ind: evaluate_sharpe_count(ind)[1])

    # 5) Final backtest mit gerundeten MA
    bs, bl = int(round(best_ind[0])), int(round(best_ind[1]))
    if bs >= bl:
        bs, bl = bl - 1, bs + 1

    df_vis = data.copy()
    df_vis['MA_short'] = df_vis['Close'].rolling(bs).mean()
    df_vis['MA_long']  = df_vis['Close'].rolling(bl).mean()
    df_vis.dropna(inplace=True)

    position = 0
    initial_price = 0.0
    wealth = start_capital
    cum_pnl = 0.0
    trades = []
    wealth_hist = []
    pos_hist = []

    for i in range(len(df_vis)):
        price = df_vis['Close'].iat[i]
        date_i = df_vis.index[i]
        # Equity
        if position == 1:
            units = positionswert / initial_price
            equity = units * price
        elif position == -1:
            units = positionswert / initial_price
            equity = positionswert + (initial_price - price) * units
        else:
            equity = wealth
        wealth_hist.append(equity)
        pos_hist.append(position)

        ms_t = df_vis['MA_short'].iat[i]
        ml_t = df_vis['MA_long'].iat[i]
        ms_y = df_vis['MA_short'].iat[i-1] if i>0 else 0
        ml_y = df_vis['MA_long'].iat[i-1] if i>0 else 0

        # Kaufsignal
        if ms_t > ml_t and ms_y <= ml_y and position == 0:
            initial_price = price
            position = 1
            positionswert = wealth
            wealth -= positionswert
            trades.append({'Typ':'Kauf','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':positionswert,
                           'Profit/Loss':None,'Kumulative P&L':cum_pnl})
        # Verkauf Long
        if ms_t < ml_t and ms_y >= ml_y and position == 1:
            position = 0
            gross = (price - initial_price)/initial_price * positionswert
            cum_pnl += gross
            wealth += positionswert + gross
            trades.append({'Typ':'Verkauf (Long)','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':None,
                           'Profit/Loss':gross,'Kumulative P&L':cum_pnl})
        # Short eröffnen
        if ms_t < ml_t and ms_y >= ml_y and position == 0:
            initial_price = price
            position = -1
            positionswert = wealth
            wealth -= positionswert
            trades.append({'Typ':'Short-Sell','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':positionswert,
                           'Profit/Loss':None,'Kumulative P&L':cum_pnl})
        # Short-Cover → Long
        if ms_t > ml_t and ms_y <= ml_y and position == -1:
            gross = (initial_price - price)/initial_price * positionswert
            cum_pnl += gross
            wealth += positionswert + gross
            trades.append({'Typ':'Short-Cover','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':None,
                           'Profit/Loss':gross,'Kumulative P&L':cum_pnl})
            # sofort Long
            initial_price = price
            position = 1
            positionswert = wealth
            wealth -= positionswert
            trades.append({'Typ':'Kauf (nach Cover)','Datum':date_i,'Kurs':price,
                           'Spesen':0,'Positionswert':positionswert,
                           'Profit/Loss':None,'Kumulative P&L':cum_pnl})

    # offene Position schließen
    if position != 0:
        last_p = df_vis['Close'].iat[-1]
        last_d = df_vis.index[-1]
        if position == 1:
            gross = (last_p - initial_price)/initial_price * positionswert
        else:
            gross = (initial_price - last_p)/initial_price * positionswert
        cum_pnl += gross
        wealth += positionswert + gross
        wealth_hist[-1] = wealth
        trades.append({'Typ':'Schließen (Ende)','Datum':last_d,'Kurs':last_p,
                       'Spesen':0,'Positionswert':None,
                       'Profit/Loss':gross,'Kumulative P&L':cum_pnl})

    trades_df = pd.DataFrame(trades)
    strat_ret = (wealth - start_capital)/start_capital * 100
    bh_ret    = (df_vis['Close'].iat[-1] - df_vis['Close'].iat[0]) / df_vis['Close'].iat[0] * 100

    df_plot = pd.DataFrame({'Close': df_vis['Close'], 'Position': pos_hist}, index=df_vis.index)
    df_wealth = pd.DataFrame({'Datum': df_vis.index, 'Wealth': wealth_hist})

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
# Streamlit App
# ---------------------------------------
st.title("✨ AI Quant Model")

st.markdown("""
Bitte wähle unten den Ticker (Yahoo Finance), den Beginn des Zeitraums und das Startkapital aus.
""")

ticker_input = st.text_input(
    label="1️⃣ Welchen Aktien-Ticker möchtest du analysieren?",
    value="AAPL",
    placeholder="z.B. AAPL"
)
start_date_input = st.date_input(
    label="2️⃣ Beginn des Analyse-Zeitraums",
    value=date(2024, 1, 1),
    max_value=date.today()
)
start_capital_input = st.number_input(
    label="3️⃣ Startkapital (€)",
    value=10000,
    min_value=1000,
    step=500
)

st.markdown("---")
run_button = st.button("🔄 Ergebnisse berechnen")

if not run_button:
    st.markdown("### ℹ️ Beschreibung dieses AI Quant Modells")
    st.markdown("Hier Dein Beschreibungstext …")
else:
    if ticker_input.strip() == "":
        st.error("Bitte gib zunächst einen gültigen Ticker ein.")
    else:
        with st.spinner("⏳ Berechne Signale und Trades…"):
            results = optimize_and_run(
                ticker_input,
                start_date_input.strftime("%Y-%m-%d"),
                float(start_capital_input)
            )

        # ------------------------------
        # 0️⃣ Optimierungsergebnisse
        # ------------------------------
        bs, bl = results["best_individual"]
        st.subheader("0. Optimierungsergebnisse")
        st.markdown(f"- **Bestes Short MA‐Fenster:** {bs}")
        st.markdown(f"- **Bestes Long MA‐Fenster:**  {bl}")
        
        # Sharpe‐Verlauf aus dem Logbook plott­en
        logbook = results["logbook"]
        # Extrahiere nur die Max-Fitness jeder Generation
        gen_max = logbook.select("max")
        df_log = pd.DataFrame({
            "Generation": list(range(len(gen_max))),
            "Max Sharpe": gen_max
        })
        df_log = df_log.set_index("Generation")
        st.line_chart(df_log, use_container_width=True)




        
        # Extrahiere Ergebnisse
        trades_df = results["trades_df"]
        strategy_return     = results["strategy_return"]
        buy_and_hold_return = results["buy_and_hold_return"]
        total_trades        = results["total_trades"]
        long_trades         = results["long_trades"]
        short_trades        = results["short_trades"]
        pos_count           = results["pos_count"]
        neg_count           = results["neg_count"]
        pos_pct             = results["pos_pct"] or (pos_count/(pos_count+neg_count)*100 if pos_count+neg_count>0 else 0)
        neg_pct             = results["neg_pct"] or (neg_count/(pos_count+neg_count)*100 if pos_count+neg_count>0 else 0)
        pos_pnl             = results["pos_pnl"]
        neg_pnl             = results["neg_pnl"]
        total_pnl           = results["total_pnl"]
        pos_perf            = results["pos_perf"] or (pos_pnl/(pos_pnl+neg_pnl)*100 if pos_pnl+neg_pnl!=0 else 0)
        neg_perf            = results["neg_perf"] or (neg_pnl/(pos_pnl+neg_pnl)*100 if pos_pnl+neg_pnl!=0 else 0)
        df_plot             = results["df_plot"]
        df_wealth           = results["df_wealth"]

        # 1) Performance-Vergleich
        st.subheader("1. Performance-Vergleich")
        fig1, ax1 = plt.subplots(figsize=(8,5))
        bars = ax1.bar(
            ['Strategie','Buy & Hold'],
            [strategy_return, buy_and_hold_return],
            color=['#2ca02c','#000000'], alpha=0.7
        )
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, h, f"{h:.2f}%", ha='center', va='bottom')
        ax1.set_ylabel("Rendite (%)")
        ax1.set_title(f"Strategie vs. Buy-&-Hold für {ticker_input}")
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig1)

        # 2) Kursdiagramm mit Phasen
        st.subheader("2. Kursdiagramm mit Phasen")
        fig2, ax2 = plt.subplots(figsize=(10,5))
        dates, prices, phases = df_plot.index, df_plot['Close'], df_plot['Position']
        ax2.plot(dates, prices, color='black', label='Close')
        curr = phases.iloc[0]; start = dates[0]
        for i in range(1,len(dates)):
            if phases.iloc[i]!=curr:
                end = dates[i-1]
                col = 'green' if curr==1 else 'red' if curr==-1 else None
                if col: ax2.axvspan(start,end,color=col,alpha=0.2)
                curr, start = phases.iloc[i], dates[i]
        col = 'green' if curr==1 else 'red' if curr==-1 else None
        if col: ax2.axvspan(start,dates[-1],color=col,alpha=0.2)
        ax2.set_title(f"{ticker_input}-Kurs mit Phasen")
        ax2.set_xlabel("Datum"); ax2.set_ylabel("Preis"); ax2.grid(True,linestyle='--',alpha=0.4)
        st.pyplot(fig2)

        # 3) Trade-Tabelle
        st.subheader("3. Tabelle der Einzeltrades")
        df_tbl = trades_df[['Datum','Typ','Kurs','Profit/Loss','Kumulative P&L']].copy()
        df_tbl['Datum'] = df_tbl['Datum'].dt.strftime('%Y-%m-%d')
        df_tbl[['Kurs','Profit/Loss','Kumulative P&L']] = df_tbl[['Kurs','Profit/Loss','Kumulative P&L']].applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
        st.dataframe(df_tbl, use_container_width=True)

        # 4) Handelsstatistiken
        st.subheader("4. Handelsstatistiken")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.metric("Einträge gesamt", total_trades)
            st.metric("Long-Einträge", long_trades)
            st.metric("Short-Einträge", short_trades)
        with c2:
            st.metric("Positive Trades", pos_count)
            st.metric("Negative Trades", neg_count)
            st.metric("Positive (%)", f"{pos_pct:.2f}%")
        with c3:
            st.metric("Strategie-Return", f"{strategy_return:.2f}%")
            st.metric("B&H-Return", f"{buy_and_hold_return:.2f}%")
            diff = strategy_return - buy_and_hold_return
            st.metric("Outperformance", f"{diff:+.2f}%")
        st.markdown(f"""
- **Negativ (%)**: {neg_pct:.2f}%  
- **P&L positiv**: {pos_pnl:.2f} EUR  
- **P&L negativ**: {neg_pnl:.2f} EUR  
- **Gesamt-P&L**: {total_pnl:.2f} EUR  
- **Perf. positiv**: {pos_perf:.2f}%  
- **Perf. negativ**: {neg_perf:.2f}%  
        """)

        # 5) Anzahl der Trades
        st.subheader("5. Anzahl der Trades")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        ax3.bar(['Gesamt','Long','Short'], [total_trades,long_trades,short_trades],
                color=['#4c72b0','#55a868','#c44e52'], alpha=0.8)
        ax3.set_ylabel("Anzahl"); ax3.set_title("Trade-Einträge")
        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig3)

        # 6) Kombiniertes Chart Price & Wealth
        st.subheader("6. Price & Wealth Performance")
        fig4, axp = plt.subplots(figsize=(10,6))
        aw = axp.twinx()
        axp.plot(df_plot.index, df_plot['Close'], label='Preis', linewidth=1, color='black', alpha=0.5)
        aw.plot(df_wealth['Datum'], df_wealth['Wealth'], label='Wealth', linewidth=1.3, color='green', alpha=0.8)
        # Phasen
        phases = df_plot['Position'].values; dates = df_plot.index
        curr, start = phases[0], dates[0]
        for i in range(1,len(dates)):
            if phases[i]!=curr:
                end = dates[i-1]
                col = 'green' if curr==1 else 'red' if curr==-1 else None
                if col: axp.axvspan(start,end,color=col,alpha=0.15)
                curr, start = phases[i], dates[i]
        col = 'green' if curr==1 else 'red' if curr==-1 else None
        if col: axp.axvspan(start,dates[-1],color=col,alpha=0.15)
        axp.set_xlabel("Datum"); axp.set_ylabel("Preis", color="black")
        aw.set_ylabel("Wealth (€)", color="green")
        lines, labels = axp.get_legend_handles_labels()
        lines2, labels2 = aw.get_legend_handles_labels()
        axp.legend(lines+lines2, labels+labels2, loc='upper left')
        axp.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig4)

        # 7) Normalized Price vs Wealth
        st.subheader("7. Normalized Price vs. Wealth")
        df_w = df_wealth.set_index('Datum').reindex(df_plot.index, method='ffill')
        price0, wealth0 = df_plot['Close'].iloc[0], df_w['Wealth'].iloc[0]
        df_plot['PriceNorm'] = df_plot['Close']/price0
        df_w['WealthNorm'] = df_w['Wealth']/wealth0
        fig5, ax5 = plt.subplots(figsize=(10,6))
        ax5.plot(df_plot.index, df_plot['PriceNorm'], label='Normierter Preis', linewidth=1, alpha=0.5, color='black')
        ax5.plot(df_w.index, df_w['WealthNorm'], label='Normierte Wealth', linewidth=1.5, alpha=0.8, color='green')
        # Phasen shading
        phases = df_plot['Position'].values; dates = df_plot.index
        curr, start = phases[0], dates[0]
        for i in range(1,len(dates)):
            if phases[i]!=curr:
                end = dates[i-1]
                col = 'green' if curr==1 else 'red' if curr==-1 else None
                if col: ax5.axvspan(start,end,color=col,alpha=0.10)
                curr, start = phases[i], dates[i]
        col = 'green' if curr==1 else 'red' if curr==-1 else None
        if col: ax5.axvspan(start,dates[-1],color=col,alpha=0.10)
        ax5.set_xlabel("Datum"); ax5.set_ylabel("Normierter Wert (t₀→1)")
        ax5.legend(loc='upper left'); ax5.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig5)
