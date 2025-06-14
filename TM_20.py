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
# ---------------------------------------
# Funktion, die im Hintergrund Optimierung und Trading ausführt
# ---------------------------------------
def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    """
    Lädt Kursdaten, führt die MA-Optimierung per GA durch,
    wählt per Post-Selection das Pareto-best Trade-arme Sharpe-Optimum
    und simuliert final das Trading.
    """
    # 1. Datenbeschaffung und -vorbereitung
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    data['Close'] = data['Close'].interpolate(method='linear')
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # 2. Fitness-Funktion für Sharpe-Optimierung (single-objective)
    def evaluate_sharpe(individual):
        ma_s, ma_l = int(individual[0]), int(individual[1])
        # ungültige Fenster
        if ma_s >= ma_l or ma_s <= 0 or ma_l <= 0:
            return (-np.inf,)
        df = data.copy()
        df['MA_short'] = df['Close'].rolling(window=ma_s).mean()
        df['MA_long']  = df['Close'].rolling(window=ma_l).mean()
        df.dropna(inplace=True)

        position = 0
        wealth_line = [start_capital]
        trade_price = 0.0

        # Backtest-Schleife
        for i in range(1, len(df)):
            price_today = float(df['Close'].iloc[i])
            ma_s_t = float(df['MA_short'].iloc[i])
            ma_l_t = float(df['MA_long'].iloc[i])
            ma_s_y = float(df['MA_short'].iloc[i-1])
            ma_l_y = float(df['MA_long'].iloc[i-1])

            # Signal Long
            if ma_s_t > ma_l_t and ma_s_y <= ma_l_y:
                if position == -1:
                    pnl = (trade_price - price_today)/trade_price * wealth_line[-1]
                    wealth_line.append(wealth_line[-1] + pnl)
                if position != 1:
                    position = 1
                    trade_price = price_today
                continue

            # Signal Short
            if ma_s_t < ma_l_t and ma_s_y >= ma_l_y:
                if position == 1:
                    pnl = (price_today - trade_price)/trade_price * wealth_line[-1]
                    wealth_line.append(wealth_line[-1] + pnl)
                if position != -1:
                    position = -1
                    trade_price = price_today
                continue

            # sonst halten
            wealth_line.append(wealth_line[-1])

        # Sharpe berechnen
        ws = pd.Series(wealth_line)
        ret = ws.pct_change().dropna()
        if ret.std() == 0:
            return (-np.inf,)
        sharpe = (ret.mean() - (0.02/252)) / ret.std() * np.sqrt(252)
        return (sharpe,)

    # 2b. Hilfsfunktion: gleicher Backtest, aber nur Trade-Anzahl zählen
    def evaluate_sharpe_and_count(individual):
        ma_s, ma_l = int(individual[0]), int(individual[1])
        if ma_s >= ma_l or ma_s <= 0 or ma_l <= 0:
            return -np.inf, 0
        df = data.copy()
        df['MA_short'] = df['Close'].rolling(window=ma_s).mean()
        df['MA_long']  = df['Close'].rolling(window=ma_l).mean()
        df.dropna(inplace=True)

        position = 0
        trade_price = 0.0
        trade_count = 0
        wealth_line = [start_capital]

        for i in range(1, len(df)):
            price_today = float(df['Close'].iloc[i])
            ma_s_t = float(df['MA_short'].iloc[i])
            ma_l_t = float(df['MA_long'].iloc[i])
            ma_s_y = float(df['MA_short'].iloc[i-1])
            ma_l_y = float(df['MA_long'].iloc[i-1])

            # Long-Signal
            if ma_s_t > ma_l_t and ma_s_y <= ma_l_y:
                # schließe Short
                if position == -1:
                    trade_count += 1
                    pnl = (trade_price - price_today)/trade_price * wealth_line[-1]
                    wealth_line.append(wealth_line[-1] + pnl)
                # eröffne Long
                if position != 1:
                    position = 1
                    trade_price = price_today
                    trade_count += 1
                continue

            # Short-Signal
            if ma_s_t < ma_l_t and ma_s_y >= ma_l_y:
                # schließe Long
                if position == 1:
                    trade_count += 1
                    pnl = (price_today - trade_price)/trade_price * wealth_line[-1]
                    wealth_line.append(wealth_line[-1] + pnl)
                # eröffne Short
                if position != -1:
                    position = -1
                    trade_price = price_today
                    trade_count += 1
                continue

            # halten
            wealth_line.append(wealth_line[-1])

        # letzte Position schließen
        if position != 0:
            last_price = float(df['Close'].iloc[-1])
            trade_count += 1
            if position == 1:
                pnl = (last_price - trade_price)/trade_price * wealth_line[-1]
            else:
                pnl = (trade_price - last_price)/trade_price * wealth_line[-1]
            wealth_line[-1] = wealth_line[-1] + pnl

        # Sharpe erneut berechnen
        ws = pd.Series(wealth_line)
        ret = ws.pct_change().dropna()
        if ret.std() == 0:
            sharpe = -np.inf
        else:
            sharpe = (ret.mean() - (0.02/252)) / ret.std() * np.sqrt(252)

        return sharpe, trade_count

    # 3. DEAP-Setup
    # Single-Objective für GA
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 5, 50)
    toolbox.register("individual",
                     tools.initCycle,
                     creator.Individual,
                     (toolbox.attr_int, toolbox.attr_int),
                     n=1)
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

    # 3.2 GA laufen lassen
    pop = toolbox.population(n=20)
    population, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=10,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    # >>> Post-Selection: Sharpe-Topkandidaten filtern, dann minimale Trades
    best_sharpe = max(ind.fitness.values[0] for ind in population)
    epsilon     = 0.01 * best_sharpe
    candidates  = [ind for ind in population
                   if ind.fitness.values[0] >= best_sharpe - epsilon]

    # Simuliere Trade-Count und wähle minimalen
    best = min(candidates, key=lambda ind: evaluate_sharpe_and_count(ind)[1])

    # 4. Gerundete MA-Werte und finaler Backtest wie gehabt
    best_short = int(round(best[0]))
    best_long  = int(round(best[1]))
    if best_short >= best_long:
        best_short, best_long = best_long-1, best_short+1

    # … hier kommt der restliche Backtest / Plot-/Trade-Tabellen-Code
    # (zeile für zeile identisch zu Deinem originalen Abschnitt 5ff.)
    # Du fügst jetzt einfach weiter die Simulation mit best_short/best_long
    # durch und gibst am Ende die gleichen Rückgabe-Strukturen zurück.

    # (aus Platzgründen hier nicht wiederholt,
    # verbleibt aber 1:1 in Deinem Skript)

    return {
        "best_individual": (best_short, best_long),
        "logbook": logbook,
        # … alle anderen Rückgabewerte wie trades_df, df_plot, df_wealth …
    }




# ---------------------------------------
# Streamlit-App
# ---------------------------------------
st.title("✨ AI Quant LS Model - INCLR")

st.markdown("""
Bitte wähle unten den Ticker (Yahoo Finance) , den Beginn des Zeitraums und das Startkapital aus.  
""")

# ------------------------------
# Eingabefelder für Ticker / Zeitfenster / Startkapital
# ------------------------------
ticker_input = st.text_input(
    label="1️⃣ Welchen Aktien-Ticker möchtest du analysieren?",
    value="",  # leerer Standardwert
    help="Gib hier das Tickersymbol ein, z.B. 'AAPL', 'MSFT' oder 'O'."
)

start_date_input = st.date_input(
    label="2️⃣ Beginn des Analyse-Zeitraums",
    value=date(2024, 1, 1),
    max_value=date.today(),
    help="Wähle das Startdatum (bis heute)."
)

start_capital_input = st.number_input(
    label="3️⃣ Startkapital (€)",
    value=10000,       # als Integer
    min_value=1000,    # ebenfalls Integer
    step=500,          # Schrittweite in ganzen Euro
    format="%d",       # zeigt keine Dezimalstellen an
    help="Gib das Startkapital in ganzen Euro ein (ab € 1.000)."
)

st.markdown("---")

# -------------
# Button zum Starten der Berechnung
# -------------
run_button = st.button("🔄 Ergebnisse berechnen")

# nur wenn der Button gedrückt wurde und ein Ticker eingegeben ist:
if run_button:
    if ticker_input.strip() == "":
        st.error("Bitte gib zunächst einen gültigen Ticker ein, z. B. 'AAPL' oder 'MSFT'.")
    else:
        start_date_str = start_date_input.strftime("%Y-%m-%d")
        with st.spinner("⏳ Berechne Signale und Trades… bitte einen Moment warten"):
            results = optimize_and_run(ticker_input, start_date_str, float(start_capital_input))

        trades_df = results["trades_df"]
        strategy_return = results["strategy_return"]
        buy_and_hold_return = results["buy_and_hold_return"]
        total_trades = results["total_trades"]
        long_trades = results["long_trades"]
        short_trades = results["short_trades"]
        pos_count = results["pos_count"]
        neg_count = results["neg_count"]
        pos_pct = results["pos_pct"]
        neg_pct = results["neg_pct"]
        pos_pnl = results["pos_pnl"]
        neg_pnl = results["neg_pnl"]
        total_pnl = results["total_pnl"]
        pos_perf = results["pos_perf"]
        neg_perf = results["neg_perf"]
        df_plot = results["df_plot"]
        df_wealth = results["df_wealth"]

        # ---------------------------------------
        # 1. Performance-Vergleich (Strategie vs. Buy & Hold)
        # ---------------------------------------
        st.subheader("1. Performance-Vergleich")
        fig_performance, ax_perf = plt.subplots(figsize=(8, 5))

        # Balken zeichnen
        bars = ax_perf.bar(
            ['Strategie', 'Buy & Hold'],
            [strategy_return, buy_and_hold_return],
            color=['#2ca02c', '#000000'],
            alpha=0.7
        )

        # Prozentwerte über die Balken schreiben
        for bar in bars:
            height = bar.get_height()
            ax_perf.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position in der Mitte des Balkens
                height,                              # y-Position genau auf dem Balken
                f"{height:.2f}%",                    # Beschriftung
                ha='center',                         # horizontal zentriert
                va='bottom'                          # vertikal direkt über dem Balken
            )

        ax_perf.set_ylabel("Rendite (%)", fontsize=12)
        ax_perf.set_title(f"Strategie vs. Buy-&-Hold für {ticker_input}", fontsize=14)
        ax_perf.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_performance)

        # ---------------------------------------
        # 2. Kursdiagramm mit Kauf-/Verkaufsphasen
        # ---------------------------------------
        st.subheader("2. Kursdiagramm mit Phasen (Kauf/Verkauf)")

        fig_trades, ax_trades = plt.subplots(figsize=(10, 5))
        dates = df_plot.index
        prices = df_plot['Close']
        positions = df_plot['Position']

        ax_trades.plot(dates, prices, label='Close-Preis', color='black', linewidth=1)

        # Phasenschattierung: Long = grün, Short = rot, Neutral = transparent
        current_phase = positions.iloc[0]
        start_idx = dates[0]
        for i in range(1, len(dates)):
            if positions.iloc[i] != current_phase:
                end_idx = dates[i - 1]
                if current_phase == 1:
                    ax_trades.axvspan(start_idx, end_idx, color='green', alpha=0.2)
                elif current_phase == -1:
                    ax_trades.axvspan(start_idx, end_idx, color='red', alpha=0.2)
                current_phase = positions.iloc[i]
                start_idx = dates[i]
        # Letzte Phase bis zum Ende
        if current_phase == 1:
            ax_trades.axvspan(start_idx, dates[-1], color='green', alpha=0.2)
        elif current_phase == -1:
            ax_trades.axvspan(start_idx, dates[-1], color='red', alpha=0.2)

        ax_trades.set_title(f"{ticker_input}-Kurs mit Kauf-/Verkaufsphasen", fontsize=14)
        ax_trades.set_xlabel("Datum", fontsize=12)
        ax_trades.set_ylabel("Preis", fontsize=12)
        ax_trades.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig_trades)

        st.markdown("""
        - **Grüne Bereiche**: Long-Phase (Kaufsignal aktiv).  
        - **Rote Bereiche**: Short-Phase (Verkaufssignal aktiv).  
        - **Ohne Schattierung**: Neutral (keine offene Position).
        """)

        # ---------------------------------------
        # 3. Tabelle der Einzeltrades
        # ---------------------------------------
        st.subheader("3. Tabelle der Einzeltrades")
        trades_table = trades_df[['Datum', 'Typ', 'Kurs', 'Profit/Loss', 'Kumulative P&L']].copy()
        trades_table['Datum'] = trades_table['Datum'].dt.strftime('%Y-%m-%d')
        trades_table['Kurs'] = trades_table['Kurs'].map('{:.2f}'.format)
        trades_table['Profit/Loss'] = trades_table['Profit/Loss'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "-")
        trades_table['Kumulative P&L'] = trades_table['Kumulative P&L'].map('{:.2f}'.format)
        st.dataframe(trades_table, use_container_width=True)

        # ---------------------------------------
        # 4. Handelsstatistiken
        # ---------------------------------------
        st.subheader("4. Handelsstatistiken")

        # Layout: drei Spalten
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Gesamtzahl der Einträge (Entry+Exit)", total_trades)
            st.metric("Davon Long-Trades (Entry-Zeilen)", long_trades)
            st.metric("Davon Short-Trades (Entry-Zeilen)", short_trades)

        with col2:
            st.metric("Positive Trades (Anzahl)", pos_count)
            st.metric("Negative Trades (Anzahl)", neg_count)
            st.metric("Positive Trades (%)", f"{pos_pct:.2f}%")

        with col3:
            # Neue Kennzahlen: Strategie‐Performance vs. Buy-&-Hold
            st.metric("Strategie-Return", f"{strategy_return:.2f}%")
            st.metric("Buy-&-Hold-Return", f"{buy_and_hold_return:.2f}%")
            # Zusatz: Differenz oder Out-/Underperformance
            diff = strategy_return - buy_and_hold_return
            if diff >= 0:
                sign = "+"
            else:
                sign = ""
            st.metric("Outperformance vs. B&H", f"{sign}{diff:.2f}%")

        # Bullet‐Points mit P&L‐Summen und Performances pro Trade-Typ
        st.markdown(f"""
        - **Negative Trades (%)**: {neg_pct:.2f}%  
        - **Gesamt-P&L der positiven Trades**: {pos_pnl:.2f} EUR  
        - **Gesamt-P&L der negativen Trades**: {neg_pnl:.2f} EUR  
        - **Gesamt-P&L des Systems**: {total_pnl:.2f} EUR  
        - **Performance positive Trades**: {pos_perf:.2f}%  
        - **Performance negative Trades**: {neg_perf:.2f}%  
        """)

        # Professioneller Vergleichstext
        st.markdown("""
        ---
        **Vergleich Modell-Performance vs. Buy-&-Hold**  
        - Das Handelssystem erzielte in diesem Zeitraum eine Gesamt-Rendite von **{strategy_return:.2f}%**,  
          während die Buy-&-Hold-Strategie nur **{buy_and_hold_return:.2f}%** erwirtschaftete.  
        - Dies entspricht einer **Outperformance von {diff:+.2f}%** gegenüber dem reinen Halten der Aktie.  
        - Insbesondere in Seitwärts- oder Trendwechsel-Phasen profitiert das System von den Long/Short-Signalen, wodurch Drawdowns verkürzt und Gewinne im Gegentrend mitgenommen werden.  
        - Die Buy-&-Hold-Strategie erzielt zwar in starken Hausse-Phasen gute Renditen, kann in volatilen Fällen aber größere Verluste hinnehmen, da sie nicht zwischen Long und Short unterscheidet.  
        ---  
        """.format(
            strategy_return=strategy_return,
            buy_and_hold_return=buy_and_hold_return,
            diff=diff
        ))

        # ---------------------------------------
        # 5. Balkendiagramm: Anzahl der Trades
        # ---------------------------------------
        st.subheader("5. Anzahl der Trades (Entry+Exit, Long, Short)")
        fig_counts, ax_counts = plt.subplots(figsize=(6, 4))
        ax_counts.bar(
            ['Einträge gesamt', 'Long-Einträge', 'Short-Einträge'],
            [total_trades, long_trades, short_trades],
            color=['#4c72b0', '#55a868', '#c44e52'],
            alpha=0.8
        )
        ax_counts.set_ylabel("Anzahl", fontsize=12)
        ax_counts.set_title("Trade-Einträge-Verteilung", fontsize=14)
        ax_counts.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig_counts)

        
        # -------------------------------------------------------------------
        # Kombiniertes Chart: Aktienkurs, Wealth Performance und Phasen
        # -------------------------------------------------------------------
        
        st.subheader("6. Price & Wealth Performance: Phases")
        
        # Erstelle das Figure‐Objekt und zwei Achsen (linke Achse für den Kurs, rechte Achse für Wealth)
        fig_combined, ax_price = plt.subplots(figsize=(10, 6))
        
        # Die zweite Y‐Achse (rechts) teilen
        ax_wealth = ax_price.twinx()
        
        # X‐Werte (Datum) holen
        dates_price = df_plot.index            # Index von df_plot (DatetimeIndex)
        dates_wealth = df_wealth["Datum"]      # Datumsspalte von df_wealth (DatetimeIndex)
        
        # 1. Aktienkurs (linke Achse,schwarz)
        ax_price.plot(
            dates_price,
            df_plot["Close"],
            label="Schlusskurs",
            color="#000000",
            linewidth=1.0,
            alpha=0.5
        )
        
        # 2. Wealth Performance (rechte Achse, gruen)
        ax_wealth.plot(
            dates_wealth,
            df_wealth["Wealth"],
            label="Wealth Performance",
            color="#2ca02c",
            linewidth=1.3,
            alpha=0.8
        )
        
        # 3. Phasen‐Shading über den Kurs‐Plot legen
        #    Wir lesen die Positionen aus df_plot: 1=Long, -1=Short, 0=Neutral
        positions = df_plot["Position"].values
        
        # Wir gehen das Datum‐Array durch und schattieren, sobald sich die Position ändert
        current_phase = positions[0]
        phase_start = dates_price[0]
        
        for i in range(1, len(dates_price)):
            if positions[i] != current_phase:
                phase_end = dates_price[i - 1]
                if current_phase == 1:
                    ax_price.axvspan(phase_start, phase_end, color="green", alpha=0.15)
                elif current_phase == -1:
                    ax_price.axvspan(phase_start, phase_end, color="red", alpha=0.15)
                # Neue Phase starten
                current_phase = positions[i]
                phase_start = dates_price[i]
        
        # Letzte Phase bis zum Ende
        if current_phase == 1:
            ax_price.axvspan(phase_start, dates_price[-1], color="green", alpha=0.15)
        elif current_phase == -1:
            ax_price.axvspan(phase_start, dates_price[-1], color="red", alpha=0.15)
        
        # 4. Achsen‐Beschriftungen, Legende, Titel, Grid
        ax_price.set_xlabel("Datum", fontsize=12, weight="normal")
        ax_price.set_ylabel("Schlusskurs", fontsize=12, color="#000000", weight="normal")
        ax_wealth.set_ylabel("Wealth (€)", fontsize=12, color="#2ca02c", weight="normal")
        
        ax_price.tick_params(axis="y", labelcolor="#000000")
        ax_wealth.tick_params(axis="y", labelcolor="#2ca02c")
        
        # Gemeinsame Legende: Wir kombinieren die Handles beider Achsen
        lines_price, labels_price = ax_price.get_legend_handles_labels()
        lines_wealth, labels_wealth = ax_wealth.get_legend_handles_labels()
        all_lines = lines_price + lines_wealth
        all_labels = labels_price + labels_wealth
        
        ax_price.legend(all_lines, all_labels, loc="upper left", frameon=True, fontsize=10)
        
        # Leichtes Grid im Hintergrund
        ax_price.grid(True, linestyle="--", alpha=0.4)
        
        # Professioneller Titel
        ax_price.set_title(
            f"{ticker_input}: Price & Wealth Performance incl. Phases",
            fontsize=14,
            weight="normal"
        )
        
        # X‐Achse optisch enger machen
        fig_combined.autofmt_xdate(rotation=0)
        
        # Plot in Streamlit einbinden
        st.pyplot(fig_combined)






        # ---------------------------------------
        # 6. Normiertes Single‐Axis‐Chart: Kurs & Wealth, beide ab 1 am selben Tag
        # ---------------------------------------
        st.subheader("7. Normalized Price vs. Wealth Index")
        
        # 1. Gemeinsames Startdatum (erster Eintrag in df_plot)
        start_date = df_plot.index[0]
        
        # 2. Wealth so zuschneiden, dass es ab genau diesem Datum beginnt
        df_wealth_synced = df_wealth[df_wealth["Datum"] >= start_date].copy()
        df_wealth_synced.set_index("Datum", inplace=True)
        
        # 3. Reindexiere df_wealth_synced auf denselben Index wie df_plot.index, mit Forward‐Fill
        df_wealth_reindexed = df_wealth_synced.reindex(df_plot.index, method="ffill")
        
        # 4. Normierung: beide Reihen auf 1 bringen (beide am selben Datum!)
        price0  = df_plot["Close"].iloc[0]
        wealth0 = df_wealth_reindexed["Wealth"].iloc[0]
        
        df_plot["PriceNorm"]            = df_plot["Close"]        / price0
        df_wealth_reindexed["WealthNorm"] = df_wealth_reindexed["Wealth"] / wealth0
        
        # 5. Plot beider Norm‐Zeitenreihen auf einer Achse
        fig_single, ax = plt.subplots(figsize=(10, 6))
        
        dates = df_plot.index
        
        # a) Normierter Kurs (schwarz Linie)
        ax.plot(
            dates,
            df_plot["PriceNorm"],
            label="Normierter Kurs",
            color="#000000",
            linewidth=1.0,
            alpha=0.5
        )
        
        # b) Normierte Wealth (gruen Linie)
        ax.plot(
            dates,
            df_wealth_reindexed["WealthNorm"],
            label="Normierte Wealth",
            color="#2ca02c",
            linewidth=1.5,
            alpha=0.8
        )
        
        # c) Phasen‐Shading (Long = grün, Short = rot)
        positions = df_plot["Position"].values
        current_phase = positions[0]
        phase_start = dates[0]
        for i in range(1, len(dates)):
            if positions[i] != current_phase:
                phase_end = dates[i - 1]
                if current_phase == 1:
                    ax.axvspan(phase_start, phase_end, color="green", alpha=0.10)
                elif current_phase == -1:
                    ax.axvspan(phase_start, phase_end, color="red", alpha=0.10)
                current_phase = positions[i]
                phase_start = dates[i]
        
        # Letzte Phase bis zum Ende
        if current_phase == 1:
            ax.axvspan(phase_start, dates[-1], color="green", alpha=0.10)
        elif current_phase == -1:
            ax.axvspan(phase_start, dates[-1], color="red", alpha=0.10)
        
        # d) Achsen‐Beschriftungen und Legende
        ax.set_xlabel("Datum", fontsize=12, weight="normal")
        ax.set_ylabel("Normierter Wert (t₀ → 1)", fontsize=12, weight="normal")
        
        ax.legend(loc="upper left", frameon=True, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        
        # e) Titel
        ax.set_title(
            f"{ticker_input}: Normalized Price vs. Wealth Index",
            fontsize=14,
            weight="normal"
        )
        
        fig_single.autofmt_xdate(rotation=0)
        st.pyplot(fig_single)


        # ───────────────────────────────────────────────────
        # 🔧 Optimierungsergebnisse (neu)
        st.markdown("---")
        st.subheader("🔧 Optimierungsergebnisse")
        
        # Metriken für die optimalen MAs
        best_short, best_long = results["best_individual"]
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("✨ MA kurz (optimal)", f"{best_short:.2f}")
        with col_b:
            st.metric("✨ MA lang (optimal)", f"{best_long:.2f}")
        
        # Fitness-Verlauf über die Generationen
        logbook = results["logbook"]
        df_log = pd.DataFrame(logbook)

        
        fig_opt, ax_opt = plt.subplots(figsize=(8, 4))
        ax_opt.plot(df_log["gen"], df_log["max"],    label="Max Sharpe", linewidth=2)
        ax_opt.plot(df_log["gen"], df_log["avg"],    label="Avg Sharpe", linewidth=1.5)
        ax_opt.fill_between(df_log["gen"], df_log["min"], df_log["max"], alpha=0.2)
        ax_opt.set_xlabel("Generation", fontsize=11)
        ax_opt.set_ylabel("Sharpe Ratio", fontsize=11)
        ax_opt.set_title("Optimierungsverlauf (Sharpe Ratio)", fontsize=13)
        ax_opt.grid(True, linestyle="--", alpha=0.4)
        ax_opt.legend(frameon=True, fontsize=10)
        st.pyplot(fig_opt)
