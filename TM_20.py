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

# @st.cache_data(show_spinner=False)      #wenn man die Speicherfunktion haben möchte --> dann diese Zeile aktivieren
def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    """
    Lädt Kursdaten für den gegebenen Ticker und Start-Datum (bis heute),
    führt die MA-Optimierung per GA durch, rundet die MA-Fenster, simuliert das Trading
    und gibt alle relevanten Ergebnisse zurück. Der Startbetrag wird durch start_capital festgelegt.
    """
    # 1. Datenbeschaffung und -aufbereitung
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    data['Close'] = data['Close'].interpolate(method='linear')
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)

    # 2. Fitness-Funktion für den GA
    def evaluate_strategy(individual):
        ma_short_window, ma_long_window = int(individual[0]), int(individual[1])
        if ma_short_window >= ma_long_window or ma_short_window <= 0 or ma_long_window <= 0:
            return -np.inf,
        df = data.copy()
        df['MA_short'] = df['Close'].rolling(window=ma_short_window).mean()
        df['MA_long'] = df['Close'].rolling(window=ma_long_window).mean()
        df.dropna(inplace=True)

        position = 0
        wealth_line = [start_capital]
        cumulative_pnl = 0

        for i in range(1, len(df)):
            price_today = float(df['Close'].iloc[i])
            ma_s_t = float(df['MA_short'].iloc[i])
            ma_l_t = float(df['MA_long'].iloc[i])
            ma_s_y = float(df['MA_short'].iloc[i - 1])
            ma_l_y = float(df['MA_long'].iloc[i - 1])

            # Long-Strategie
            if ma_s_t > ma_l_t and ma_s_y <= ma_l_y:
                if position == -1:
                    pnl = (trade_price - price_today) / trade_price * wealth_line[-1]
                    cumulative_pnl += pnl
                    wealth_line.append(wealth_line[-1] + pnl)
                if position != 1:
                    position = 1
                    trade_price = price_today

            # Short-Strategie
            elif ma_s_t < ma_l_t and ma_s_y >= ma_l_y:
                if position == 1:
                    pnl = (price_today - trade_price) / trade_price * wealth_line[-1]
                    cumulative_pnl += pnl
                    wealth_line.append(wealth_line[-1] + pnl)
                if position != -1:
                    position = -1
                    trade_price = price_today

            # Position halten
            else:
                wealth_line.append(wealth_line[-1])

        wealth_series = pd.Series(wealth_line)
        returns = wealth_series.pct_change().dropna()
        if returns.std() == 0:
            return -np.inf,
        sharpe = (returns.mean() - (0.02 / 252)) / returns.std() * np.sqrt(252)
        return sharpe,

    # 3. DEAP-Setup für GA
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 5, 50)
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (toolbox.attr_int, toolbox.attr_int),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_strategy)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 3.1 Statistiken und Hall of Fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    # 3.2 GA laufen lassen (zeichnet logbook und hof auf)
    population = toolbox.population(n=20)
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=10,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    # 3.3 Bestes Individuum
    best = hof[0]




    

    # 4. Gerundete MA-Werte
    best_short = int(round(best[0]))
    best_long = int(round(best[1]))
    if best_short >= best_long:
        best_short, best_long = min(best_short, best_long - 1), max(best_short + 1, best_long)

    # 5. Handelsmodell mit den gerundeten MA-Werten
    data_vis = data.copy()
    data_vis['MA_short'] = data_vis['Close'].rolling(window=best_short).mean()
    data_vis['MA_long'] = data_vis['Close'].rolling(window=best_long).mean()
    data_vis.dropna(inplace=True)

    position = 0
    initial_price = None
    wealth = start_capital
    cumulative_pnl = 0
    trades = []
    positionswert = 0

    # <<< HIER: Listen für Equity und Position initialisieren
    wealth_history = []         # Wealth pro Tag (Cash + offener Positionswert)
    position_history = []       # Positionsverlauf (1=Long, -1=Short, 0=Neutral)

    for i in range(len(data_vis)):
        price_today = float(data_vis['Close'].iloc[i])
        date = data_vis.index[i]

        # Equity-Berechnung: Cash + Marktwert offener Position
        if position == 1:
            # Long: Einheiten = positionswert / initial_price
            units = positionswert / initial_price
            equity = units * price_today
        elif position == -1:
            # Short: positionswert als Margin, Gewinn/Verlust aus Preisdifferenz
            units = positionswert / initial_price
            equity = positionswert + (initial_price - price_today) * units
        else:
            # Neutral: nur Cash
            equity = wealth

        # Equity und Position speichern
        wealth_history.append(equity)
        position_history.append(position)

        # Signallogik wie gehabt
        ma_s_t = float(data_vis['MA_short'].iloc[i])
        ma_l_t = float(data_vis['MA_long'].iloc[i])
        if i > 0:
            ma_s_y = float(data_vis['MA_short'].iloc[i - 1])
            ma_l_y = float(data_vis['MA_long'].iloc[i - 1])
        else:
            ma_s_y = ma_l_y = 0

        # Kaufsignal (Long eröffnen)
        if ma_s_t > ma_l_t and ma_s_y <= ma_l_y and position == 0:
            initial_price = price_today
            position = 1
            buy_fee = 0
            positionswert = wealth - buy_fee
            wealth -= positionswert
            trades.append({
                'Typ': 'Kauf',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': buy_fee,
                'Positionswert': positionswert,
                'Profit/Loss': None,
                'Kumulative P&L': cumulative_pnl
            })

        # Verkaufssignal (Long schließen)
        if ma_s_t < ma_l_t and ma_s_y >= ma_l_y and position == 1:
            position = 0
            gross = (price_today - initial_price) / initial_price * positionswert
            sell_fee = 0
            net = gross - sell_fee
            cumulative_pnl += net
            wealth += positionswert + net
            trades.append({
                'Typ': 'Verkauf (Long)',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': sell_fee,
                'Positionswert': None,
                'Profit/Loss': net,
                'Kumulative P&L': cumulative_pnl
            })

        # Short-Signal (Short eröffnen)
        if ma_s_t < ma_l_t and ma_s_y >= ma_l_y and position == 0:
            initial_price = price_today
            position = -1
            short_fee = 0
            positionswert = wealth - short_fee
            wealth -= positionswert
            trades.append({
                'Typ': 'Short-Sell',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': short_fee,
                'Positionswert': positionswert,
                'Profit/Loss': None,
                'Kumulative P&L': cumulative_pnl
            })

        # Short-Cover + direkt neu kaufen
        if ma_s_t > ma_l_t and ma_s_y <= ma_l_y and position == -1:
            gross = (initial_price - price_today) / initial_price * positionswert
            cover_fee = 0
            net_cover = gross - cover_fee
            cumulative_pnl += net_cover
            wealth += positionswert + net_cover
            trades.append({
                'Typ': 'Short-Cover',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': cover_fee,
                'Positionswert': None,
                'Profit/Loss': net_cover,
                'Kumulative P&L': cumulative_pnl
            })

            # Direkt neues Long eröffnen
            initial_price = price_today
            position = 1
            buy_fee = 0
            positionswert = wealth - buy_fee
            wealth -= positionswert
            trades.append({
                'Typ': 'Kauf (nach Short-Cover)',
                'Datum': date,
                'Kurs': price_today,
                'Spesen': buy_fee,
                'Positionswert': positionswert,
                'Profit/Loss': None,
                'Kumulative P&L': cumulative_pnl
            })

    # Offene Position zum Schluss schließen
    if position != 0:
        last_price = float(data_vis['Close'].iloc[-1])
        last_date = data_vis.index[-1]
        if position == 1:
            gross = (last_price - initial_price) / initial_price * positionswert
        else:
            gross = (initial_price - last_price) / initial_price * positionswert
        close_fee = 0
        net_close = gross - close_fee
        cumulative_pnl += net_close
        wealth += positionswert + net_close
        trades.append({
            'Typ': 'Schließen (Ende)',
            'Datum': last_date,
            'Kurs': last_price,
            'Spesen': close_fee,
            'Positionswert': None,
            'Profit/Loss': net_close,
            'Kumulative P&L': cumulative_pnl
        })
        # Letzten Equity-Wert überschreiben, weil hier geschlossen wurde
        wealth_history[-1] = wealth

    trades_df = pd.DataFrame(trades)

    # Renditen berechnen (jetzt relativ zum start_capital)
    strategy_return = (wealth - start_capital) / start_capital * 100
    buy_and_hold_return = (data_vis['Close'].iloc[-1] - data_vis['Close'].iloc[0]) / data_vis['Close'].iloc[0] * 100

    # Statistik zu positiven/negativen Trades
    pos_trades = trades_df[trades_df['Profit/Loss'] > 0]
    neg_trades = trades_df[trades_df['Profit/Loss'] < 0]
    pos_count = len(pos_trades)
    neg_count = len(neg_trades)
    pos_pnl = pos_trades['Profit/Loss'].sum()
    neg_pnl = neg_trades['Profit/Loss'].sum()

    # Gesamtzahl der Einträge (Entry + Exit)
    total_trades = len(trades_df)

    # Anzahl aller tatsächlich abgeschlossenen Exit-Trades
    closed_trades = pos_count + neg_count

    # Korrekte Prozentberechnung: nur auf abgeschlossene Trades bezogen
    if closed_trades > 0:
        pos_pct = pos_count / closed_trades * 100
        neg_pct = neg_count / closed_trades * 100
    else:
        pos_pct = neg_pct = 0

    total_pnl = pos_pnl + neg_pnl
    if total_pnl != 0:
        pos_perf = pos_pnl / total_pnl * 100
        neg_perf = neg_pnl / total_pnl * 100
    else:
        pos_perf = neg_perf = 0

    # DataFrame für Plot vorbereiten (Close und Position)
    df_plot = data_vis[['Close']].copy().iloc[len(data_vis) - len(position_history):]
    df_plot['Position'] = position_history

    # <<< HIER: DataFrame für Equity Curve erzeugen
    df_wealth = pd.DataFrame({
        "Datum": data_vis.index,       # data_vis.index ist ein DatetimeIndex
        "Wealth": wealth_history       # wealth_history wurde im Loop befüllt
    })

    return {
        "trades_df": trades_df,
        "strategy_return": float(strategy_return),
        "buy_and_hold_return": float(buy_and_hold_return),
        "total_trades": total_trades,
        "long_trades": len(trades_df[trades_df['Typ'].str.contains("Kauf")]),
        "short_trades": len(trades_df[trades_df['Typ'].str.contains("Short")]),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "pos_pnl": float(pos_pnl),
        "neg_pnl": float(neg_pnl),
        "total_pnl": float(total_pnl),
        "pos_perf": pos_perf,
        "neg_perf": neg_perf,
        "df_plot": df_plot,
        "df_wealth": df_wealth,
        # **neu**:
        "best_individual": best,
        "logbook": logbook
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
