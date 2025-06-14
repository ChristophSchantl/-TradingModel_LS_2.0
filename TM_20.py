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
# 1) Optimierungs- und Backtest-Funktion (KORRIGIERT)
# ──────────────────────────────────────────────────────────────
def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    # 1. Daten laden mit Fehlerbehandlung
    try:
        end_date_str = datetime.now().strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start_date_str, end=end_date_str, progress=False)
        if data.empty:
            raise ValueError("Keine Daten für diesen Ticker/Zeitraum verfügbar")
    except Exception as e:
        raise ValueError(f"Fehler beim Laden der Daten: {str(e)}")

    # 2. Preisspalte auswählen
    price_col = None
    for col in ['Close', 'Adj Close']:
        if col in data.columns:
            price_col = col
            break
    if price_col is None:
        num_cols = data.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            raise ValueError("Keine numerische Preisspalte gefunden")
        price_col = num_cols[0]
    
    prices = data[price_col].interpolate().dropna()
    if len(prices) < 2:
        raise ValueError("Zu wenige Datenpunkte für Analyse")

    # 3. Fitness-Funktion mit verbesserter Logik
    def evaluate_strategy(ind):
        s, l = map(int, ind)
        if s >= l or s <= 0 or l <= 0:
            return (-np.inf,)
        
        try:
            ma_s = prices.rolling(s).mean()
            ma_l = prices.rolling(l).mean()
            df = pd.DataFrame({
                'Price': prices,
                'MA_s': ma_s,
                'MA_l': ma_l
            }).dropna()
            
            if len(df) < 2:
                return (0.0,)
                
            position = 0
            entry_price = 0.0
            equity = [start_capital]
            
            for i in range(1, len(df)):
                current_price = df['Price'].iloc[i]
                prev_ma_s = df['MA_s'].iloc[i-1]
                prev_ma_l = df['MA_l'].iloc[i-1]
                curr_ma_s = df['MA_s'].iloc[i]
                curr_ma_l = df['MA_l'].iloc[i]
                
                # Long Signal
                if curr_ma_s > curr_ma_l and prev_ma_s <= prev_ma_l:
                    if position == -1:  # Close short
                        pnl = (entry_price - current_price) / entry_price * equity[-1]
                        equity.append(equity[-1] + pnl)
                    position = 1
                    entry_price = current_price
                
                # Short Signal
                elif curr_ma_s < curr_ma_l and prev_ma_s >= prev_ma_l:
                    if position == 1:  # Close long
                        pnl = (current_price - entry_price) / entry_price * equity[-1]
                        equity.append(equity[-1] + pnl)
                    position = -1
                    entry_price = current_price
                
                equity.append(equity[-1])  # No action
            
            returns = pd.Series(equity).pct_change().dropna()
            if returns.empty or returns.std() == 0:
                return (0.0,)
                
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            return (sharpe,)
            
        except Exception:
            return (0.0,)

    # 4. DEAP Setup mit verbesserter Initialisierung
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_s", random.randint, 5, 50)
    toolbox.register("attr_l", random.randint, 10, 200)
    
    def init_individual():
        s = toolbox.attr_s()
        l = toolbox.attr_l()
        while s >= l:  # Ensure s < l
            s = toolbox.attr_s()
            l = toolbox.attr_l()
        return creator.Individual([s, l])
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_strategy)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, 
                    low=[5, 10], up=[50, 200], indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 5. GA-Lauf mit verbesserten Parametern
    pop = toolbox.population(n=30)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(1)
    
    try:
        pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.7,  # Höhere Crossover-Wahrscheinlichkeit
            mutpb=0.3,  # Höhere Mutationswahrscheinlichkeit
            ngen=20,    # Mehr Generationen
            stats=stats,
            halloffame=hof,
            verbose=False
        )
    except Exception as e:
        raise ValueError(f"Optimierung fehlgeschlagen: {str(e)}")

    # 6. Bestes Individuum auswählen
    if not hof.items:
        raise ValueError("Keine gültige Lösung gefunden")
    
    best_ind = hof[0]
    bs, bl = map(int, best_ind)
    if bs >= bl:  # Sicherheitscheck
        bs, bl = min(bs, bl-1), max(bs+1, bl)

    # 7. Backtest mit verbesserter Positionsverfolgung
    df_bt = pd.DataFrame({'Price': prices})
    df_bt['MA_s'] = df_bt['Price'].rolling(bs).mean()
    df_bt['MA_l'] = df_bt['Price'].rolling(bl).mean()
    df_bt = df_bt.dropna()
    
    if df_bt.empty:
        raise ValueError("Keine Daten nach MA-Berechnung")

    position = 0
    entry_price = 0.0
    capital = start_capital
    trades = []
    equity_history = []
    position_history = []

    for i in range(len(df_bt)):
        current_price = df_bt['Price'].iloc[i]
        current_date = df_bt.index[i]
        
        # Equity-Berechnung
        if position == 1:  # Long
            current_equity = capital * (current_price / entry_price)
        elif position == -1:  # Short
            current_equity = capital * (2 - (current_price / entry_price))
        else:
            current_equity = capital
            
        equity_history.append(current_equity)
        position_history.append(position)

        if i == 0:
            continue
            
        ma_s_prev = df_bt['MA_s'].iloc[i-1]
        ma_l_prev = df_bt['MA_l'].iloc[i-1]
        ma_s_curr = df_bt['MA_s'].iloc[i]
        ma_l_curr = df_bt['MA_l'].iloc[i]

        # Long Entry
        if ma_s_curr > ma_l_curr and ma_s_prev <= ma_l_prev and position == 0:
            position = 1
            entry_price = current_price
            trades.append({
                'Typ': 'Long Entry',
                'Datum': current_date,
                'Preis': current_price,
                'Position': position
            })
            
        # Long Exit
        elif ma_s_curr < ma_l_curr and ma_s_prev >= ma_l_prev and position == 1:
            pnl = (current_price - entry_price) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({
                'Typ': 'Long Exit',
                'Datum': current_date,
                'Preis': current_price,
                'P&L': pnl,
                'Position': position
            })
            
        # Short Entry
        elif ma_s_curr < ma_l_curr and ma_s_prev >= ma_l_prev and position == 0:
            position = -1
            entry_price = current_price
            trades.append({
                'Typ': 'Short Entry',
                'Datum': current_date,
                'Preis': current_price,
                'Position': position
            })
            
        # Short Exit
        elif ma_s_curr > ma_l_curr and ma_s_prev <= ma_l_prev and position == -1:
            pnl = (entry_price - current_price) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({
                'Typ': 'Short Exit',
                'Datum': current_date,
                'Preis': current_price,
                'P&L': pnl,
                'Position': position
            })

    # Offene Position schließen
    if position != 0:
        last_price = df_bt['Price'].iloc[-1]
        last_date = df_bt.index[-1]
        if position == 1:
            pnl = (last_price - entry_price) / entry_price * capital
        else:
            pnl = (entry_price - last_price) / entry_price * capital
            
        capital += pnl
        trades.append({
            'Typ': 'Position Close',
            'Datum': last_date,
            'Preis': last_price,
            'P&L': pnl,
            'Position': 0
        })
        equity_history[-1] = capital
        position_history[-1] = 0

    # Ergebnisse aufbereiten
    trades_df = pd.DataFrame(trades)
    if 'Datum' in trades_df.columns and pd.api.types.is_datetime64_any_dtype(trades_df['Datum']):
        trades_df['Datum'] = trades_df['Datum'].dt.strftime('%Y-%m-%d')
    
    strat_ret = (capital - start_capital) / start_capital * 100
    bh_ret = (df_bt['Price'].iloc[-1] / df_bt['Price'].iloc[0] - 1) * 100
    
    df_plot = pd.DataFrame({
        'Price': df_bt['Price'],
        'Position': position_history
    }, index=df_bt.index)
    
    df_wealth = pd.DataFrame({
        'Datum': df_bt.index,
        'Wealth': equity_history
    })

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
# 2) Streamlit-UI (KORRIGIERT)
# ──────────────────────────────────────────────────────────────
def main():
    st.title("✨ MA-Crossover Strategie-Optimierung")
    st.markdown("""
    **Moving Average Crossover Strategie** mit genetischer Algorithmus-Optimierung.  
    Der GA findet die optimalen Fenstergrößen für den Short- und Long-MA.
    """)

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Aktiensymbol (z.B. AAPL):", "AAPL").strip().upper()
    with col2:
        start_date = st.date_input(
            "Startdatum:", 
            date(2020, 1, 1), 
            max_value=date.today()
        )

    capital = st.number_input(
        "Startkapital (€):", 
        min_value=1000.0, 
        value=10000.0, 
        step=1000.0
    )

    if st.button("🚀 Strategie optimieren und ausführen"):
        with st.spinner("Optimiere MA-Parameter..."):
            try:
                results = optimize_and_run(
                    ticker=ticker,
                    start_date_str=start_date.strftime("%Y-%m-%d"),
                    start_capital=capital
                )
            except Exception as e:
                st.error(f"Fehler: {str(e)}")
                st.stop()

        # Ergebnisse anzeigen
        st.success("Optimierung erfolgreich abgeschlossen!")
        
        # 1. Optimierungsergebnisse
        st.subheader("Optimierte Parameter")
        bs, bl = results["best_individual"]
        col1, col2 = st.columns(2)
        col1.metric("Short MA Fenster", f"{bs} Tage")
        col2.metric("Long MA Fenster", f"{bl} Tage")
        
        # Konvergenzplot
        st.subheader("Optimierungsverlauf")
        df_log = pd.DataFrame({
            "Max Sharpe": results["logbook"].select("max"),
            "Durchschnitt": results["logbook"].select("avg")
        })
        st.line_chart(df_log)
        
        # 2. Performance-Vergleich
        st.subheader("Performance-Vergleich")
        strat_ret = results["strategy_return"]
        bh_ret = results["buy_and_hold_return"]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(
            ["Strategie", "Buy & Hold"], 
            [strat_ret, bh_ret],
            color=["#4c72b0", "#55a868"],
            alpha=0.7
        )
        
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height,
                f'{height:.2f}%',
                ha='center', 
                va='bottom'
            )
            
        ax.set_ylabel("Rendite (%)")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig)
        
        # 3. Equity-Kurve
        st.subheader("Equity-Kurve und Handelsphasen")
        
        fig2, ax = plt.subplots(figsize=(10, 5))
        
        # Preise plotten
        ax.plot(
            results["df_plot"].index,
            results["df_plot"]["Price"],
            label="Preis",
            color='black',
            alpha=0.7
        )
        
        # Phasen markieren
        positions = results["df_plot"]["Position"]
        current_pos = positions.iloc[0]
        start_idx = results["df_plot"].index[0]
        
        for i in range(1, len(positions)):
            if positions.iloc[i] != current_pos:
                end_idx = results["df_plot"].index[i-1]
                if current_pos == 1:
                    ax.axvspan(start_idx, end_idx, color='green', alpha=0.2)
                elif current_pos == -1:
                    ax.axvspan(start_idx, end_idx, color='red', alpha=0.2)
                current_pos = positions.iloc[i]
                start_idx = results["df_plot"].index[i]
        
        # Letzte Phase
        if current_pos == 1:
            ax.axvspan(start_idx, results["df_plot"].index[-1], color='green', alpha=0.2)
        elif current_pos == -1:
            ax.axvspan(start_idx, results["df_plot"].index[-1], color='red', alpha=0.2)
        
        # Wealth-Kurve auf zweiter Y-Achse
        ax2 = ax.twinx()
        ax2.plot(
            results["df_wealth"]["Datum"],
            results["df_wealth"]["Wealth"],
            label="Equity",
            color='blue',
            linewidth=2
        )
        
        ax.set_xlabel("Datum")
        ax.set_ylabel("Preis")
        ax2.set_ylabel("Equity (€)")
        
        # Legende kombinieren
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        st.pyplot(fig2)
        
        # 4. Trades anzeigen
        st.subheader("Handelssignale")
        if results["trades_df"].empty:
            st.warning("Keine Trades während dieses Zeitraums")
        else:
            st.dataframe(results["trades_df"], use_container_width=True)

if __name__ == "__main__":
    main()
