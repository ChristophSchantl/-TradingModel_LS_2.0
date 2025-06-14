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
# 1) Optimierungs- und Backtest-Funktion (VOLLSTÄNDIG KORRIGIERT)
# ──────────────────────────────────────────────────────────────
def optimize_and_run(ticker: str, start_date_str: str, start_capital: float):
    # 1. Daten mit robuster Fehlerbehandlung laden
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        data = yf.download(ticker, start=start_date_str, end=end_date, progress=False)
        if data.empty:
            raise ValueError("Keine Daten für diesen Ticker/Zeitraum verfügbar")
    except Exception as e:
        raise ValueError(f"Datenladefehler: {str(e)}")

    # 2. Preisspalte identifizieren
    price_col = next((col for col in ['Close', 'Adj Close'] if col in data.columns), None)
    if price_col is None:
        num_cols = data.select_dtypes(include=np.number).columns
        price_col = num_cols[0] if len(num_cols) > 0 else None
    
    if price_col is None:
        raise ValueError("Keine numerische Preisspalte gefunden")
    
    prices = data[price_col].interpolate().dropna()
    if len(prices) < 10:
        raise ValueError("Zu wenige Datenpunkte (mind. 10 benötigt)")

    # 3. DEAP Initialisierung
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    # 4. Fitness-Funktion mit vollständiger Positionsverfolgung
    def evaluate(individual):
        short, long = map(int, individual)
        if short >= long or short <= 0:
            return (-np.inf,)
        
        try:
            ma_short = prices.rolling(short).mean()
            ma_long = prices.rolling(long).mean()
            df = pd.DataFrame({
                'Price': prices,
                'MA_Short': ma_short,
                'MA_Long': ma_long
            }).dropna()
            
            if len(df) < 2:
                return (0.0,)
            
            position = 0  # 0=neutral, 1=long, -1=short
            entry_price = 0.0
            equity = [start_capital]
            
            for i in range(1, len(df)):
                current_price = df['Price'].iloc[i]
                prev_short = df['MA_Short'].iloc[i-1]
                prev_long = df['MA_Long'].iloc[i-1]
                curr_short = df['MA_Short'].iloc[i]
                curr_long = df['MA_Long'].iloc[i]
                
                # Long-Signal
                if curr_short > curr_long and prev_short <= prev_long:
                    if position == -1:  # Short schließen
                        pnl = (entry_price - current_price) / entry_price * equity[-1]
                        equity.append(equity[-1] + pnl)
                    position = 1
                    entry_price = current_price
                
                # Short-Signal
                elif curr_short < curr_long and prev_short >= prev_long:
                    if position == 1:  # Long schließen
                        pnl = (current_price - entry_price) / entry_price * equity[-1]
                        equity.append(equity[-1] + pnl)
                    position = -1
                    entry_price = current_price
                
                equity.append(equity[-1])  # Keine Änderung
            
            returns = pd.Series(equity).pct_change().dropna()
            if returns.empty or returns.std() == 0:
                return (0.0,)
                
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            return (sharpe,)
            
        except Exception:
            return (0.0,)

    # 5. GA Setup
    toolbox = base.Toolbox()
    toolbox.register("attr_short", random.randint, 2, 50)
    toolbox.register("attr_long", random.randint, 10, 200)
    
    def init_individual():
        s = toolbox.attr_short()
        l = toolbox.attr_long()
        while s >= l:  # Sicherstellen dass short < long
            s = toolbox.attr_short()
            l = toolbox.attr_long()
        return creator.Individual([s, l])
    
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=[2,10], up=[50,200], indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 6. GA-Ausführung
    pop = toolbox.population(n=50)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    hof = tools.HallOfFame(1)
    
    try:
        pop, logbook = algorithms.eaSimple(
            pop, toolbox,
            cxpb=0.7,
            mutpb=0.3,
            ngen=25,
            stats=stats,
            halloffame=hof,
            verbose=False
        )
    except Exception as e:
        raise ValueError(f"Optimierung fehlgeschlagen: {str(e)}")

    if not hof.items:
        raise ValueError("Keine gültige Lösung gefunden")
    
    best = hof[0]
    bs, bl = map(int, best)
    if bs >= bl:  # Sicherheitscheck
        bs, bl = min(bs, bl-1), max(bs+1, bl)

    # 7. Backtest mit optimierten Parametern
    df_bt = pd.DataFrame({'Price': prices})
    df_bt['MA_Short'] = df_bt['Price'].rolling(bs).mean()
    df_bt['MA_Long'] = df_bt['Price'].rolling(bl).mean()
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
        
        # Equity berechnen
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
            
        # Signalerkennung
        prev_short = df_bt['MA_Short'].iloc[i-1]
        prev_long = df_bt['MA_Long'].iloc[i-1]
        curr_short = df_bt['MA_Short'].iloc[i]
        curr_long = df_bt['MA_Long'].iloc[i]

        # Long Entry
        if curr_short > curr_long and prev_short <= prev_long and position == 0:
            position = 1
            entry_price = current_price
            trades.append({
                'Typ': 'Long Entry',
                'Datum': current_date,
                'Preis': current_price,
                'Position': position,
                'P&L': None
            })
            
        # Long Exit
        elif curr_short < curr_long and prev_short >= prev_long and position == 1:
            pnl = (current_price - entry_price) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({
                'Typ': 'Long Exit',
                'Datum': current_date,
                'Preis': current_price,
                'Position': position,
                'P&L': pnl
            })
            
        # Short Entry
        elif curr_short < curr_long and prev_short >= prev_long and position == 0:
            position = -1
            entry_price = current_price
            trades.append({
                'Typ': 'Short Entry',
                'Datum': current_date,
                'Preis': current_price,
                'Position': position,
                'P&L': None
            })
            
        # Short Exit
        elif curr_short > curr_long and prev_short <= prev_long and position == -1:
            pnl = (entry_price - current_price) / entry_price * capital
            capital += pnl
            position = 0
            trades.append({
                'Typ': 'Short Exit',
                'Datum': current_date,
                'Preis': current_price,
                'Position': position,
                'P&L': pnl
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
            'Typ': 'Final Exit',
            'Datum': last_date,
            'Preis': last_price,
            'Position': 0,
            'P&L': pnl
        })
        equity_history[-1] = capital
        position_history[-1] = 0

    # 8. Ergebnisse aufbereiten
    trades_df = pd.DataFrame(trades if trades else [{
        'Typ': 'Keine Trades',
        'Datum': df_bt.index[0] if len(df_bt) > 0 else pd.NaT,
        'Preis': 0,
        'Position': 0,
        'P&L': 0
    }])
    
    if pd.api.types.is_datetime64_any_dtype(trades_df['Datum']):
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
    }).set_index('Datum')

    return {
        "best_params": (bs, bl),
        "logbook": logbook,
        "trades": trades_df,
        "strategy_return": strat_ret,
        "buy_hold_return": bh_ret,
        "price_data": df_plot,
        "wealth_data": df_wealth
    }

# ──────────────────────────────────────────────────────────────
# 2) Streamlit UI (VOLLSTÄNDIG KORRIGIERT)
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="MA-Crossover Optimierung", layout="wide")
    st.title("✨ Moving Average Crossover Strategie-Optimierung")
    
    with st.expander("ℹ️ Info", expanded=True):
        st.markdown("""
        **Strategie**:  
        - Kaufsignal wenn Short-MA über Long-MA kreuzt  
        - Verkaufssignal wenn Short-MA unter Long-MA kreuzt  
        - **Genetischer Algorithmus** findet optimale Fenstergrößen  
        """)

    # Eingabefelder
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Aktiensymbol (z.B. AAPL):", "AAPL").strip().upper()
    with col2:
        start_date = st.date_input("Startdatum:", date(2020, 1, 1), max_value=date.today())
    with col3:
        capital = st.number_input("Startkapital (€):", min_value=1000.0, value=10000.0, step=1000.0)

    if st.button("🚀 Optimierung starten", type="primary"):
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
        
        st.success("Optimierung erfolgreich abgeschlossen!")
        st.divider()

        # 1. Optimierungsergebnisse
        st.subheader("Optimierte Parameter")
        bs, bl = results["best_params"]
        cols = st.columns(4)
        cols[0].metric("Short MA", f"{bs} Tage")
        cols[1].metric("Long MA", f"{bl} Tage")
        cols[2].metric("Strategie-Rendite", f"{results['strategy_return']:.2f}%")
        cols[3].metric("Buy & Hold", f"{results['buy_hold_return']:.2f}%")

        # 2. Optimierungsverlauf
        st.subheader("Optimierungsverlauf (Sharpe Ratio)")
        df_log = pd.DataFrame(results["logbook"])
        st.line_chart(df_log[["max", "avg"]])

        # 3. Preis- und Equity-Chart
        st.subheader("Preisentwicklung & Strategie-Performance")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Preisplot
        ax.plot(
            results["price_data"].index,
            results["price_data"]["Price"],
            label="Preis",
            color='black',
            alpha=0.7
        )
        ax.set_ylabel("Preis", color='black')
        ax.tick_params(axis='y', labelcolor='black')
        
        # Phasenmarkierung
        pos = results["price_data"]["Position"]
        current_pos = pos.iloc[0] if len(pos) > 0 else 0
        start_idx = results["price_data"].index[0] if len(pos) > 0 else None
        
        for i in range(1, len(pos)):
            if pos.iloc[i] != current_pos:
                end_idx = results["price_data"].index[i-1]
                if current_pos == 1:
                    ax.axvspan(start_idx, end_idx, color='green', alpha=0.1)
                elif current_pos == -1:
                    ax.axvspan(start_idx, end_idx, color='red', alpha=0.1)
                current_pos = pos.iloc[i]
                start_idx = results["price_data"].index[i]
        
        if start_idx is not None:
            if current_pos == 1:
                ax.axvspan(start_idx, results["price_data"].index[-1], color='green', alpha=0.1)
            elif current_pos == -1:
                ax.axvspan(start_idx, results["price_data"].index[-1], color='red', alpha=0.1)
        
        # Equity Plot auf zweiter Y-Achse
        ax2 = ax.twinx()
        ax2.plot(
            results["wealth_data"].index,
            results["wealth_data"]["Wealth"],
            label="Equity",
            color='blue',
            linewidth=1.5
        )
        ax2.set_ylabel("Equity (€)", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        # Legende
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        st.pyplot(fig)

        # 4. Trades anzeigen
        st.subheader("Handelssignale")
        if len(results["trades"]) <= 1:
            st.warning("Keine relevanten Trades im gewählten Zeitraum")
        else:
            st.dataframe(
                results["trades"],
                column_config={
                    "Datum": st.column_config.DatetimeColumn("Datum"),
                    "Preis": st.column_config.NumberColumn("Preis", format="%.2f"),
                    "P&L": st.column_config.NumberColumn("P&L", format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )

if __name__ == "__main__":
    main()
