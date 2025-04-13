# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Import the backend function
from condor_backend import run_full_simulation # Assuming your backend is correct now

# --- Page Configuration ---
st.set_page_config(page_title="Iron Condor Analyzer", layout="wide")

# --- Caching Function for Expirations (Unchanged) ---
@st.cache_data(ttl=3600)
def get_expiration_dates_cached(ticker):
    if not ticker: return []
    try:
        stock = yf.Ticker(ticker); expirations = stock.options
        if not expirations: return None
        valid_expirations = [d for d in expirations if isinstance(d, str) and len(d) == 10]
        return valid_expirations
    except Exception as e:
        print(f"Error fetching expirations for {ticker}: {e}"); return []

# --- Plotting Function (Unchanged) ---
def plot_pnl(lp, sp, sc, lc, credit):
    try:
        margin_factor = 1.5; put_width = sp - lp; call_width = lc - sc
        plot_min = lp - put_width * (margin_factor - 1); plot_max = lc + call_width * (margin_factor - 1)
        if plot_min >= plot_max: plot_min = sp * 0.8; plot_max = sc * 1.2
        x_prices = np.linspace(plot_min, plot_max, 300); y_pnl = np.zeros_like(x_prices)
        max_profit = credit; max_loss = max(put_width, call_width) - credit
        y_pnl[(x_prices >= sp) & (x_prices <= sc)] = max_profit
        put_loss_indices = x_prices < sp; y_pnl[put_loss_indices] = max_profit - (sp - x_prices[put_loss_indices])
        call_loss_indices = x_prices > sc; y_pnl[call_loss_indices] = max_profit - (x_prices[call_loss_indices] - sc)
        y_pnl = np.clip(y_pnl, -max_loss, max_profit)
        lower_be = sp - credit; upper_be = sc + credit
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_prices, y_pnl, label='P&L @ Expiry', color='blue', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Breakeven')
        ax.axhline(max_profit, color='green', linestyle=':', linewidth=0.8, label=f'Max P (${max_profit:.2f})')
        ax.axhline(-max_loss, color='red', linestyle=':', linewidth=0.8, label=f'Max L (${max_loss:.2f})')
        strikes = {'LP': lp, 'SP': sp, 'SC': sc, 'LC': lc}; strike_colors = {'LP': 'darkred', 'SP': 'red', 'SC': 'lightgreen', 'LC': 'darkgreen'}
        for label, strike in strikes.items(): ax.axvline(strike, color=strike_colors[label], linestyle='--', linewidth=0.7, label=f'{label} ({strike:.0f})') # Use .0f for strike formatting
        ax.axvline(lower_be, color='orange', linestyle='-.', linewidth=0.9, label=f'Lower BE ({lower_be:.2f})')
        ax.axvline(upper_be, color='orange', linestyle='-.', linewidth=0.9, label=f'Upper BE ({upper_be:.2f})')
        ax.set_title(f'Condor P&L ({lp:.0f}/{sp:.0f} P, {sc:.0f}/{lc:.0f} C)'); ax.set_xlabel('Underlying Price @ Expiry'); ax.set_ylabel('Profit / Loss ($)')
        ax.grid(True, linestyle=':', alpha=0.6); ax.legend(fontsize='small', loc='best')
        fig.tight_layout(); return fig
    except Exception as plot_e: st.error(f"Error plotting: {plot_e}"); return None

# --- App Layout ---
st.title("📈 Iron Condor Strategy Analyzer")

# Input sections in columns
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Analysis Setup")
    ticker_input = st.text_input("Ticker Symbol:", value="QQQ", key="ticker").upper()
    # Expiration selection... (unchanged)
    expirations = [];
    if ticker_input:
        expirations = get_expiration_dates_cached(ticker_input)
        if expirations is None: st.warning(f"No dates for {ticker_input}.")
        elif not expirations: st.warning(f"Could not get dates for {ticker_input}.")
        expirations = expirations or []
        try: today_str = datetime.now().strftime('%Y-%m-%d'); expirations = [d for d in expirations if d >= today_str]
        except Exception: pass
    selected_expiration = st.selectbox("Expiration Date:", options=expirations, index=0 if expirations else None, disabled=not expirations)

with c2:
    st.subheader("Simulation Parameters")
    # Group simulation parameters
    hist_period_input = st.selectbox("Historical Period:", options=["3y", "5y", "7y", "10y", "15y", "20y"], index=3, key="hist_period")
    simulation_method = st.selectbox("Simulation Method:", options=['Historical Bins', 'Weighted Historical', 'Implied Volatility'], index=0, key="sim_method")
    num_sims_input = st.number_input("Number of Simulations:", min_value=1000, max_value=50000, value=10000, step=1000, key="num_sims")
    strike_sp_input = st.number_input("Strike Spacing Filter:", min_value=0, max_value=20, value=1, step=1, key="strike_sp")

# Action Button below inputs
analyze_button = st.button("Analyze Condors", key="analyze", type="primary", use_container_width=True, disabled=not ticker_input or not selected_expiration)

st.divider() # Visually separate inputs from outputs

# --- Results Area ---
st.subheader("Analysis Results")
status_placeholder = st.empty() # Area for messages (spinner, success, error)

# --- Analysis Execution & Display Logic ---
# Use session state to keep results
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'error_message' not in st.session_state: st.session_state.error_message = None
if 'run_time' not in st.session_state: st.session_state.run_time = 0
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

if analyze_button:
    # Reset previous results when button is clicked
    st.session_state.results_df = None
    st.session_state.error_message = None
    st.session_state.analysis_done = False
    st.session_state.run_time = 0

    if ticker_input and selected_expiration:
        # Date check... (unchanged)
        is_future_date = False;
        try:
            if datetime.strptime(selected_expiration, '%Y-%m-%d').date() > datetime.now().date(): is_future_date = True
        except Exception: pass
        if not is_future_date and simulation_method == 'Implied Volatility':
            st.session_state.error_message = "Implied Volatility sim requires future date."
            st.session_state.analysis_done = True # Mark as 'done' to show error
        else:
            with status_placeholder, st.spinner(f"Analyzing {ticker_input} ({selected_expiration})..."):
                start_run_time = time.time()
                results, error_msg = run_full_simulation(ticker=ticker_input, expiration_date=selected_expiration, hist_period=hist_period_input, num_sims=num_sims_input, strike_sp=strike_sp_input, simulation_method=simulation_method)
                st.session_state.results_df = results
                st.session_state.error_message = error_msg # Store potential error
                end_run_time = time.time(); st.session_state.run_time = end_run_time - start_run_time
                st.session_state.analysis_done = True # Mark analysis as complete

# --- Display results ---
# This block now runs if the analysis_done flag is True (set after button press)
if st.session_state.analysis_done:
    results_df = st.session_state.results_df
    error_message = st.session_state.error_message
    run_time = st.session_state.run_time

    status_placeholder.empty() # Clear spinner area

    if error_message:
        status_placeholder.error(error_message) # Show error message here
    elif results_df is not None and not results_df.empty:
        total_strategies_found = len(results_df)
        # Show success message
        status_placeholder.success(f"Analysis complete! Found {total_strategies_found} strategies ({simulation_method}) in {run_time:.2f}s.")

        # --- P&L Plot Section ---
        st.subheader("P&L Profile Plot")
        # Select strategy (show top N options for selection)
        # Add num_rows_to_display input here, specific to the plot selector
        num_rows_for_plot_select = st.number_input(
            "Select Top N strategies for Plotting:",
             min_value=10,
             max_value=min(500, total_strategies_found), # Sensible max
             value=min(50, total_strategies_found), # Default to 50 or total found
             step=10, key="plot_select_n",
             help="Limits the dropdown list for selecting a strategy to plot."
             )
        df_for_plot_select = results_df.head(int(num_rows_for_plot_select))
        strategy_options = {
            f"#{i} (ROI: {row.get('ROI (%)','N/A'):.2f}%): {row['Long Put']:.0f}/{row['Short Put']:.0f}P, {row['Short Call']:.0f}/{row['Long Call']:.0f}C": i
            for i, row in df_for_plot_select.iterrows()
            }
        options_list = ["Select a strategy..."] + list(strategy_options.keys())
        selected_strategy_label = st.selectbox("Choose strategy to plot:", options=options_list, index=0)

        plot_display_area = st.empty() # Placeholder specifically for the plot image
        if selected_strategy_label != "Select a strategy...":
            selected_index = strategy_options[selected_strategy_label]
            selected_row = results_df.loc[selected_index] # Get data from original df
            fig = plot_pnl(selected_row['Long Put'], selected_row['Short Put'], selected_row['Short Call'], selected_row['Long Call'], selected_row['Net Premium Received (Open)'])
            if fig:
                plot_display_area.pyplot(fig) # Display plot in its dedicated area
            else:
                plot_display_area.warning("Could not generate plot for the selected strategy.")


        st.divider()

        # --- Full Results Table Section ---
        st.subheader(f"Full Results Table ({total_strategies_found} Strategies)")
        # Add num_rows_to_display for the main table here
        num_rows_main_table = st.number_input(
            "Rows to Display in Table:",
            min_value=10,
            max_value=total_strategies_found + 10, # Allow slightly more than found
            value=min(50, total_strategies_found), # Default
            step=10,
            key="main_table_rows",
            help="Number of rows to display in the main results table below."
            )
        st.dataframe(results_df.head(int(num_rows_main_table)), height=600) # Display SLICED dataframe
        # Download button remains for FULL results
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(label=f"Download All {total_strategies_found} Results as CSV", data=csv, file_name=f"condor_{ticker_input}_{selected_expiration}_{simulation_method.replace(' ','_')}.csv", mime='text/csv')

    elif results_df is not None and results_df.empty:
         # If results_df is empty after simulation
         status_placeholder.warning(f"Analysis ran ({simulation_method}), but no valid strategies generated results (check strikes, filters, or data).")
    elif error_message is None: # If analysis wasn't triggered or backend had unknown issue
         status_placeholder.info("Enter parameters and click 'Analyze Condors'.")
