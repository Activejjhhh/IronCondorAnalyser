import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import norm
import time
from joblib import Parallel, delayed
import numba
import traceback

# --- Configuration (Keep as is) ---
TICKER = "QQQ"
EXPIRATION_DATE = "2025-04-17" 
NUM_SIMULATIONS = 10000
STRIKE_SPACING = 1
N_JOBS = -1

# --- Data Fetching Functions (Keep as is) ---
# ... (create_weekly_distribution function) ...
def create_weekly_distribution(ticker="QQQ", period="20y"):
    print(f"Fetching historical data for {ticker}...")
    try:
        data_yf = yf.download(ticker, period=period, interval="1d", progress=False)
        if not isinstance(data_yf, pd.DataFrame) or data_yf.empty: print(f"Error: yfinance did not return a valid DataFrame."); return None, None
        data = data_yf
        if isinstance(data_yf.columns, pd.MultiIndex):
             try:
                 if data_yf.columns.nlevels > 1: data = data_yf.droplevel(axis=1, level=1)
                 else: data = data_yf.droplevel(axis=1, level=0)
             except Exception as drop_ex: print(f"Warning: Error dropping multi-index level ({drop_ex})."); data = data_yf
        if 'Close' not in data.columns: print(f"Error: 'Close' column not found. Columns: {data.columns}"); return None, None
        weekly_data = data['Close'].resample('W').last().pct_change() * 100
        weekly_data = weekly_data.dropna()
        if weekly_data.empty: print(f"Error: No weekly data after pct_change."); return None, None
        bins = np.arange(-20, 21, 1)
        freq_table = pd.cut(weekly_data, bins=bins, right=False).value_counts(normalize=True).sort_index()
        bin_categories = pd.IntervalIndex.from_breaks(bins, closed='left')
        freq_table_full = freq_table.reindex(bin_categories, fill_value=0.0)
        if freq_table_full.sum() > 0: freq_table_full = freq_table_full / freq_table_full.sum()
        else: print("Warning: Sum of probabilities is zero.")
        return freq_table_full, freq_table_full.index
    except Exception as e: print(f"An error occurred in create_weekly_distribution: {e}"); traceback.print_exc(); return None, None

# ... (fetch_option_chain_data function) ...
def fetch_option_chain_data(ticker, expiration_date):
    print(f"Fetching option chain for {ticker}, expiry {expiration_date}...")
    try:
        stock = yf.Ticker(ticker)
        options_dates = stock.options
        if not options_dates: print(f"Error: No option dates found."); return None, None, None
        if expiration_date not in options_dates: print(f"Error: Expiration date {expiration_date} not found. Available: {options_dates}"); return None, None, None
        hist = stock.history(period="1d")
        if hist.empty or 'Close' not in hist.columns: print(f"Error: Could not fetch current price."); return None, None, None
        current_price = hist['Close'].iloc[-1]
        option_chain = stock.option_chain(expiration_date)
        calls = option_chain.calls; puts = option_chain.puts
        if calls.empty and puts.empty: print(f"Warning: Calls and puts empty."); return pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), current_price
        if not calls.empty: calls = calls.assign(type='call')
        if not puts.empty: puts = puts.assign(type='put')
        options = pd.concat([calls, puts], ignore_index=True)
        options = options[(options['bid'].notna()) & (options['ask'].notna()) & (options['bid'] > 0) & (options['ask'] > 0) & (options['ask'] >= options['bid'])]
        if options.empty: print(f"Warning: No options with valid bid/ask."); return pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), current_price
        options['mid_price'] = (options['bid'] + options['ask']) / 2
        options['strike'] = options['strike'].astype(float)
        options = options[["strike", "type", "mid_price", "impliedVolatility"]].dropna()
        calls_df = options[(options['type'] == 'call') & (options['strike'] > current_price)].copy()
        puts_df = options[(options['type'] == 'put') & (options['strike'] < current_price)].copy()
        if not puts_df.empty: puts_df.sort_values('strike', ascending=False, inplace=True)
        if not calls_df.empty: calls_df.sort_values('strike', ascending=True, inplace=True)
        print(f"Fetched {len(puts_df)} OTM puts and {len(calls_df)} OTM calls.")
        return puts_df.reset_index(drop=True), calls_df.reset_index(drop=True), current_price
    except Exception as e: print(f"Error in fetch_option_chain_data: {e}"); traceback.print_exc(); return None, None, None


# --- Numba Optimized P&L Calculation (Keep as is) ---
@numba.jit(nopython=True)
def calculate_pnl_vectorized(simulated_prices, short_put, short_call, credit, max_loss, max_profit):
    n = len(simulated_prices)
    pnl = np.empty(n, dtype=np.float64)
    for i in range(n):
        price = simulated_prices[i]
        if price >= short_put and price <= short_call: pnl[i] = credit
        elif price < short_put: pnl[i] = credit - (short_put - price)
        else: pnl[i] = credit - (price - short_call)
    pnl = np.clip(pnl, -max_loss, max_profit)
    return pnl

# --- Core Simulation Logic (MODIFIED process_condor_combo) ---

def process_condor_combo(combo, put_prices, call_prices, simulated_prices, num_simulations):
    """
    Calculates metrics for a single Iron Condor combination.
    *** Returns dict including EV, Sharpe, Kelly Pct; removes leg prices ***
    """
    long_put, short_put, short_call, long_call = combo
    try:
        # Retrieve needed prices only
        lp_price = put_prices[long_put]
        sp_price = put_prices[short_put]
        sc_price = call_prices[short_call]
        lc_price = call_prices[long_call]
    except KeyError: return None

    # --- Calculations ---
    credit = (sp_price + sc_price) - (lp_price + lc_price)
    if credit <= 0.01: return None
    put_width = short_put - long_put
    call_width = long_call - short_call
    if put_width <= 0 or call_width <= 0: return None
    required_margin = max(put_width, call_width)
    max_loss = required_margin - credit
    if max_loss <= 0.01: return None
    max_profit = credit

    # --- P&L Simulation ---
    pnl = calculate_pnl_vectorized(simulated_prices, short_put, short_call, credit, max_loss, max_profit)

    # --- Calculate Base Metrics ---
    avg_pnl = np.mean(pnl)
    wins = np.sum(pnl > 0.001) # Count wins (use small threshold)
    win_rate = (wins / num_simulations) * 100

    # --- MODIFICATION: Calculate EV, Sharpe, Kelly ---
    # Expected Value (EV) - Average outcome per trade from simulation
    ev = avg_pnl

    # Sharpe Ratio (assuming Risk-Free Rate = 0)
    std_dev_pnl = np.std(pnl)
    if std_dev_pnl > 1e-9: # Avoid division by zero or near-zero
        sharpe_ratio = avg_pnl / std_dev_pnl
    else:
        # If std dev is zero, all outcomes were the same.
        # If avg_pnl > 0, it's infinite Sharpe (riskless profit in simulation)
        # If avg_pnl <= 0, Sharpe is 0 or negative infinity. Let's use 0.
        sharpe_ratio = np.inf if avg_pnl > 1e-9 else 0.0

    # Kelly Criterion Percentage
    kelly_pct = 0.0 # Default value
    W = win_rate / 100.0 # Win probability

    if W > 0 and W < 1: # Need both wins and losses for standard Kelly calc
        wins_pnl = pnl[pnl > 0.001]
        losses_pnl = pnl[pnl <= 0.001]

        avg_win = np.mean(wins_pnl) if len(wins_pnl) > 0 else 0
        # Use absolute value for average loss amount
        avg_loss = np.mean(np.abs(losses_pnl)) if len(losses_pnl) > 0 else 0

        if avg_loss > 1e-9: # Ensure avg loss is not zero to calculate R
            R = avg_win / avg_loss # Win/Loss Ratio
            if R > 0: # Ensure R is positive
                 kelly_pct = W - ((1.0 - W) / R)
            # If R is 0 (avg_win is 0), Kelly would be negative, indicating don't bet.
            # Default kelly_pct=0.0 handles this implicitly.
        else:
            # If avg_loss is effectively zero (only wins or zero-P&L outcomes)
            # and W < 1 (meaning some zero P&L outcomes exist),
            # Kelly is theoretically W. Let's be conservative and cap or use W.
            # If W=1, this case isn't hit. Let's set Kelly to W here.
             kelly_pct = W # Bet fraction equal to win probability if no losses occur

    elif W >= 1.0: # 100% win rate in simulation
        kelly_pct = 1.0 # Bet max recommended (theoretically)

    # --- END MODIFICATION ---

    # Other metrics (calculated previously)
    roi = (avg_pnl / max_loss) * 100 if max_loss > 0.01 else np.inf
    risk_reward_ratio = max_profit / max_loss if max_loss > 0.01 else float('inf')

    # --- Return dictionary (Updated) ---
    return {
        # Strategy Legs
        "Long Put": long_put, "Short Put": short_put, "Short Call": short_call, "Long Call": long_call,
        # Debugging Info (Widths)
        "Put Spread Width": round(put_width, 2), "Call Spread Width": round(call_width, 2),
         # Key Metrics
        "Net Premium Received (Open)": round(credit, 2),
        "Margin Req.": round(required_margin, 2),
        "Max Loss": round(max_loss, 2),
        "Credit (Max Profit)": round(max_profit, 2),
        # Performance Metrics
        "Win Rate (%)": round(win_rate, 2),
        "Avg P&L": round(avg_pnl, 4),
        "EV": round(ev, 4),                          # <-- NEW
        "Sharpe Ratio": round(sharpe_ratio, 4) if sharpe_ratio != np.inf else 'inf', # <-- NEW
        "Kelly Pct": round(kelly_pct * 100, 2),      # <-- NEW (as percentage)
        "Risk-Reward Ratio": round(risk_reward_ratio, 2) if risk_reward_ratio != float('inf') else 'inf',
        "ROI (%)": round(roi, 2) if roi != float('inf') else 'inf'
        # Removed: LP Price, SP Price, SC Price, LC Price
    }


# --- Main Simulation Function (MODIFIED FOR COLUMN ORDER) ---
def simulate_iron_condor_performance_optimized(puts_df, calls_df, distribution, bins_categories, current_price, num_simulations=1000, strike_spacing=1):
    """
    Simulates Iron Condor performance using optimized techniques.
    *** Saves output to CSV with EV, Sharpe, Kelly; removes leg prices ***
    """
    # ... (Initial checks, pre-computation, sampling, combo generation, parallel processing remain the same) ...
    if puts_df is None or calls_df is None or distribution is None or bins_categories is None: print("Missing input data."); return None
    if puts_df.empty or calls_df.empty: print("No OTM puts or calls available."); return None

    # === Pre-computation & Setup ===
    if strike_spacing > 0:
        puts_df = puts_df[puts_df['strike'] % strike_spacing == 0].copy()
        calls_df = calls_df[calls_df['strike'] % strike_spacing == 0].copy()
        if puts_df.empty or calls_df.empty: print(f"No strikes left after spacing filter {strike_spacing}."); return None
    put_prices = puts_df.set_index('strike')['mid_price'].to_dict()
    call_prices = calls_df.set_index('strike')['mid_price'].to_dict()
    put_strikes = sorted(put_prices.keys(), reverse=True)
    call_strikes = sorted(call_prices.keys())
    if not put_strikes or not call_strikes or len(put_strikes) < 2 or len(call_strikes) < 2: print("Not enough unique strikes."); return None

    # === Vectorized Sampling ===
    print(f"Generating {num_simulations} price scenarios...")
    # (Sampling logic unchanged)
    if not isinstance(distribution, pd.Series) or not isinstance(bins_categories, pd.IntervalIndex): print("Error: Incorrect types for distribution/bins."); return None
    bin_probs = distribution.values
    if not np.isclose(bin_probs.sum(), 1.0):
        print(f"Warning: Probabilities sum to {bin_probs.sum()}, renormalizing.")
        if bin_probs.sum() > 0: bin_probs /= bin_probs.sum()
        else: print("Error: Probabilities sum to zero."); return None
    bin_left_edges = bins_categories.left.to_numpy(); bin_right_edges = bins_categories.right.to_numpy()
    sampled_bin_indices = np.random.choice(len(bins_categories), size=num_simulations, p=bin_probs)
    sampled_moves = np.random.uniform(bin_left_edges[sampled_bin_indices], bin_right_edges[sampled_bin_indices])
    simulated_prices = current_price * (1 + sampled_moves / 100.0)
    print("Price scenarios generated.")

    # === Precompute Valid Strike Combinations ===
    print("Generating valid condor combinations...")
    valid_combos = []
    # (Combo generation loop unchanged)
    for i in range(len(put_strikes)):
        sp = put_strikes[i]
        for j in range(i + 1, len(put_strikes)):
            lp = put_strikes[j]
            for k in range(len(call_strikes)):
                sc = call_strikes[k]
                if sc > sp:
                    for l in range(k + 1, len(call_strikes)): lc = call_strikes[l]; valid_combos.append((lp, sp, sc, lc))
    if not valid_combos: print("No valid condor combinations found."); return None
    print(f"Generated {len(valid_combos)} potential condor strategies.")

    # === Parallel Processing ===
    print(f"Simulating P&L for {len(valid_combos)} combos using {N_JOBS if N_JOBS != -1 else 'all'} cores...")
    start_time = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=10)(delayed(process_condor_combo)(combo, put_prices, call_prices, simulated_prices, num_simulations) for combo in valid_combos)
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    results = [r for r in results if r is not None]
    if not results: print("No valid simulation results."); return None

    # === Results Processing ===
    results_df = pd.DataFrame(results)

    # --- MODIFICATION: Define column order INCLUDING new metrics ---
    desired_columns = [
        # Strategy Legs
        "Long Put", "Short Put", "Short Call", "Long Call",
        # Debugging Info - Widths
        "Put Spread Width", "Call Spread Width",
        # Key Metrics
        "Net Premium Received (Open)",
        "Margin Req.",
        "Max Loss",
        "Credit (Max Profit)",
        # Performance Metrics
        "Win Rate (%)",
        "Avg P&L",
        "EV",                 # <-- NEW
        "Sharpe Ratio",       # <-- NEW
        "Kelly Pct",          # <-- NEW
        "Risk-Reward Ratio",
        "ROI (%)"
        # Removed: LP Price, SP Price, SC Price, LC Price
    ]
    existing_columns = [col for col in desired_columns if col in results_df.columns]
    results_df = results_df[existing_columns]
    # --- END MODIFICATION ---

    # Sort results (e.g., by ROI or Sharpe Ratio)
    sort_column = "ROI (%)" # Or "Sharpe Ratio" or "EV"
    if sort_column in results_df.columns:
        results_df = results_df.sort_values(by=sort_column, ascending=False)
        print(f"\nSorting results by {sort_column}")
    else:
        print(f"Warning: Sort column '{sort_column}' not found.")


    # Display top results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1800) # Increased width further
    pd.set_option('display.max_rows', 100)
    print(f"\n🏆 **Top {min(100, len(results_df))} Iron Condor Strategies (Sorted by {sort_column}):**\n")
    print(results_df.head(100).to_string(index=False))

    # --- Save to CSV (Unchanged) ---
    try:
        output_filename = f"iron_condor_analysis_{TICKER}_{EXPIRATION_DATE}.csv"
        print(f"\nSaving results to {output_filename} (CSV format)...")
        results_df.to_csv(output_filename, index=False)
        print("Results saved successfully as CSV.")
    except Exception as e: print(f"\nError saving results to CSV: {e}"); traceback.print_exc()

    return results_df


# --- Main Execution (Keep as is) ---
if __name__ == "__main__":
    print("Starting Iron Condor Analysis...")
    start_overall_time = time.time()
    # (Steps 1, 2, 3 unchanged)
    distribution, bin_categories = create_weekly_distribution(ticker=TICKER)
    if distribution is None or bin_categories is None: print("Exiting: Error fetching historical data."); exit()
    puts, calls, current_price = fetch_option_chain_data(TICKER, EXPIRATION_DATE)
    if puts is None or calls is None or current_price is None: print("Exiting: Error fetching option data."); exit()
    if puts.empty or calls.empty: print(f"Exiting: No OTM puts or calls found for {TICKER} {EXPIRATION_DATE}."); exit()
    condor_df = simulate_iron_condor_performance_optimized(
        puts_df=puts, calls_df=calls, distribution=distribution, bins_categories=bin_categories,
        current_price=current_price, num_simulations=NUM_SIMULATIONS, strike_spacing=STRIKE_SPACING)
    end_overall_time = time.time()
    if condor_df is None: print("\nSimulation did not produce results.")
    else: print(f"\nAnalysis complete. Total time: {end_overall_time - start_overall_time:.2f} seconds.")
