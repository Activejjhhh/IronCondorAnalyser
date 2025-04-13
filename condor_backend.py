# condor_backend.py
import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
from scipy.stats import norm
import time
from joblib import Parallel, delayed
import numba
import traceback
from datetime import datetime
from py_vollib.black_scholes import black_scholes # For Greeks
from py_vollib.black_scholes.greeks import analytical as greeks_analytical # For Greeks

# --- Configuration ---
N_JOBS = -1 # Use -1 for local machine power


def create_weekly_distribution(ticker, period):
    """
    Fetches historical data, creates weekly return distribution AND returns raw returns.
    Corrected handling for single-ticker download results.
    """
    print(f"Fetching historical data for {ticker} (period: {period})...")
    raw_weekly_returns = None
    try:
        # Download adjusted data
        data_yf = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)

        if not isinstance(data_yf, pd.DataFrame) or data_yf.empty:
            print(f"Error: yfinance download failed or returned empty DataFrame for {ticker}.")
            return None, None, None

        # --- START OF CHANGES ---

        # 1. Check if 'Close' column exists
        if 'Close' not in data_yf.columns:
            print(f"Error: 'Close' column not found in downloaded data. Columns: {data_yf.columns}")
            return None, None, None

        # 2. Select the 'Close' column to ensure we have price data
        close_prices = data_yf['Close']

        # 3. Explicitly ensure it's a pandas Series
        #    (This handles cases where yfinance might return a single-column DataFrame)
        if not isinstance(close_prices, pd.Series):
            if isinstance(close_prices, pd.DataFrame) and len(close_prices.columns) == 1:
                # If it's a DataFrame with one column, convert it to a Series
                print("Warning: Downloaded 'Close' data was a DataFrame, converting to Series.")
                close_prices = close_prices.iloc[:, 0]
            else:
                # If it's something else unexpected, we can't proceed
                print(f"Error: 'Close' price data is not a pandas Series or convertible DataFrame. Type: {type(close_prices)}")
                return None, None, None

        # --- END OF CHANGES ---

        # 4. Ensure index is DatetimeIndex before resampling
        if not isinstance(close_prices.index, pd.DatetimeIndex):
             # This might happen if the index wasn't parsed correctly initially
             try:
                 close_prices.index = pd.to_datetime(close_prices.index)
                 print("Converted index to DatetimeIndex.")
             except Exception as idx_e:
                 print(f"Error converting index to DatetimeIndex: {idx_e}")
                 return None, None, None


        # --- Calculate weekly returns (Now using the guaranteed 'close_prices' Series) ---
        weekly_close = close_prices.resample('W').ffill()
        weekly_data = weekly_close.pct_change() * 100 # weekly_data should definitely be a Series now
        weekly_data = weekly_data.dropna()
        raw_weekly_returns = weekly_data

        if weekly_data.empty:
            print(f"Error: No weekly data after pct_change for {ticker}.")
            return None, None, None # Return None for all if no data

        # --- Robust Binning (This should now work correctly) ---
        min_ret, max_ret = weekly_data.min(), weekly_data.max() # These should be scalars now

        # Check if min/max are valid scalars
        if not np.isscalar(min_ret) or not np.isscalar(max_ret) or np.isnan(min_ret) or np.isnan(max_ret):
             print(f"Warning: Invalid min/max return determined ({min_ret}, {max_ret}). Using default bins.")
             bins = np.arange(-20, 21, 1)
        elif np.isclose(min_ret, max_ret): # Check if min/max are close
             bins = np.array([min_ret - 0.5, min_ret + 0.5])
             print(f"Warn: All weekly returns near {min_ret:.2f}. Using single bin.")
        else: # Calculate dynamic bins
            bin_min = np.floor(min_ret / 5) * 5 - 5
            bin_max = np.ceil(max_ret / 5) * 5 + 6
            if bin_max <= bin_min: # Check order
                print(f"Warning: bin_max ({bin_max:.2f}) <= bin_min ({bin_min:.2f}). Using default bins.")
                bins = np.arange(-20, 21, 1)
            else:
                bins = np.arange(bin_min, bin_max, 1)

        # Ensure at least 2 bins for pd.cut
        if len(bins) < 2:
             print("Warning: Bin calculation resulted in < 2 bins. Using default.")
             bins = np.arange(-20, 21, 1)

        # --- Frequency Table Calculation (pd.cut expects 1D array/Series) ---
        # weekly_data should be a Series here, which is acceptable 1D input
        freq_table = pd.cut(weekly_data, bins=bins, right=False, include_lowest=True).value_counts(normalize=True).sort_index()
        bin_categories = pd.IntervalIndex.from_breaks(bins, closed='left')
        freq_table_full = freq_table.reindex(bin_categories, fill_value=0.0)

        # Normalize probabilities
        if freq_table_full.sum() > 0:
            freq_table_full = freq_table_full / freq_table_full.sum()
        else:
            print("Warning: Sum of probabilities is zero after binning/reindexing.")
            return None, None, raw_weekly_returns

        print("Historical distribution created.")
        return freq_table_full, freq_table_full.index, raw_weekly_returns

    # --- Catch specific potential errors and general errors ---
    except ValueError as ve:
         # Catch ValueErrors which might come from pd.cut or resampling
         print(f"ValueError in create_weekly_distribution: {ve}")
         traceback.print_exc()
         return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred in create_weekly_distribution: {e}")
        traceback.print_exc()
        return None, None, None

# ... (rest of condor_backend.py remains the same: fetch_option_chain_data, calculate_pnl_vectorized, process_condor_combo, run_full_simulation) ...


def fetch_option_chain_data(ticker, expiration_date):
    """
    Fetches and preprocesses option chain data.
    Returns puts_df, calls_df, current_price, error_message.
    """
    print(f"Fetching option chain for {ticker}, expiry {expiration_date}...")
    try:
        stock = yf.Ticker(ticker)
        try:
            options_dates = stock.options
        except Exception as oe:
            print(f"Error fetching option dates for {ticker}: {oe}")
            return None, None, None, f"Error fetching option dates: {oe}"

        if not options_dates:
            return None, None, None, f"No option expiration dates found for {ticker}."
        if expiration_date not in options_dates:
            return None, None, None, f"Expiration date {expiration_date} not found. Available: {options_dates}"

        # Use 5d period for more robust price fetching
        hist = stock.history(period="5d", auto_adjust=True) # Use adjusted prices here too
        if hist.empty or 'Close' not in hist.columns or hist['Close'].dropna().empty:
            print(f"Error: Could not fetch recent closing price for {ticker}.")
            return None, None, None, "Could not fetch current price."
        current_price = hist['Close'].dropna().iloc[-1]

        option_chain = stock.option_chain(expiration_date)
        calls = option_chain.calls
        puts = option_chain.puts

        if calls.empty and puts.empty:
            print(f"Warning: Calls and puts empty for {ticker} {expiration_date}.")
            # Return empty frames but valid price
            return pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), \
                   pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), \
                   current_price, \
                   "Warning: No calls or puts listed for this expiration."

        # Assign type
        if not calls.empty: calls = calls.assign(type='call')
        if not puts.empty: puts = puts.assign(type='put')
        options = pd.concat([calls, puts], ignore_index=True)

        # Filter invalid prices before calculating mid_price
        # Check bid/ask > 0 and ask >= bid (standard check)
        options = options[
            options['bid'].notna() & options['ask'].notna() &
            (options['bid'] > 1e-6) & (options['ask'] > 1e-6) &
            (options['ask'] >= options['bid']) # Use standard >= check
        ].copy()

        if options.empty:
             print(f"Warning: No options with valid bid/ask after filtering for {ticker} {expiration_date}.")
             return pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), \
                    pd.DataFrame(columns=["strike", "type", "mid_price", "impliedVolatility"]), \
                    current_price, \
                    "Warning: No options with valid bid/ask prices found."

        options['mid_price'] = (options['bid'] + options['ask']) / 2
        options['strike'] = options['strike'].astype(float)

        # Handle impliedVolatility column
        if 'impliedVolatility' not in options.columns:
             print("Warning: 'impliedVolatility' column missing from yfinance data.")
             options['impliedVolatility'] = np.nan
        options['impliedVolatility'] = pd.to_numeric(options['impliedVolatility'], errors='coerce')

        # Keep essential columns, allow NaN IV to proceed for options that have prices
        options = options[["strike", "type", "mid_price", "impliedVolatility"]].dropna(subset=['strike', 'type', 'mid_price'])

        # Filter OTM
        calls_df = options[(options['type'] == 'call') & (options['strike'] > current_price)].copy()
        puts_df = options[(options['type'] == 'put') & (options['strike'] < current_price)].copy()

        # Sort
        if not puts_df.empty: puts_df.sort_values('strike', ascending=False, inplace=True)
        if not calls_df.empty: calls_df.sort_values('strike', ascending=True, inplace=True)

        print(f"Fetched {len(puts_df)} OTM puts and {len(calls_df)} OTM calls.")
        return puts_df.reset_index(drop=True), calls_df.reset_index(drop=True), current_price, None # No error

    except Exception as e:
        print(f"An error occurred in fetch_option_chain_data: {e}")
        traceback.print_exc()
        return None, None, None, f"An unexpected error occurred: {e}"


# --- Numba Optimized P&L Calculation ---
@numba.jit(nopython=True)
def calculate_pnl_vectorized(simulated_prices, short_put, short_call, credit, max_loss, max_profit):
    """Calculates P&L for an array of simulated prices using Numba."""
    n = len(simulated_prices)
    pnl = np.empty(n, dtype=np.float64) # Pre-allocate array

    for i in range(n):
        price = simulated_prices[i]
        if price >= short_put and price <= short_call:
            pnl[i] = credit # Max profit zone
        elif price < short_put:
            # Loss on put side: credit - (short_put - price)
            pnl[i] = credit - (short_put - price)
        else: # price > short_call
            # Loss on call side: credit - (price - short_call)
            pnl[i] = credit - (price - short_call)

    # Clip P&L to max loss and max profit
    pnl = np.clip(pnl, -max_loss, max_profit)
    return pnl


# --- Core Simulation Logic (Includes Greeks Calculation) ---
def process_condor_combo(
    combo,
    put_prices, call_prices,   # Prices
    put_ivs, call_ivs,         # IVs
    simulated_prices,          # Simulations
    num_simulations,           # Count
    current_price, T, r=0.0    # S, T, risk-free rate
    ):
    """
    Calculates metrics for a single Iron Condor combination, including Greeks.
    """
    long_put, short_put, short_call, long_call = combo
    try:
        # Retrieve prices and IVs safely using .get()
        lp_price = put_prices.get(long_put); sp_price = put_prices.get(short_put)
        sc_price = call_prices.get(short_call); lc_price = call_prices.get(long_call)
        lp_iv = put_ivs.get(long_put, np.nan); sp_iv = put_ivs.get(short_put, np.nan)
        sc_iv = call_ivs.get(short_call, np.nan); lc_iv = call_ivs.get(long_call, np.nan)

        # Check if any price is missing
        if any(p is None for p in [lp_price, sp_price, sc_price, lc_price]):
             # This check might be redundant if combos are generated from dict keys, but keep for safety
             # print(f"Debug: Price is None for a leg in combo {combo}")
             raise KeyError("Price is None for a leg")

    except KeyError as e:
        # print(f"Debug: Price Key Error for combo {combo}: {e}") # Optional debug
        return None

    # --- P&L / Base Metrics Calculations ---
    credit = (sp_price + sc_price) - (lp_price + lc_price)
    if credit <= 0.01: return None # Skip non-credit or negligible credit trades
    put_width = short_put - long_put; call_width = long_call - short_call
    if put_width <= 0 or call_width <= 0: return None # Ensure positive widths
    required_margin = max(put_width, call_width)
    max_loss = required_margin - credit
    if max_loss <= 0.01: return None # Skip no-risk/guaranteed profit trades
    max_profit = credit

    # --- P&L Simulation & Derived Metrics ---
    pnl = calculate_pnl_vectorized(simulated_prices, short_put, short_call, credit, max_loss, max_profit)
    avg_pnl = np.mean(pnl); wins = np.sum(pnl > 0.001); win_rate = (wins / num_simulations) * 100
    ev = avg_pnl; std_dev_pnl = np.std(pnl)
    sharpe_ratio = (avg_pnl / std_dev_pnl) if std_dev_pnl > 1e-9 else (np.inf if avg_pnl > 1e-9 else 0.0)

    # Kelly Calculation
    kelly_pct = 0.0; W = win_rate / 100.0
    if W > 0 and W < 1: # Need wins and losses
        wins_pnl = pnl[pnl > 0.001]; losses_pnl = pnl[pnl <= 0.001]
        # Check if there are actual wins/losses before calculating mean
        avg_win = np.mean(wins_pnl) if len(wins_pnl) > 0 else 0
        avg_loss = np.mean(np.abs(losses_pnl)) if len(losses_pnl) > 0 else 0
        if avg_loss > 1e-9: # Avoid division by zero
            R = avg_win / avg_loss # Win/Loss Ratio
            if R > 0: # Ensure positive ratio
                 kelly_pct = W - ((1.0 - W) / R)
            # If R=0 (avg_win=0), kelly remains 0 (or negative, implying don't bet)
        else:
             # Only wins or zero P&L outcomes
             kelly_pct = W # Bet fraction = win probability (theoretically)
    elif W >= 1.0: # 100% wins in simulation
        kelly_pct = 1.0

    # Other Metrics
    roi = (avg_pnl / max_loss) * 100 # Max loss is guaranteed > 0 here
    risk_reward_ratio = max_profit / max_loss # Max loss is guaranteed > 0 here

    # --- Calculate Greeks ---
    net_delta, net_gamma, net_theta, net_vega = np.nan, np.nan, np.nan, np.nan
    try:
        ivs = [lp_iv, sp_iv, sc_iv, lc_iv]
        strikes = [long_put, short_put, short_call, long_call]
        flags = ['p', 'p', 'c', 'c']
        signs = [-1, 1, 1, -1] # Long P, Short P, Short C, Long C

        # Check if all IVs are valid numbers and T > 0
        if not np.isnan(ivs).any() and T > 1e-6:
            greeks_sum = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
            for k, flag, iv, sign in zip(strikes, flags, ivs, signs):
                 # Use a small positive value if IV is near zero to avoid math errors in py_vollib
                 if iv <= 1e-6: iv = 1e-6
                 # Calculate Greeks using py_vollib
                 greeks_sum['delta'] += sign * greeks_analytical.delta(flag, current_price, k, T, r, iv)
                 greeks_sum['gamma'] += sign * greeks_analytical.gamma(flag, current_price, k, T, r, iv)
                 # Theta is often returned negative for long options (decay), so adding short theta (positive sign) increases net theta
                 # py_vollib returns theta per year, convert to per day
                 greeks_sum['theta'] += sign * greeks_analytical.theta(flag, current_price, k, T, r, iv) / 365.25
                 # Vega is per 1 vol point (100%), convert to per 1% change
                 greeks_sum['vega'] += sign * greeks_analytical.vega(flag, current_price, k, T, r, iv) / 100.0

            net_delta = greeks_sum['delta']
            net_gamma = greeks_sum['gamma']
            net_theta = greeks_sum['theta']
            net_vega = greeks_sum['vega']
        # Else: Greeks remain NaN if any IV is missing or T is invalid

    except Exception as greek_e:
        # print(f"Warning: Greek calc error for {combo}: {greek_e}") # Optional debug
        pass # Keep Greeks as NaN on error

    # --- Return Results Dictionary ---
    return {
        "Long Put": long_put, "Short Put": short_put, "Short Call": short_call, "Long Call": long_call,
        "Put Spread Width": round(put_width, 2), "Call Spread Width": round(call_width, 2),
        "Net Premium Received (Open)": round(credit, 2), "Margin Req.": round(required_margin, 2),
        "Max Loss": round(max_loss, 2), "Credit (Max Profit)": round(max_profit, 2),
        "Win Rate (%)": round(win_rate, 2), "Avg P&L": round(avg_pnl, 4), "EV": round(ev, 4),
        "Sharpe Ratio": round(sharpe_ratio, 4) if sharpe_ratio != np.inf else 'inf',
        "Kelly Pct": round(kelly_pct * 100, 2),
        "Risk-Reward Ratio": round(risk_reward_ratio, 2) if risk_reward_ratio != float('inf') else 'inf',
        "ROI (%)": round(roi, 2) if roi != float('inf') else 'inf',
        "Net Delta": round(net_delta, 4) if not np.isnan(net_delta) else np.nan,
        "Net Gamma": round(net_gamma, 4) if not np.isnan(net_gamma) else np.nan,
        "Net Theta": round(net_theta, 4) if not np.isnan(net_theta) else np.nan, # Per day
        "Net Vega": round(net_vega, 4) if not np.isnan(net_vega) else np.nan    # Per 1% IV change
    }


# --- Main Simulation Function (No Pre-filtering, Passes data for Greeks) ---
def run_full_simulation(
    ticker, expiration_date, hist_period, num_sims, strike_sp,
    simulation_method='Historical Bins'
    ):
    """
    Runs simulation WITHOUT pre-filtering. Calculates Greeks. Accepts UI parameters.
    """
    N_JOBS = -1
    T = np.nan # Initialize Time to expiration
    print(f"\n--- Starting Simulation (No Pre-filtering, With Greeks) ---")
    print(f"Ticker: {ticker}, Expiry: {expiration_date}, Method: {simulation_method}")
    print(f"Params: Hist Period={hist_period}, Num Sims={num_sims}, Strike Spacing={strike_sp}")
    simulated_prices = None

    # 1. Get Historical Data
    distribution, bin_categories, raw_weekly_returns = create_weekly_distribution(ticker=ticker, period=hist_period)
    hist_dist_available = (distribution is not None and bin_categories is not None)
    raw_returns_available = (raw_weekly_returns is not None and not raw_weekly_returns.empty)
    # Early exit checks
    if simulation_method == 'Historical Bins' and not hist_dist_available: return None, "Error: Failed distribution."
    if simulation_method == 'Weighted Historical' and not raw_returns_available: return None, "Error: Failed raw returns."
    if simulation_method != 'Implied Volatility' and not hist_dist_available and not raw_returns_available: return None, "Error: Failed hist data."
    elif simulation_method == 'Implied Volatility' and not hist_dist_available and not raw_returns_available: print("Warning: Hist data failed, using IV.")

    # 2. Fetch Option Chain & Current Price
    puts_df_orig, calls_df_orig, current_price, fetch_error = fetch_option_chain_data(ticker, expiration_date)
    if fetch_error: return None, f"Error fetching options: {fetch_error}"
    if current_price is None: return None, "Error: Failed current price."
    if puts_df_orig is None or calls_df_orig is None: return None, "Error: Option fetch returned None."
    print(f"Current Price fetched: {current_price:.2f}")

    # 3. Calculate Time T
    try:
        today = datetime.now().date(); print(f"DEBUG: Today = {today}")
        expiry_date_obj = datetime.strptime(expiration_date, '%Y-%m-%d').date(); print(f"DEBUG: Expiry = {expiry_date_obj}")
        days_to_expiry = (expiry_date_obj - today).days; print(f"DEBUG: Days = {days_to_expiry}")
        if days_to_expiry <= 0:
             if days_to_expiry == 0: T = 1 / (365.25 * 24 * 60); print("Warn: Today expiry, small T.")
             else: return None, "Error: Expiry in past."
        else: T = days_to_expiry / 365.25
        print(f"Time to Expiration (T) = {T:.4f} years")
        if T < 1/365.25 and days_to_expiry > 0 : print(f"WARN: T seems small ({T:.4f}).")
    except ValueError: return None, "Error: Cannot parse expiry date for T calc."
    except Exception as t_err: return None, f"Error calculating T: {t_err}"

    # 4. Apply initial strike spacing filter
    puts_df = puts_df_orig.copy(); calls_df = calls_df_orig.copy()
    if strike_sp > 0:
        puts_df = puts_df[puts_df['strike'] % strike_sp == 0].copy()
        calls_df = calls_df[calls_df['strike'] % strike_sp == 0].copy()
        print(f"Spacing {strike_sp}. Puts: {len(puts_df)}, Calls: {len(calls_df)}")
    if puts_df.empty or calls_df.empty: m = f"No strikes left after filter."; print(m); return None, m

    # === Generate Simulated Prices ===
    print(f"Generating {num_sims} price scenarios using {simulation_method}...")
    generation_error = None
    avg_iv = None
    try:
        # --- Ensure T is valid and calculated ---
        if T is None or np.isnan(T) or T <= 1e-9: # Need positive T for scaling
            raise ValueError("Time to expiration (T) is not valid for simulation.")

        # --- Calculate number of weeks ---
        weeks_to_expiry = T * 52.18 # Approx weeks per year

        if simulation_method == 'Historical Bins' or simulation_method == 'Weighted Historical':
            if not raw_returns_available: # Need raw returns for stats
                 raise ValueError("Raw historical returns required for scaling.")

            # Calculate historical weekly stats (percentage)
            mu_w = raw_weekly_returns.mean()
            sigma_w = raw_weekly_returns.std()

            if sigma_w < 1e-9: # Handle case of zero volatility
                print("Warning: Historical weekly standard deviation is near zero.")
                # Simulate no change or based only on drift
                scaled_total_pct_move = mu_w * weeks_to_expiry
                simulated_prices = current_price * (1 + scaled_total_pct_move / 100.0) * np.ones(num_sims) # Array of same value
            else:
                # Scale stats to the full period T
                mu_T_total = mu_w * weeks_to_expiry
                sigma_T_total = sigma_w * np.sqrt(weeks_to_expiry)

                # Simulate the TOTAL percentage change over T using scaled Normal distribution
                print(f"Simulating TOTAL move over T={T:.4f} years ({weeks_to_expiry:.2f} weeks) using Normal dist:")
                print(f"  Scaled Mean (Total %): {mu_T_total:.4f}, Scaled StdDev (Total %): {sigma_T_total:.4f}")
                simulated_total_pct_moves = np.random.normal(loc=mu_T_total, scale=sigma_T_total, size=num_sims)

                # Calculate final simulated prices
                simulated_prices = current_price * (1 + simulated_total_pct_moves / 100.0)
                # Ensure prices don't go below zero (optional but good practice)
                simulated_prices = np.maximum(simulated_prices, 1e-6)

        elif simulation_method == 'Implied Volatility':
            # --- This method ALREADY correctly scales with sqrt(T) ---
            all_options = pd.concat([puts_df, calls_df])
            # (Rest of IV calculation logic remains the same...)
            if 'impliedVolatility' in all_options.columns and not all_options['impliedVolatility'].isnull().all():
                 atm_options = all_options[(all_options['strike'] > current_price * 0.9) & (all_options['strike'] < current_price * 1.1)]['impliedVolatility'].dropna()
                 if not atm_options.empty: avg_iv = atm_options.mean()
                 else: avg_iv = all_options['impliedVolatility'].dropna().mean(); avg_iv = avg_iv if not np.isnan(avg_iv) else 0.30
            else: avg_iv = 0.30
            if avg_iv <= 1e-6: avg_iv = 0.30

            # GBM Calculation (already includes T)
            Z = np.random.standard_normal(num_sims)
            drift = (0.0 - 0.5 * avg_iv**2) * T # Assuming risk-free rate r=0 for drift here
            diffusion = avg_iv * np.sqrt(T) * Z
            simulated_prices = current_price * np.exp(drift + diffusion)
            # Ensure prices don't go below zero
            simulated_prices = np.maximum(simulated_prices, 1e-6)
            print(f"Prices generated via Log-Normal (T={T:.4f}, Avg IV={avg_iv:.4f}).")

        else:
            raise ValueError(f"Unknown sim method '{simulation_method}'.")

        if simulated_prices is None or len(simulated_prices) != num_sims:
            raise ValueError("Failed price gen array.")
        print("Price scenarios successfully generated.")

    except Exception as gen_e:
        print(f"Error during price generation: {gen_e}")
        traceback.print_exc()
        return None, f"Error generating simulated prices: {gen_e}"

    # === Condor Combination & Analysis ===
    print("Generating and analyzing condor combinations...")
    put_prices = puts_df.set_index('strike')['mid_price'].to_dict()
    call_prices = calls_df.set_index('strike')['mid_price'].to_dict()
    put_ivs = puts_df.set_index('strike')['impliedVolatility'].to_dict()
    call_ivs = calls_df.set_index('strike')['impliedVolatility'].to_dict()
    put_strikes = sorted(put_prices.keys(), reverse=True)
    call_strikes = sorted(call_prices.keys())

    print(f"Num put strikes: {len(put_strikes)}")
    print(f"Num call strikes: {len(call_strikes)}")
    if put_strikes:
        print(f"Put strikes: {put_strikes}")
    if call_strikes:
        print(f"Call strikes: {call_strikes}")

    if not put_strikes or not call_strikes or len(put_strikes) < 2 or len(call_strikes) < 2:
        m = "Not enough unique strikes."
        print(m)
        return None, m

    valid_combos = []
    combo_gen_start_time = time.time()

    for i in range(len(put_strikes)):
        sp = put_strikes[i]
        for j in range(i + 1, len(put_strikes)):
            lp = put_strikes[j]

            # Efficiently find first call index > sp
            first_call_idx = -1
            for k, sc_strike in enumerate(call_strikes):
                if sc_strike > sp:
                    first_call_idx = k
                    break
            if first_call_idx == -1:
                continue  # No calls > sp

            for k in range(first_call_idx, len(call_strikes)):
                sc = call_strikes[k]
                # Ensure long call exists
                if k + 1 < len(call_strikes):
                    for l in range(k + 1, len(call_strikes)):
                        lc = call_strikes[l]
                        valid_combos.append((lp, sp, sc, lc))

    combo_gen_end_time = time.time()
    print(f"Generated {len(valid_combos)} potential condor strategies in {combo_gen_end_time - combo_gen_start_time:.2f} seconds.")


    if not valid_combos:
        m = "No valid condor combinations found (LP < SP < SC < LC)."
        print(m)
        if put_strikes and call_strikes:
             max_put = put_strikes[0]; min_call = call_strikes[0]; print(f"DEBUG: Max Put={max_put}, Min Call={min_call}")
             if min_call <= max_put: print(">>> Root Cause Likely: Lowest call <= highest put!")
             else: print(">>> Strikes OK (min_call > max_put). Issue elsewhere?")
        else: print(">>> Cannot check strikes.")
        return None, m

    # === Parallel Processing ===
    print(f"Simulating P&L and Greeks for {len(valid_combos)} combos...")
    start_sim_time = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(process_condor_combo)(
            combo, put_prices, call_prices, put_ivs, call_ivs,
            simulated_prices, num_sims, current_price, T, r=0.0
            )
        for combo in valid_combos
    )
    end_sim_time = time.time()
    print(f"Simulation completed in {end_sim_time - start_sim_time:.2f} seconds.")

    results = [r for r in results if r is not None]
    if not results: m = "No valid sim results after processing."; print(m); return None, m

    # --- Create Final DataFrame ---
    results_df = pd.DataFrame(results)
    desired_columns = [
        "Long Put", "Short Put", "Short Call", "Long Call", "Put Spread Width", "Call Spread Width",
        "Net Premium Received (Open)", "Margin Req.", "Max Loss", "Credit (Max Profit)",
        "Win Rate (%)", "Avg P&L", "EV", "Sharpe Ratio", "Kelly Pct", "Risk-Reward Ratio", "ROI (%)",
        "Net Delta", "Net Gamma", "Net Theta", "Net Vega"
    ]
    existing_columns = [col for col in desired_columns if col in results_df.columns]; results_df = results_df[existing_columns]
    sort_column = "ROI (%)";
    if sort_column in results_df.columns: results_df = results_df.sort_values(by=sort_column, ascending=False)
    else: print(f"Warning: Sort column '{sort_column}' not found.")
    print(f"Analysis successful. Found {len(results_df)} valid strategies.")
    return results_df, None # Return DataFrame and no error message
