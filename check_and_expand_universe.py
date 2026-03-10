"""
check_and_expand_universe.py

Run this ONCE on the VM to:
1. Check how many stocks were actually downloaded
2. Try the full S&P 500 scrape
3. Cache the data so training reuses it automatically

Usage:
    python check_and_expand_universe.py
"""

import pandas as pd
from stock_return_project4 import download_price_history, load_sp500_constituents

START = "2010-01-01"
END   = "2024-01-01"

# ------------------------------------------------------------------
# Step 1: Check what the S&P 500 scrape gives us
# ------------------------------------------------------------------
print("Fetching S&P 500 constituents from Wikipedia...", flush=True)
sp500 = load_sp500_constituents()
print(f"  Constituents found: {len(sp500)}", flush=True)

if len(sp500) <= 55:
    print("  WARNING: Wikipedia scrape failed — using fallback 50-stock universe.")
    print("  Will try to download anyway; check your VM internet access.")
else:
    print(f"  Full S&P 500 available ({len(sp500)} tickers).")

# ------------------------------------------------------------------
# Step 2: Download and cache the full universe
# ------------------------------------------------------------------
print(f"\nDownloading price history for {len(sp500)} tickers ({START} to {END})...")
print("This may take a few minutes — data will be cached for future runs.\n", flush=True)

raw_prices = download_price_history(
    start=START,
    end=END,
    tickers=None,       # None = use full S&P 500
    use_cache=True,     # saves to data_cache/ automatically
)

n_tickers = raw_prices.columns.get_level_values(0).nunique()
n_dates   = len(raw_prices)

print(f"\n=== Universe Summary ===")
print(f"  Tickers successfully downloaded : {n_tickers}")
print(f"  Trading days                    : {n_dates}")
print(f"  Date range                      : {raw_prices.index[0].date()} → {raw_prices.index[-1].date()}")
print(f"  Approx. training samples        : ~{n_tickers * n_dates:,} (ticker × day)")

if n_tickers < 100:
    print("\n  NOTE: Universe is small. Check VM network or Wikipedia access.")
elif n_tickers >= 400:
    print("\n  Great! Full S&P 500 universe ready for training.")
else:
    print(f"\n  Partial universe ({n_tickers} stocks). Still much better than 50.")

print("\nCache saved to data_cache/ — training will reuse this automatically.")
