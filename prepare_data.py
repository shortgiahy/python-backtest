"""
prepare_data.py
===============
One-time conversion script: turns the Databento GLBX zip file into
MNQ_5min.csv in the exact format our backtest engine expects.

Run this ONCE before run.py:
    python prepare_data.py

If anything looks wrong, check the diagnostic output printed at each step.
"""

import sys
import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration — edit these if your paths differ
# ---------------------------------------------------------------------------

ZIP_PATH    = r"C:\Users\cheez\GLBX-20260218-5GBT9PUQ8Q.zip"
OUTPUT_CSV  = r"C:\Users\cheez\MNQ_5min.csv"
EXTRACT_DIR = Path(r"C:\Users\cheez\databento_raw")

# Databento uses "America/New_York" for Eastern time.
# All timestamps in the raw data are UTC; we convert to ET so that
# 09:30 in our strategy means the actual NYSE/CME market open.
EASTERN_TZ = "America/New_York"

# We filter rows to only keep MNQ bars.
# Databento continuous-contract symbols look like "MNQ.c.0".
# Individual contract symbols look like "MNQH6", "MNQM6", etc.
# Filtering for any symbol containing "MNQ" catches all of these.
MNQ_FILTER = "MNQ"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Step 1: Extract the zip ----
    print(f"\n[1/5] Extracting {Path(ZIP_PATH).name} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        contents = z.namelist()
        print(f"      Files inside zip: {contents}")
        z.extractall(EXTRACT_DIR)

    # ---- Step 2: Find what type of data file we have ----
    csv_files = sorted(EXTRACT_DIR.rglob("*.csv"))
    dbn_files = sorted(EXTRACT_DIR.rglob("*.dbn")) + sorted(EXTRACT_DIR.rglob("*.dbn.zst"))

    print(f"\n[2/5] Files found after extraction:")
    print(f"      CSV : {[f.name for f in csv_files]}")
    print(f"      DBN : {[f.name for f in dbn_files]}")

    # ---- Step 3: Read the data ----
    print(f"\n[3/5] Reading data ...")

    if csv_files:
        df = _read_csv(csv_files[0])
    elif dbn_files:
        df = _read_dbn(dbn_files[0])
    else:
        print("\nERROR: No CSV or DBN files found inside the zip.")
        print("       Try opening the zip manually and telling us what files are inside.")
        sys.exit(1)

    # ---- Step 4: Filter for MNQ and normalise columns ----
    print(f"\n[4/5] Normalising ...")
    df = _normalise(df)

    if df.empty:
        print("ERROR: No rows remain after filtering and normalisation.")
        sys.exit(1)

    # ---- Step 5: Save ----
    print(f"\n[5/5] Saving to {OUTPUT_CSV} ...")
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"  Done. {len(df):,} bars written to MNQ_5min.csv")
    print(f"  Date range : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Columns    : {df.columns.tolist()}")
    print(f"\n  First 3 rows:")
    print(df.head(3).to_string(index=False))
    print(f"{'='*55}\n")


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    """Read a Databento CSV export."""
    print(f"      Reading CSV: {path.name}")
    df = pd.read_csv(path, low_memory=False)
    print(f"      Shape   : {df.shape}")
    print(f"      Columns : {df.columns.tolist()}")
    print(f"      Sample  :\n{df.head(2).to_string()}\n")
    return df


def _read_dbn(path: Path) -> pd.DataFrame:
    """
    Read a Databento DBN (binary) file using the official databento library.

    If the library is not installed, this function prints installation
    instructions and exits rather than crashing with an ImportError.
    """
    try:
        import databento as db
    except ImportError:
        print("\nERROR: The 'databento' library is required to read .dbn files.")
        print("       Install it by running:")
        print("           python -m pip install databento")
        print("       Then run this script again.")
        sys.exit(1)

    print(f"      Reading DBN: {path.name}")
    store = db.DBNStore.from_file(str(path))
    df    = store.to_df()

    print(f"      Shape   : {df.shape}")
    print(f"      Columns : {df.columns.tolist()}")
    print(f"      Dtypes  :\n{df.dtypes.to_string()}")
    print(f"      Sample  :\n{df.head(2).to_string()}\n")
    return df


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert whatever Databento gives us into:
        timestamp, open, high, low, close, volume

    Handles the two most common Databento output formats:
      A) CSV with human-readable timestamps and float prices
      B) DBN-derived DataFrame with UTC DatetimeIndex and possibly
         integer prices (scaled by 1e9)
    """
    df = df.copy()

    # -- A. Get the timestamp into a column called 'timestamp' --
    #
    # Case 1: DatetimeIndex (common when reading DBN via .to_df())
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        # The reset_index call names the old index column after the index name.
        # Common names: 'ts_event', 'index', 'timestamp'.
        old_name = df.columns[0]
        df = df.rename(columns={old_name: "timestamp"})

    # Case 2: ts_event column (Databento's standard timestamp column name)
    elif "ts_event" in df.columns:
        df = df.rename(columns={"ts_event": "timestamp"})

    # Case 3: separate date + time columns (some export formats)
    elif "date" in df.columns and "time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str)
        )

    # -- B. Convert timestamp to Eastern time --
    #
    # Databento stores all timestamps in UTC.
    # Our strategy uses session_open="09:30" which is 09:30 Eastern (ET).
    # If we don't convert, the engine would look for bars at 09:30 UTC =
    # 04:30 ET (pre-market), find nothing, and generate zero signals.
    #
    # tz_convert() handles daylight saving automatically.
    # tz_localize(None) strips the timezone label from the result so
    # pandas treats it as a plain local timestamp (what our loader expects).
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = ts.dt.tz_convert(EASTERN_TZ).dt.tz_localize(None)

    # -- C. Filter for MNQ --
    #
    # A full Globex dataset contains thousands of instruments.
    # We keep only rows whose symbol contains "MNQ".
    if "symbol" in df.columns:
        symbols_present = df["symbol"].dropna().unique()
        mnq_symbols     = [s for s in symbols_present if MNQ_FILTER in str(s)]
        print(f"      Symbols in file (showing MNQ matches): {mnq_symbols}")

        if not mnq_symbols:
            print(f"\n      WARNING: No symbols containing '{MNQ_FILTER}' found.")
            print(f"      All symbols (first 30): {list(symbols_present[:30])}")
            print("      Check that you downloaded MNQ data from Databento.")
            sys.exit(1)

        df = df[df["symbol"].isin(mnq_symbols)].copy()
        print(f"      Rows after MNQ filter: {len(df):,}")

    # -- D. Fix prices if still in Databento integer format --
    #
    # Databento's DBN format stores prices as int64 scaled by 1e9.
    # Example: a price of 15000.25 is stored as 15000250000000.
    # When read via .to_df(), the library usually converts these to floats,
    # but occasionally the raw integers slip through.
    #
    # Heuristic: if the median open price is above 1,000,000, prices are
    # almost certainly still in raw integer form.
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        median_val = df[col].median()
        if median_val > 1_000_000:
            print(f"      '{col}' median={median_val:.0f} → dividing by 1e9")
            df[col] = df[col] / 1e9

    # -- E. Volume --
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    else:
        # Some Databento schemas name it differently.
        for alt in ["size", "qty", "count"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "volume"})
                break
        if "volume" not in df.columns:
            print("      WARNING: no volume column found — filling with 0.")
            df["volume"] = 0

    # -- F. Keep only the columns our loader needs, in order --
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Could not find these required columns: {missing}")
        print(f"       Available columns: {df.columns.tolist()}")
        sys.exit(1)

    df = df[required].dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
