"""
streamlit_app.py — BLS Jobs Report (PAYEMS) Revision Explorer
---------------------------------------------------------------
Features
 - Sidebar: fetch/update data from ALFRED (FRED vintages) with your API key
 - Filters: date range, recession shading, metric selection, histogram bins
 - Charts: histogram of revisions, boxplot (recession vs expansion), time series
 - Caching: saves fetched panel & revisions to speed up iteration

How to run locally:
   1) pip install streamlit requests pandas pyarrow altair python-dateutil
   2) set your key in one of two ways:
        - environment:   export FRED_API_KEY="your_key"   (Windows PowerShell: setx FRED_API_KEY "your_key")
        - Streamlit secrets: create .streamlit/secrets.toml with FRED_API_KEY = "your_key"
   3) streamlit run streamlit_app.py

Notes
 - Internet access is required on your machine to call FRED/ALFRED.
 - We focus on PAYEMS (All Employees: Total Nonfarm, monthly SA). You can add more series later.
"""

import os
import time
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import numpy as np
import requests
import streamlit as st

# -----------------------------
# Config & paths
# -----------------------------
st.set_page_config(page_title="BLS Revisions Explorer", layout="wide")
DATA_DIR = Path("data/revisions_app")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SERIES_DEFAULT = "PAYEMS"  # Total Nonfarm, SA (thousands)
USREC_SERIES = "USREC"     # NBER recession indicator (0/1)
BASE = "https://api.stlouisfed.org/fred"

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def get_api_key() -> str:
    """Return the FRED API key, preferring Streamlit secrets over environment vars."""
    key = st.secrets.get("FRED_API_KEY", "") if hasattr(st, "secrets") else ""
    return key or os.getenv("FRED_API_KEY", "")


def fred(endpoint: str, api_key: str, **params):
    """Helper to call the FRED/ALFRED API."""
    p = {"api_key": api_key, "file_type": "json"}
    p.update(params)
    url = f"{BASE}/{endpoint}"
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def get_vintage_dates(series_id: str, api_key: str) -> list[str]:
    """Fetch the list of vintage release dates for a given series."""
    data = fred("series/vintagedates", api_key=api_key, series_id=series_id)
    return data.get("vintage_dates", [])


@st.cache_data(show_spinner=False)
def fetch_obs_for_vintages(series_id: str, api_key: str, vintage_dates: list[str], batch: int = 80, pause: float = 0.15) -> pd.DataFrame:
    """Pull observations for many vintage dates, batching to avoid timeouts."""
    frames = []
    for i in range(0, len(vintage_dates), batch):
        vd = ",".join(vintage_dates[i:i+batch])
        data = fred("series/observations", api_key=api_key, series_id=series_id, vintage_dates=vd)
        obs = data.get("observations", [])
        if not obs:
            continue
        df = pd.DataFrame(obs)
        frames.append(df)
        time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["ref_month", "vintage", "value"])
    panel = pd.concat(frames, ignore_index=True)
    # types & tidy
    panel["ref_month"] = pd.to_datetime(panel["date"])                 # reference month
    panel["vintage"]   = pd.to_datetime(panel["realtime_start"])       # release snapshot date
    panel["value"]     = pd.to_numeric(panel["value"].replace(".", pd.NA))
    panel = panel[["ref_month", "vintage", "value"]].dropna().sort_values(["ref_month", "vintage"])
    return panel


def _first_second_third(g: pd.DataFrame) -> pd.Series:
    """Return the first, second and third estimates from grouped revisions."""
    g = g.drop_duplicates(subset=["vintage"]).sort_values("vintage")
    vals = g["value"].tolist()
    return pd.Series({
        "first":  vals[0] if len(vals) >= 1 else pd.NA,
        "second": vals[1] if len(vals) >= 2 else pd.NA,
        "third":  vals[2] if len(vals) >= 3 else pd.NA,
        "n_vintages": len(vals)
    })


@st.cache_data(show_spinner=False)
def build_revisions(panel: pd.DataFrame) -> pd.DataFrame:
    """Construct a DataFrame with first, second, third estimates and revision metrics."""
    if panel.empty:
        return pd.DataFrame()
    rev = (
        panel.groupby("ref_month")[ ["vintage", "value"] ]
             .apply(lambda g: _first_second_third(g))
             .sort_index()
    )
    rev["rev_2nd_minus_1st"] = rev["second"] - rev["first"]
    rev["rev_3rd_minus_1st"] = rev["third"]  - rev["first"]
    rev["rev_3rd_minus_2nd"] = rev["third"]  - rev["second"]
    rev["pct_rev_3rd_vs_1st"] = 100 * rev["rev_3rd_minus_1st"] / rev["first"]
    return rev.reset_index()


@st.cache_data(show_spinner=False)
def fetch_usrec(api_key: str) -> pd.DataFrame:
    """Fetch the NBER recession indicator (USREC) from FRED."""
    data = fred("series/observations", api_key=api_key, series_id=USREC_SERIES, observation_start="1939-01-01")
    obs = pd.DataFrame(data.get("observations", []))
    if obs.empty:
        return pd.DataFrame(columns=["ref_month", "USREC"])
    obs["ref_month"] = pd.to_datetime(obs["date"])
    obs["USREC"] = pd.to_numeric(obs["value"])
    return obs[["ref_month", "USREC"]]


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    """Persist a DataFrame to a named Parquet file in the data directory."""
    fp = DATA_DIR / f"{name}.parquet"
    df.to_parquet(fp, index=False)
    return fp


def load_parquet(name: str) -> pd.DataFrame:
    """Load a named Parquet file from the data directory if it exists."""
    fp = DATA_DIR / f"{name}.parquet"
    if fp.exists():
        return pd.read_parquet(fp)
    return pd.DataFrame()


# -----------------------------
# UI — Sidebar
# -----------------------------
st.title("BLS Jobs Report Revision Explorer (PAYEMS vintages)")
with st.sidebar:
    st.header("Data Controls")
    st.caption("Use your ALFRED/FRED key. Get one free at St. Louis Fed.")

    # Allow user override of key for this session
    stored_key = get_api_key()
    api_key = st.text_input("FRED API Key", value=stored_key, type="password")

    series_id = st.selectbox("Series", options=[SERIES_DEFAULT], index=0, help="Start with PAYEMS; add more series later.")

    colA, colB = st.columns(2)
    with colA:
        do_fetch = st.button("Fetch / Update Data", type="primary")
    with colB:
        clear_cache = st.button("Clear Cache")

    st.divider()
    st.subheader("Filters")
    show_recessions = st.toggle("Show NBER Recessions", value=True)
    metric = st.selectbox(
        "Revision Metric",
        ["rev_2nd_minus_1st", "rev_3rd_minus_1st", "rev_3rd_minus_2nd", "pct_rev_3rd_vs_1st"],
        index=1,
        help="Raw deltas are in thousands; percent normalizes across history."
    )

    # raw data info for debug feature
    show_debug = st.sidebar.toggle("Show debug info", value=False)


# -----------------------------
# Data fetch / cache actions
# -----------------------------
if clear_cache:
    st.cache_data.clear()
    st.toast("Cache cleared.")

    for fp in DATA_DIR.glob("*.parquet"):
        try:
            fp.unlink()
        except Exception as e:
            st.warning(f"Could not delete {fp.name}: {e}")

    for k in ("rev", "rev_filt"):
        if k in st.session_state:
            del st.session_state[k]

    st.toast("Cache cleared (memory + files).")
    st.rerun()

# Try to load cached artifacts first
panel = load_parquet("panel_payems")
rev = load_parquet("revisions_payems")
usrec = load_parquet("usrec")

if do_fetch:
    if not api_key:
        st.error("Please enter your FRED API key in the sidebar.")
    else:
        with st.spinner("Fetching vintage dates..."):
            vdates = get_vintage_dates(series_id, api_key)
        if not vdates:
            st.error("No vintage dates found. Check your API key and network.")
        else:
            st.success(f"Found {len(vdates)} vintages (releases). Fetching observations in batches...")
            with st.spinner("Downloading observations across vintages (this can take ~seconds)..."):
                panel = fetch_obs_for_vintages(series_id, api_key, vdates)
            if panel.empty:
                st.error("No observations returned. Try again or check API limits.")
            else:
                save_parquet(panel, "panel_payems")
                with st.spinner("Building first/second/third revisions table..."):
                    rev = build_revisions(panel)
                save_parquet(rev, "revisions_payems")
                with st.spinner("Fetching USREC (recession indicator)..."):
                    usrec = fetch_usrec(api_key)
                if not usrec.empty:
                    save_parquet(usrec, "usrec")
                st.success("Data updated!")

# Guard: if we still don't have data, show instructions
if rev.empty:
    st.info("""
**Getting started**

1) Enter your FRED API key in the sidebar.
2) Click **Fetch / Update Data**.
3) Then use the filters to explore revisions.

This app builds a panel of PAYEMS vintages (one snapshot per monthly release),
reconstructs first→second→third estimates for each reference month, and computes revision deltas.
    """)
    st.stop()

# Merge recession indicator if available
if not usrec.empty:
    rev = rev.merge(usrec, on="ref_month", how="left")
else:
    rev["USREC"] = 0

# Date range selector (dependent on data)
min_date = pd.to_datetime(rev["ref_month"].min()).date()
max_date = pd.to_datetime(rev["ref_month"].max()).date()
start_d, end_d = st.slider(
    "Date range",
    min_value=min_date,
    max_value=max_date,
    value=(max(min_date, date(1990,1,1)), max_date),
)
mask = (rev["ref_month"].dt.date >= start_d) & (rev["ref_month"].dt.date <= end_d)
rev_filt = rev.loc[mask].copy()

# -----------------------------
# Charts
# -----------------------------

if show_debug:
    st.subheader("Debug: data going into charts")

    st.write("Rows in full `rev`:", len(rev))
    st.write("Rows after filter (`rev_filt`):", len(rev_filt))

    # Metric being plotted
    st.write("Metric selected:", metric)

    # Check NA and basic stats
    st.write("Nulls in selected metric:", rev_filt[metric].isna().sum())
    st.write("`describe()` of selected metric:")
    st.write(rev_filt[metric].describe())

    # Peek at a few rows
    st.write("Head (filtered):")
    st.dataframe(rev_filt[["ref_month", "first", "second", "third", metric]].head(10))

    # Are values accidentally in fraction form? (e.g., 0.001 = 0.1%)
    # If the magnitudes look too small, uncomment the scale-up line.
    # if rev_filt[metric].abs().max() < 1:
    #     st.warning("Values look like fractions; scaling by 100 to percent for this debug plot.")
    #     debug_vals = rev_filt[metric] * 100
    # else:
    debug_vals = rev_filt[metric]

    # Quick alternative histogram using explicit bin step
    debug_hist = alt.Chart(pd.DataFrame({"val": debug_vals})).mark_bar().encode(
        x=alt.X("val:Q", bin=alt.Bin(step=0.01), title=f"{metric} (bin step=0.01)"),
        y=alt.Y("count()", title="Count"),
        tooltip=[alt.Tooltip("count()", title="Count")]
    ).properties(title="DEBUG histogram (explicit 0.01 bin step)", height=220)
    st.altair_chart(debug_hist, use_container_width=True)

    # Show unique value count to detect over-filtering
    st.write("Unique values in metric:", debug_vals.nunique())

    # Optional: show min/max dates of filtered data
    if not rev_filt.empty:
        st.write(
            "Filtered date range:",
            rev_filt["ref_month"].min().date(), "→", rev_filt["ref_month"].max().date()
        )

# Create columns for potential future use (not used in histogram but reserved)
left, right = st.columns([1, 1])

# Histogram

# Clean numeric series
vals = pd.to_numeric(rev_filt[metric], errors="coerce").dropna()
if vals.size < 2:
    st.info("Not enough data to plot a histogram for this selection.")
else:
    # Determine if the metric is a percentage
    is_percent = metric.startswith("pct_") or "%" in metric.lower()

    # Estimate bin width using the Freedman–Diaconis rule for robustness to outliers
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = q75 - q25
    # Fallback to standard deviation if IQR is zero or non-finite
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = vals.std(ddof=0)
    # Compute bin width
    h = 2 * iqr * (len(vals) ** (-1/3))
    # Fallback if bin width is invalid
    if not np.isfinite(h) or h <= 0:
        h = (vals.max() - vals.min()) / max(np.sqrt(len(vals)), 1.0)

    # Configure axis labels based on metric type
    title_text = f"{metric} (%)" if is_percent else metric
    fmt_text   = ".2f" if is_percent else ",.0f"
    x_axis = alt.Axis(title=title_text, format=fmt_text)

    # Determine a bin width using the Freedman–Diaconis rule and clamp to sensible minimums
    bin_width = h
    if is_percent:
        bin_width = float(np.round(bin_width, 4)) if np.isfinite(bin_width) else (vals.max() - vals.min()) / 50.0
        if bin_width <= 0:
            bin_width = (vals.max() - vals.min()) / 50.0
        bin_width = max(bin_width, 0.01)
    else:
        bin_width = float(np.round(bin_width)) if np.isfinite(bin_width) else (vals.max() - vals.min()) / 50.0
        if bin_width <= 0:
            bin_width = (vals.max() - vals.min()) / 50.0
        bin_width = max(bin_width, 1.0)

    # Determine the histogram extent, always including zero
    min_val = min(float(vals.min()), 0.0)
    max_val = max(float(vals.max()), 0.0)

    # Define the binning parameters for Altair; using extent ensures bins cover zero
    bin_params = alt.Bin(extent=[min_val, max_val], step=bin_width)

    # Build the histogram directly from the filtered revisions DataFrame
    bar_chart = (
        alt.Chart(rev_filt)
        .mark_bar(
            color="steelblue",
            opacity=0.8,
            strokeWidth=0  # remove outlines for true bar fill
        )
        .encode(
            x=alt.X(f"{metric}:Q", bin=bin_params, axis=x_axis, scale=alt.Scale(domain=[min_val, max_val])),
            y=alt.Y("count()", title="Count"),
            tooltip=[
                # Show the bin start and end along with count
                alt.Tooltip(f"bin_{metric}_start:Q", title="Bin start", format=fmt_text),
                alt.Tooltip(f"bin_{metric}_end:Q",   title="Bin end",   format=fmt_text),
                alt.Tooltip("count()", title="Count"),
            ],
        )
        .properties(
            height=400,
        )
    )

    # Create a vertical dashed line at x=0 for reference. Using a transform on the
    # same dataset ensures the rule is always present, even when the domain shifts.
    zero_line = (
        alt.Chart(rev_filt)
        .transform_calculate(x_zero='0')
        .mark_rule(
            color="firebrick",
            strokeDash=[4, 4],
            size=2
        )
        .encode(x='x_zero:Q')
    )

    # Overlay the zero line on top of the bar chart and set a title
    hist_chart = (
        bar_chart + zero_line
    ).properties(
        title=f"Histogram of {metric} ({start_d} to {end_d})"
    )
    st.altair_chart(hist_chart, use_container_width=True)


# Time series with optional recession shading
line = alt.Chart(rev_filt).mark_line().encode(
    x=alt.X('ref_month:T', title='Reference Month'),
    y=alt.Y(f"{metric}:Q", title=metric)
).properties(title=f"Time Series: {metric}", height=320)

if show_recessions and 'USREC' in rev_filt.columns:
    # Create recession bands
    bands = (
        rev_filt
        .assign(g=rev_filt['USREC'].ne(rev_filt['USREC'].shift()).cumsum())
        .query('USREC == 1')
        .groupby('g', as_index=False)
        .agg(start=('ref_month','min'), end=('ref_month','max'))
    )
    if not bands.empty:
        rect = alt.Chart(bands).mark_rect(opacity=0.15).encode(
            x='start:T', x2='end:T'
        )
        st.altair_chart(rect + line, use_container_width=True)
    else:
        st.altair_chart(line, use_container_width=True)
else:
    st.altair_chart(line, use_container_width=True)

st.divider()

# -----------------------------
# Data preview & download
# -----------------------------
st.subheader("Revisions table (filtered)")
st.dataframe(
    rev_filt[["ref_month","first","second","third","rev_2nd_minus_1st","rev_3rd_minus_1st","rev_3rd_minus_2nd","pct_rev_3rd_vs_1st","USREC"]]
    .sort_values("ref_month"),
    use_container_width=True,
    height=320,
)

csv = rev_filt.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", data=csv, file_name="revisions_filtered.csv", mime="text/csv")

st.caption("Tip: cache the raw panel & revisions to Parquet; re-run this app after each jobs report to refresh vintages.")