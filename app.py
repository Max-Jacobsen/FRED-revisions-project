"""
streamlit_app.py — BLS Jobs Report (PAYEMS) Revision Explorer
"""

import os
import time
from datetime import date
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Config & paths
# -----------------------------
st.set_page_config(page_title="BLS Revisions Explorer", layout="wide")
DATA_DIR = Path("data/revisions_app")
DATA_DIR.mkdir(parents=True, exist_ok=True)

SERIES_DEFAULT = "PAYEMS"
USREC_SERIES = "USREC"     # Recession indicator (0/1)
DFF_SERIES = "DFF"          # Federal Funds Effective Rate (mean)
EFFR_SERIES = "EFFR"        # Effective Federal Funds Rate (median)
BASE = "https://api.stlouisfed.org/fred"


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def get_api_key() -> str:
    key = ""
    # Streamlit attaches a ``secrets`` attribute at runtime; protect
    # against environments where it may not be defined.
    if hasattr(st, "secrets"):
        key = st.secrets.get("FRED_API_KEY", "")
    return key or os.getenv("FRED_API_KEY", "")


def fred(endpoint: str, api_key: str, **params) -> dict:
    p = {"api_key": api_key, "file_type": "json"}
    p.update(params)
    url = f"{BASE}/{endpoint}"
    r = requests.get(url, params=p, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(show_spinner=False)
def get_vintage_dates(series_id: str, api_key: str) -> list[str]:
    data = fred("series/vintagedates", api_key=api_key, series_id=series_id)
    return data.get("vintage_dates", [])


@st.cache_data(show_spinner=False)
def fetch_obs_for_vintages(
    series_id: str,
    api_key: str,
    vintage_dates: list[str],
    batch: int = 80,
    pause: float = 0.15,
) -> pd.DataFrame:
    frames = []
    for i in range(0, len(vintage_dates), batch):
        vd = ",".join(vintage_dates[i : i + batch])
        data = fred(
            "series/observations", api_key=api_key, series_id=series_id, vintage_dates=vd
        )
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
    panel["ref_month"] = pd.to_datetime(panel["date"])  # reference month
    panel["vintage"] = pd.to_datetime(panel["realtime_start"])  # release snapshot date
    # Convert values to numeric; replace missing with NA.  Multiply by 1 000 to
    # convert from “thousands of employees” to actual counts.
    panel["value"] = pd.to_numeric(panel["value"].replace(".", pd.NA)) * 1000.0
    panel = (
        panel[["ref_month", "vintage", "value"]]
        .dropna()
        .sort_values(["ref_month", "vintage"])
    )
    return panel


def _first_second_third(g: pd.DataFrame) -> pd.Series:
    g = g.drop_duplicates(subset=["vintage"]).sort_values("vintage")
    vals = g["value"].tolist()
    return pd.Series(
        {
            "first_estimate": vals[0] if len(vals) >= 1 else pd.NA,
            "second_estimate": vals[1] if len(vals) >= 2 else pd.NA,
            "third_estimate": vals[2] if len(vals) >= 3 else pd.NA,
            "n_vintages": len(vals),
        }
    )


@st.cache_data(show_spinner=False)
def build_revisions(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    rev = (
        panel.groupby("ref_month")[["vintage", "value"]]
        .apply(lambda g: _first_second_third(g))
        .sort_index()
    )
    # Compute revision deltas on the actual counts
    rev["revision_second_minus_first"] = (
        rev["second_estimate"] - rev["first_estimate"]
    )
    rev["revision_third_minus_first"] = (
        rev["third_estimate"] - rev["first_estimate"]
    )
    rev["revision_third_minus_second"] = (
        rev["third_estimate"] - rev["second_estimate"]
    )
    # Percent revision expressed as a percentage
    rev["percent_revision_third_vs_first"] = (
        100 * rev["revision_third_minus_first"] / rev["first_estimate"]
    )
    return rev.reset_index()


@st.cache_data(show_spinner=False)
def fetch_usrec(api_key: str) -> pd.DataFrame:
    data = fred(
        "series/observations",
        api_key=api_key,
        series_id=USREC_SERIES,
        observation_start="1939-01-01",
    )
    obs = pd.DataFrame(data.get("observations", []))
    if obs.empty:
        return pd.DataFrame(columns=["ref_month", "USREC"])
    obs["ref_month"] = pd.to_datetime(obs["date"])
    obs["USREC"] = pd.to_numeric(obs["value"])
    return obs[["ref_month", "USREC"]]


@st.cache_data(show_spinner=False)
def fetch_interest_rate(api_key: str, series_id: str) -> pd.DataFrame:
    try:
        data = fred(
            "series/observations",
            api_key=api_key,
            series_id=series_id,
            observation_start="1928-01-01",
        )
    except Exception:
        return pd.DataFrame(columns=["ref_month", "rate"])
    obs = pd.DataFrame(data.get("observations", []))
    if obs.empty:
        return pd.DataFrame(columns=["ref_month", "rate"])
    # Convert the date to datetime and values to float
    obs["date"] = pd.to_datetime(obs["date"])
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    # Compute monthly average by grouping by year-month
    obs["ref_month"] = obs["date"].dt.to_period("M").dt.to_timestamp()
    monthly = obs.groupby("ref_month", as_index=False)["value"].mean()
    monthly = monthly.rename(columns={"value": "rate"})
    return monthly[["ref_month", "rate"]]


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

try:
    dff = load_parquet("dff_monthly")  # type: ignore[assignment]
except Exception:
    dff = pd.DataFrame()
try:
    effr = load_parquet("effr_monthly")  # type: ignore[assignment]
except Exception:
    effr = pd.DataFrame()


# -----------------------------
# UI — Sidebar
# -----------------------------
st.title("BLS Jobs Report Revision Explorer")
with st.sidebar:
   
    stored_key = get_api_key()

    if not stored_key:
        st.error(
            "No FRED API key found. Please add `FRED_API_KEY` to your "
            ".streamlit/secrets.toml` or set the environment variable."
        )

    # Series selector 
    series_id = st.selectbox("Series", options=[SERIES_DEFAULT], index=0)

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
        [
            "revision_second_minus_first",
            "revision_third_minus_first",
            "revision_third_minus_second",
            "percent_revision_third_vs_first",
        ],
        index=3,
        help=(
            "Raw deltas are shown in number of employees (not thousands). "
            "Percent normalises differences across history."
        ),
    )

    # Economic period filter
    econ_period = st.selectbox(
        "Economic Period",
        ["All", "Recession", "Non‑recession"],
        index=0,
        help=(
            "Filter the dataset to only those months where the NBER recession "
            "indicator is 1 (Recession) or 0 (Non‑recession). Selecting 'All' "
            "includes every month regardless of economic state. "
            "The 'Non‑recession' category encompasses periods when the economy "
            "is not in an official recession, which includes expansions and recoveries."
        ),
    )

    # raw data info for debug feature
    show_debug = st.toggle("Show debug info", value=False)

    # Interest‑rate controls    
    rate_option = st.selectbox(
        "Interest Rate Series",
        options=[
            "None",
            f"{DFF_SERIES} (mean, 1954+)",
            f"{EFFR_SERIES} (median, 2000+)",
        ],
        index=0,
        help=(
            "Choose the interest‑rate series to merge with the revisions data.  "
            "DFF is the mean‑based Federal Funds Effective Rate available from July 1954. "
            "EFFR is the median‑based Effective Federal Funds Rate available from July 2000."
        ),
    )

    if rate_option.startswith(EFFR_SERIES):
        st.caption(
            "**Note:** The EFFR series begins on July 3, 2000.  Earlier months in "
            "the revisions data will have no interest‑rate values when this series is selected. Therefore, they will not be included in the visualization when filtered on."
        )
    
    if rate_option.startswith(DFF_SERIES):
        st.caption(
            "**Note:** The DFF series begins on July 1, 1954.  Earlier months in "
            "the revisions data will have no interest‑rate values when this series is selected. Therefore, they will not be included in the visualization when filtered on."
        )
    
    rate_min_val: float
    rate_max_val: float
    rate_df_for_slider: pd.DataFrame | None = None
    if rate_option.startswith(DFF_SERIES):
        rate_df_for_slider = dff if isinstance(dff, pd.DataFrame) else None
    elif rate_option.startswith(EFFR_SERIES):
        rate_df_for_slider = effr if isinstance(effr, pd.DataFrame) else None

    if rate_df_for_slider is not None and not rate_df_for_slider.empty:
        rate_min_val = float(rate_df_for_slider["rate"].min())
        rate_max_val = float(rate_df_for_slider["rate"].max())
    else:
        rate_min_val, rate_max_val = 0.0, 10.0
    rate_default_min = round(rate_min_val, 2)
    rate_default_max = round(rate_max_val, 2)

    # Display the slider only when a rate series has been selected
    if rate_option != "None":
        rate_range = st.slider(
            "Interest Rate Range (%)",
            min_value=rate_default_min,
            max_value=rate_default_max,
            value=(rate_default_min, rate_default_max),
            step=0.01,
            help=(
                "Select the range of the chosen federal funds rate (monthly average) to "
                "include in the analysis.  Only months where the monthly average lies "
                "within this range will be retained."
            ),
        )
    else:

        rate_range = (rate_default_min, rate_default_max)


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

# Load cached interest‑rate series if available
dff = load_parquet("dff_monthly")
effr = load_parquet("effr_monthly")

if do_fetch:
    if not stored_key:
        st.error(
            "No API key available. Please configure your FRED API key in the "
            "secrets file."
        )
    else:
        with st.spinner("Fetching vintage dates..."):
            vdates = get_vintage_dates(series_id, stored_key)
        if not vdates:
            st.error("No vintage dates found. Check your API key and network.")
        else:
            st.success(
                f"Found {len(vdates)} vintages (releases). Fetching observations in batches..."
            )
            with st.spinner(
                "Downloading observations across vintages (this can take ~seconds)..."
            ):
                panel = fetch_obs_for_vintages(series_id, stored_key, vdates)
            if panel.empty:
                st.error("No observations returned. Try again or check API limits.")
            else:
                save_parquet(panel, "panel_payems")
                with st.spinner("Building first/second/third revisions table..."):
                    rev = build_revisions(panel)
                save_parquet(rev, "revisions_payems")
                with st.spinner("Fetching USREC (recession indicator)..."):
                    usrec = fetch_usrec(stored_key)
                if not usrec.empty:
                    save_parquet(usrec, "usrec")

                # Fetch interest‑rate series for both DFF and EFFR
                with st.spinner(
                    "Fetching federal funds rates (DFF and EFFR) and computing monthly averages..."
                ):
                    dff = fetch_interest_rate(stored_key, DFF_SERIES)
                    effr = fetch_interest_rate(stored_key, EFFR_SERIES)
                if not dff.empty:
                    save_parquet(dff, "dff_monthly")
                if not effr.empty:
                    save_parquet(effr, "effr_monthly")

                st.success("Data updated!")

# If no data
if rev.empty:
    st.info(
        """
        **Getting started**

        1) Click **Fetch / Update Data**. (Sourced from FRED)
        3) Then use the filters to explore revisions.

        This app builds a panel of PAYEMS vintages,
        reconstructing first → second → third estimates for each reference month, and computes
        revision on actual employment counts.
        """
    )
    st.stop()

# Merge recession indicator
if not usrec.empty:
    rev = rev.merge(usrec, on="ref_month", how="left")
else:
    rev["USREC"] = 0


# Merge the selected interest‑rate series
rate_df: pd.DataFrame | None = None
if isinstance(rate_option, str):
    if rate_option.startswith(DFF_SERIES):
        rate_df = dff if isinstance(dff, pd.DataFrame) else None
    elif rate_option.startswith(EFFR_SERIES):
        rate_df = effr if isinstance(effr, pd.DataFrame) else None


if rate_df is not None and not rate_df.empty:
    rate_df = rate_df.rename(columns={"rate": "interest_rate"})
    rev = rev.merge(rate_df, on="ref_month", how="left")
else:
    rev["interest_rate"] = pd.NA

# Date range selector 
min_date = pd.to_datetime(rev["ref_month"].min()).date()
max_date = pd.to_datetime(rev["ref_month"].max()).date()

month_range = pd.date_range(start=min_date, end=max_date, freq="MS").date.tolist()
default_start = max(min_date, date(1990, 1, 1)).replace(day=1)
default_end = month_range[-1].replace(day=1)

start_d, end_d = st.sidebar.select_slider(
    "Date range (monthly)",
    options=month_range,
    value=(default_start, default_end),
    format_func=lambda d: d.strftime("%Y-%m"),
)

mask = (rev["ref_month"].dt.date >= start_d) & (rev["ref_month"].dt.date <= end_d)
rev_filt = rev.loc[mask].copy()

# Apply economic period filter
if econ_period != "All" and "USREC" in rev_filt.columns:
    if econ_period == "Recession":
        rev_filt = rev_filt[rev_filt["USREC"] == 1].copy()
    elif econ_period == "Non‑recession":
        rev_filt = rev_filt[rev_filt["USREC"] == 0].copy()

# Apply interest‑rate filter 
time_series_enabled = True
try:
    rate_min_selected, rate_max_selected = rate_range
    # tol for rounding
    tol = 1e-8
    if rate_option != "None":
        if (rate_min_selected - rate_default_min) > tol or (rate_default_max - rate_max_selected) > tol:

            rev_filt = rev_filt[
                rev_filt["interest_rate"].notna()
                & (rev_filt["interest_rate"] >= rate_min_selected)
                & (rev_filt["interest_rate"] <= rate_max_selected)
            ].copy()
            time_series_enabled = False
except NameError:
    pass

    # Disable time series if not 'All'
if econ_period != "All":
    time_series_enabled = False


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
    debug_columns = [
        "ref_month",
        "first_estimate",
        "second_estimate",
        "third_estimate",
        metric,
    ]
    # Include the interest rate column in debug view if it exists
    if "interest_rate" in rev_filt.columns:
        debug_columns.append("interest_rate")
    st.dataframe(rev_filt[debug_columns].head(10))

    # Inspect histogram binning; adjust scale if values are fractions
    debug_vals = rev_filt[metric]

    # Quick alternative histogram using explicit bin step
    debug_hist = alt.Chart(pd.DataFrame({"val": debug_vals})).mark_bar().encode(
        x=alt.X("val:Q", bin=alt.Bin(step=0.01), title=f"{metric} (bin step=0.01)"),
        y=alt.Y("count()", title="Count"),
        tooltip=[alt.Tooltip("count()", title="Count")],
    ).properties(title="DEBUG histogram (explicit 0.01 bin step)", height=220)
    st.altair_chart(debug_hist, use_container_width=True)

    # Show unique value count to detect over-filtering
    st.write("Unique values in metric:", debug_vals.nunique())

    # Optional: show min/max dates of filtered data
    if not rev_filt.empty:
        st.write(
            "Filtered date range:",
            rev_filt["ref_month"].min().date(),
            "→",
            rev_filt["ref_month"].max().date(),
        )

# Histogram

vals = pd.to_numeric(rev_filt[metric], errors="coerce").dropna()
if vals.size < 2:
    st.info("Not enough data to plot a histogram for this selection.")
else:
    # Determine if the metric is a percentage
    is_percent = metric.startswith("percent_") or "%" in metric.lower()

    # Estimate bin width using the Freedman–Diaconis rule for robustness to outliers
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = q75 - q25
    # Fallback to standard deviation if IQR is zero or non-finite
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = vals.std(ddof=0)
    # Compute bin width
    h = 2 * iqr * (len(vals) ** (-1 / 3))
    # Fallback if bin width is invalid
    if not np.isfinite(h) or h <= 0:
        h = (vals.max() - vals.min()) / max(np.sqrt(len(vals)), 1.0)

    # Configure axis labels based on metric type
    title_text = f"{metric} (%)" if is_percent else metric
    fmt_text = ".2f" if is_percent else ",.0f"
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

    # Define the binning parameters
    bin_params = alt.Bin(extent=[min_val, max_val], step=bin_width)

    # Build the histogram directly from the filtered revisions DataFrame. 
    bar_chart = (
        alt.Chart(rev_filt)
        .mark_bar(
            color="steelblue",
            opacity=0.8,
            strokeWidth=0,
        )
        .encode(
            x=alt.X(
                f"{metric}:Q",
                bin=bin_params,
                axis=x_axis,
                scale=alt.Scale(domain=[min_val, max_val]),
            ),
            y=alt.Y("count()", title="Count"),
            tooltip=[
                # Number of observations in this bin
                alt.Tooltip("count()", title="Count"),
                # Mean value of the revision metric within this bin
                alt.Tooltip(
                    f"mean({metric}):Q",
                    title="Mean value in bin",
                    format=fmt_text,
                ),
                # Minimum value within the bin
                alt.Tooltip(
                    f"min({metric}):Q",
                    title="Min value in bin",
                    format=fmt_text,
                ),
                # Maximum value within the bin
                alt.Tooltip(
                    f"max({metric}):Q",
                    title="Max value in bin",
                    format=fmt_text,
                ),
            ],
        )
        .properties(
            height=400,
        )
    )

    # Create a vertical dashed line at x=0 for reference
    zero_line = (
        alt.Chart(rev_filt)
        .transform_calculate(x_zero="0")
        .mark_rule(color="firebrick", strokeDash=[4, 4], size=2)
        .encode(x="x_zero:Q")
    )

    # Overlay the zero line on top of the bar chart and set a title
    hist_chart = (bar_chart + zero_line).properties(
        title=f"Histogram of {metric} ({start_d} to {end_d})"
    )
    st.altair_chart(hist_chart, use_container_width=True)


# Time series with optional recession shading
if time_series_enabled:
    time_series_tooltip = [
        alt.Tooltip("ref_month:T", title="Reference Month"),
        alt.Tooltip(
            f"{metric}:Q",
            title=metric,
            format=".2f" if metric.startswith("percent_") or "%" in metric.lower() else ",.0f",
        ),
        alt.Tooltip("first_estimate:Q", title="First estimate", format=",.0f"),
        alt.Tooltip("second_estimate:Q", title="Second estimate", format=",.0f"),
        alt.Tooltip("third_estimate:Q", title="Third estimate", format=",.0f"),
    ]

    if "interest_rate" in rev_filt.columns:
        time_series_tooltip.append(
            alt.Tooltip(
                "interest_rate:Q",
                title="Interest rate (%)",
                format=".2f",
            )
        )
    line = (
        alt.Chart(rev_filt)
        .mark_line()
        .encode(
            x=alt.X("ref_month:T", title="Reference Month"),
            y=alt.Y(f"{metric}:Q", title=metric),
            tooltip=time_series_tooltip,
        )
        .properties(title=f"Time Series: {metric}", height=320)
        .interactive()
    )
    if show_recessions and "USREC" in rev_filt.columns:
        # Create recession bands
        bands = (
            rev_filt.assign(g=rev_filt["USREC"].ne(rev_filt["USREC"].shift()).cumsum())
            .query("USREC == 1")
            .groupby("g", as_index=False)
            .agg(start=("ref_month", "min"), end=("ref_month", "max"))
        )
        if not bands.empty:
            rect = alt.Chart(bands).mark_rect(opacity=0.15).encode(x="start:T", x2="end:T")
            st.altair_chart(rect + line, use_container_width=True)
        else:
            st.altair_chart(line, use_container_width=True)
    else:
        st.altair_chart(line, use_container_width=True)
else:
    st.info(
        "Time series chart is hidden because a sub‑selection filter is applied. "
        "Remove or reset the filters to view the full time series."
    )

st.divider()

# -----------------------------
# Data preview & download
# -----------------------------
st.subheader("Revisions table (filtered)")

with st.expander("Show data table", expanded=False):
    display_cols = [
        "ref_month",
        "first_estimate",
        "second_estimate",
        "third_estimate",
        "revision_second_minus_first",
        "revision_third_minus_first",
        "revision_third_minus_second",
        "percent_revision_third_vs_first",
        "USREC",
    ]
    # Append the interest‑rate column
    if "interest_rate" in rev_filt.columns:
        display_cols.append("interest_rate")
    st.dataframe(
        rev_filt[display_cols].sort_values("ref_month"),
        use_container_width=True,
        height=320,
    )
    # download CSV
    csv = rev_filt.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="revisions_filtered.csv",
        mime="text/csv",
    )

st.caption("Tip: re-run this app after each jobs report to refresh vintages.")