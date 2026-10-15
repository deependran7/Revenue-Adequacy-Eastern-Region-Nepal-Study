import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calc_exceedance_q(flows, exceedance_pct=45):
    """
    Compute the discharge value at a given exceedance probability (e.g. Q45)
    from a flow-duration-curve perspective.

    Parameters
    ----------
    flows : array-like
        Discharge values (NaNs are dropped internally).
    exceedance_pct : float
        Exceedance probability in percent (default 45 -> Q45).

    Returns
    -------
    float
        Discharge value corresponding to the requested exceedance probability.
    """
    flows = pd.Series(flows).dropna().sort_values(ascending=False).values
    n = len(flows)
    if n == 0:
        return np.nan

    exceedance = np.arange(1, n + 1) / (n + 1) * 100
    return np.interp(exceedance_pct, exceedance, flows)


def Discharge_P_MAX_PU():
    # ---------- Read & prepare data ----------
    df = pd.read_csv("Data/Final-discharge-data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Hour"] = df["Date"].dt.hour

    value_cols = df.columns.difference(["Date", "Month", "Day", "Hour"])

    # Average for each hour across all years
    avg_hourly = (
        df.groupby(["Month", "Day", "Hour"])[value_cols]
        .mean()
        .reset_index()
    )

    avg_hourly["Date"] = (
        pd.to_datetime(
            "2000-"
            + avg_hourly["Month"].astype(str).str.zfill(2)
            + "-"
            + avg_hourly["Day"].astype(str).str.zfill(2)
        )
        + pd.to_timedelta(avg_hourly["Hour"], unit="h")
    )

    cols = ["Date"] + [c for c in avg_hourly.columns if c not in ["Date", "Month", "Day", "Hour"]]
    avg_hourly = avg_hourly[cols].set_index("Date")
    df = avg_hourly.reset_index()

    date_col = "Date"
    plant_cols = [c for c in df.columns if c != date_col]

    # ---------- Q45 / intake discharge using the helper function ----------
    intake_df = pd.DataFrame({date_col: df[date_col]})
    q45_results = []

    for plant in plant_cols:
        q45 = calc_exceedance_q(df[plant], exceedance_pct=45)
        q45_results.append({"Plant": plant, "Q45 (m3/s)": q45})
        intake_df[plant] = np.minimum(df[plant], q45)

    q45_df = pd.DataFrame(q45_results)  # noqa: F841 (kept for potential export/debug)

    # ---------- Smoothing (7-day moving average on 15-min-ish data => 96 window) ----------
    smooth_df = intake_df.copy()
    for col in intake_df.columns:
        if col != date_col:
            smooth_df[col] = (
                intake_df[col].rolling(window=96, center=True, min_periods=1).mean()
            )

    # ---------- Normalize by max value per plant, clipped to <= 1 ----------
    max_norm_df = smooth_df.copy()
    for col in smooth_df.columns:
        if col != date_col:
            col_max = smooth_df[col].max()
            normalized = smooth_df[col] / col_max if col_max else smooth_df[col]
            max_norm_df[col] = normalized.clip(upper=1.0)  # enforce max <= 1

    max_norm_df = max_norm_df.set_index(date_col)

    # ---------- Expand to hourly resolution over one year ----------
    hourly_index = pd.date_range(start=max_norm_df.index.min(), periods=8760, freq="h")
    hourly_df = max_norm_df.reindex(hourly_index).interpolate(method="time")
    hourly_df.index = pd.date_range(start="2020-01-01", periods=8760, freq="h")

    # Re-clip after interpolation, since interpolation can overshoot slightly above 1
    hourly_df = hourly_df.clip(upper=0.99)
    max_norm_df = hourly_df

    # ---------- Aggregate stats across plants ----------
    data_cols = [c for c in max_norm_df.columns if c != date_col]
    mean_series = max_norm_df[data_cols].mean(axis=1)
    std_series = max_norm_df[data_cols].std(axis=1)

    upper = (mean_series + std_series).clip(upper=1.0)  # clamp upper band too
    lower = mean_series - std_series

    feature_df = pd.DataFrame({"mean": mean_series, "upper": upper, "lower": lower})
    feature_df.index.name = "Snapshot"

    # ---------- Plot ----------
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    avg_hourly.plot(legend=False, ax=ax[0], x_compat=True)
    hourly_df.plot(legend=False, ax=ax[1], x_compat=True)
    feature_df.plot(ax=ax[2], x_compat=True)
    ax[0].set_ylabel('Discharge ($m^3/sec$)')
    ax[1].set_ylabel('Normalized Discharge Exceedance')
    ax[2].set_ylabel('Normalized Statistics')

    for a in ax:
        a.xaxis.set_major_locator(mdates.MonthLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        a.set_xlabel("")
        a.grid(False)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return feature_df