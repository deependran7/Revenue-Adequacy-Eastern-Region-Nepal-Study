import pandas as pd
from pathlib import Path
file_list = ['Data/IEX/IEX_data_2023.xlsx', 'Data/IEX/IEX_data_2024.xlsx', 'Data/IEX/IEX_data_2025.xlsx']
def get_IEX_data():
    all_data = []

    for file in file_list:
        print(f"Processing: {file}")
        xl = pd.ExcelFile(file)
        
        for sheet in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet)
            
            # Standardize column names
            df.columns = df.columns.str.strip()
            
            # Create the 15-min timestamp
            # We use errors='coerce' to handle any empty rows or headers
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
            
            # Split '00:15-00:30' to get '00:15'
            df['Start_Time'] = df['Time Block'].str.split('-').str[0]
            
            # Combine Date and Start Time
            df['Timestamp'] = pd.to_datetime(
                df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Start_Time'], 
                errors='coerce'
            )
            
            # Drop rows where timestamp couldn't be created (e.g. empty rows)
            df = df.dropna(subset=['Timestamp'])
            
            all_data.append(df)

    # Combine everything
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Final sort to keep time-series integrity
    master_df = master_df.sort_values(by='Timestamp').reset_index(drop=True)
    return master_df



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def IEX_data(df_MCP,annual_df):
    df_MCP = df_MCP.reset_index()
    df = df_MCP
    df['Time_Str'] = df['Timestamp'].dt.strftime('%H:%M')
    stats = df.groupby('Time_Str')['MCP (Rs/MWh) *'].agg(['mean', 'std', 'max', 'min']).reset_index()
    MCP_India_mean,MCP_India_high, MCP_India_low = stats['mean'], stats['mean']+stats['std'],stats['mean']-stats['std']



    # 1. Prepare the data
    df['Date'] = df['Timestamp'].dt.date
    df['Time_Str'] = df['Timestamp'].dt.strftime('%H:%M')

    # Add week number and day of week for weekly grouping
    df['Week'] = df['Timestamp'].dt.isocalendar().week.astype(int)
    df['Year'] = df['Timestamp'].dt.year
    df['YearWeek'] = df['Timestamp'].dt.strftime('%Y-W%U')  # e.g., '2022-W01'
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
    df['DayName'] = df['Timestamp'].dt.strftime('%a')  # Mon, Tue...

    # Create a 'WeekSlot': combines DayOfWeek + Time to give 7*96 = 672 blocks per week
    df['WeekSlot'] = df['DayName'] + ' ' + df['Time_Str']  # e.g., 'Mon 00:00'

    # Preserve ordering by creating a sort key
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df['DayOrder'] = df['DayOfWeek']  # 0–6, already ordered

    # 2. Define Price Thresholds
    thresholds = np.arange(2000, 8500, 250)

    # 3. Calculate Exceedance Probability for each weekly block and each threshold
    exceedance_matrix = []
    for t in thresholds:
        prob = df.groupby('WeekSlot')['MCP (Rs/MWh) *'].apply(lambda x: (x > t).mean() * 100)
        exceedance_matrix.append(prob)

    # Create DataFrame and reorder columns by actual day+time order
    res_df = pd.DataFrame(exceedance_matrix, index=thresholds)

    # Sort columns by (DayOfWeek, Time) so Mon 00:00 → Sun 23:45
    slot_order = (
        df[['WeekSlot', 'DayOrder', 'Time_Str']]
        .drop_duplicates()
        .sort_values(['DayOrder', 'Time_Str'])['WeekSlot']
        .tolist()
    )
    res_df = res_df[slot_order]  # reorder columns

    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharex=False)

    # --- Heatmap ---
    sns.heatmap(res_df, ax=ax1, cmap='YlOrRd',
                cbar_kws={'label': 'Probability of Exceeding (%)'})
    ax1.set_xlabel('Week Slot (Day + Time, 672 Blocks)', fontsize=12)
    ax1.set_ylabel('Price Threshold (Rs/MWh)', fontsize=12)

    # Show tick every 96 blocks (one per day) labelled with day name
    day_tick_positions = np.arange(0, 7 * 96, 96)
    ax1.set_xticks(day_tick_positions)
    ax1.set_xticklabels(day_order, rotation=45, fontsize=10)

    # Draw vertical lines to separate days
    for pos in day_tick_positions:
        ax1.axvline(x=pos, color='black', linewidth=0.8, linestyle='--')

    # --- Weekly overlay (one line per unique YearWeek) ---
    unique_weeks = df['YearWeek'].unique()
    for yw in unique_weeks:
        week_data = (
            df[df['YearWeek'] == yw]
            .sort_values(['DayOrder', 'Time_Str'])
        )
        ax2.plot(range(len(week_data)), week_data['MCP (Rs/MWh) *'],
                color='blue', alpha=0.08, linewidth=0.5)

    # Average weekly profile
    avg_profile = (
        df.groupby(['DayOrder', 'Time_Str'])['MCP (Rs/MWh) *']
        .mean()
        .reset_index()
        .sort_values(['DayOrder', 'Time_Str'])
    )
    std_profile = (
        df.groupby(['DayOrder', 'Time_Str'])['MCP (Rs/MWh) *']
        .std()
        .reset_index()
        .sort_values(['DayOrder', 'Time_Str'])
    )
    ax2.plot(range(len(avg_profile)), avg_profile['MCP (Rs/MWh) *'],
            color='red', linewidth=2, label='Multi-Year Weekly Average')
    ax2.plot(range(len(avg_profile)), avg_profile['MCP (Rs/MWh) *']+std_profile['MCP (Rs/MWh) *'],
            color='blue', linewidth=2, label='Multi-Year Weekly Average')

    ax2.plot(range(len(avg_profile)), avg_profile['MCP (Rs/MWh) *']- std_profile['MCP (Rs/MWh) *'],
            color='green', linewidth=2, label='Multi-Year Weekly Average')

    # Optional high/low bands — ensure they are 672 points long
    # ax2.plot(range(672), MCP_India_high)
    # ax2.plot(range(672), MCP_India_low)

    ax2.set_xlabel('Week Slot (Day + Time, 672 Blocks)', fontsize=12)
    ax2.set_ylabel('Price (Rs/MWh)', fontsize=12)

    # Day separators and labels on ax2
    for pos in day_tick_positions:
        ax2.axvline(x=pos, color='gray', linewidth=0.8, linestyle='--')
    ax2.set_xticks(day_tick_positions)
    ax2.set_xticklabels(day_order, rotation=45, fontsize=10)

    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.show()


    

    # ── 1. Prepare hourly slots ───────────────────────────────────────────────────
    df['Date']      = df['Timestamp'].dt.date
    df['Hour']      = df['Timestamp'].dt.hour
    df['Month']     = df['Timestamp'].dt.month
    df['MonthName'] = df['Timestamp'].dt.strftime('%b')   # Jan, Feb …
    df['Year']      = df['Timestamp'].dt.year

    # Unique hour slot per year: 0–8759  (or 0–8783 for leap years)
    # Use (month, day-of-year, hour) key so slots align across years even with leap years
    df['DayOfYear'] = df['Timestamp'].dt.dayofyear        # 1–365/366
    df['Day'] = df['Timestamp'].dt.day
    df['HourSlot']  = (df['DayOfYear'] - 1) * 24 + df['Hour']   # 0–8783

    # Readable label for ticks: month abbreviation at day-of-year 1 of each month
    month_starts = (
        df.groupby('Month')['HourSlot'].min()
        .reset_index()
        .sort_values('Month')
    )
    month_names = (
        df[['Month', 'MonthName']]
        .drop_duplicates()
        .sort_values('Month')['MonthName']
        .tolist()
    )

    # ── 2. Price thresholds ───────────────────────────────────────────────────────
    thresholds = np.arange(2000, 8500, 250)

    # ── 3. Exceedance probability (threshold × hourly slot) ──────────────────────
    exceedance = {}
    for t in thresholds:
        exceedance[t] = (
            df.groupby('HourSlot')['MCP (Rs/MWh) *']
            .apply(lambda x: (x > t).mean() * 100)
        )

    res_df = pd.DataFrame(exceedance).T                   # rows = thresholds, cols = hour slots
    res_df = res_df.sort_index(axis=1)                    # ensure slot 0 → max in order

    # ── 4. Annual hourly profile (mean ± 1 std) ───────────────────────────────────
    profile = (
        df.groupby('HourSlot')['MCP (Rs/MWh) *']
        .agg(mean='mean', std='std')
        .reset_index()
        .sort_values('HourSlot')
    )
    x       = profile['HourSlot'].values
    y_mean  = profile['mean'].values
    y_upper = y_mean + profile['std'].values
    y_lower = y_mean - profile['std'].values

    # ── 5. Plot ───────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 7), sharex=False)

    # ── Left: Exceedance heatmap ──────────────────────────────────────────────────
    sns.heatmap(
        res_df,
        ax=ax1,
        cmap='YlOrRd',
        cbar_kws={'label': 'Exceedance probability (%)'},
        xticklabels=False,          # we set custom ticks below
    )
    ax1.set_xlabel('Hour of year', fontsize=12)
    ax1.set_ylabel('Price threshold (Rs/MWh)', fontsize=12)
    #ax1.set_title('Hourly exceedance probability', fontsize=13)

    # Month separator lines and tick labels on heatmap
    for _, row in month_starts.iterrows():
        ax1.axvline(x=row['HourSlot'], color='black', linewidth=0.8, linestyle='--')

    ax1.set_xticks(month_starts['HourSlot'].values)
    ax1.set_xticklabels(month_names, rotation=45, fontsize=9)

    # ── Right: Mean ± std annual profile ─────────────────────────────────────────
    # Individual year traces (faint)
    for yr in df['Year'].unique():
        yr_data = (
            df[df['Year'] == yr]
            .sort_values('HourSlot')
        )
        ax2.plot(
            yr_data['HourSlot'],
            yr_data['MCP (Rs/MWh) *'],
            color='steelblue', alpha=0.06, linewidth=0.5,
        )

    # Shaded ± 1 std band
    ax2.fill_between(x, y_lower, y_upper,
                    color='steelblue', alpha=0.20, label='Mean ± 1 std')

    # Upper and lower edges as dashed lines
    ax2.plot(x, y_upper, color='steelblue', linewidth=1.2,
            linestyle='--', label='Upper (mean + 1 std)')
    ax2.plot(x, y_lower, color='steelblue', linewidth=1.2,
            linestyle='--', label='Lower (mean − 1 std)')

    # Mean profile on top
    ax2.plot(x, y_mean, color='crimson', linewidth=2.0, label='Annual average')

    # Month separators and labels
    for _, row in month_starts.iterrows():
        ax2.axvline(x=row['HourSlot'], color='gray', linewidth=0.8, linestyle='--')

    ax2.set_xticks(month_starts['HourSlot'].values)
    ax2.set_xticklabels(month_names, rotation=45, fontsize=9)
    ax2.set_xlabel('Hour of year', fontsize=12)
    ax2.set_ylabel('Price (Rs/MWh)', fontsize=12)
    #ax2.set_title('Annual hourly price profile (multi-year)', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.25)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    #plt.savefig('annual_hourly_profile.png', dpi=150, bbox_inches='tight')
    plt.show()



    # 2. DROP LEAP DAY directly from the source dataframe to eliminate the 24 extra hours
    df_filtered = df[~((df['Month'] == 2) & (df['Day'] == 29))]

    # 3. Group by 'Month', 'Day', and 'Hour' (NOT DayOfYear)
    # This forces any overlapping years to merge into the exact same 8,760 calendar slots
    profile = (
        df_filtered.groupby(['Month', 'Day', 'Hour'])['MCP (Rs/MWh) *']
        .agg(mean='mean', std='std')
        .reset_index()
        .sort_values(['Month', 'Day', 'Hour'])
    )

    # 4. Calculate bounds
    profile['upper'] = profile['mean'] + profile['std']
    profile['lower'] = profile['mean'] - profile['std']

    # 5. Assign EXACTLY 8,760 hours using a non-leap year (e.g., 2025)
    profile['Datetime'] = annual_df.Snapshot

    # 6. Set index (len(profile) will now be exactly 8760)
    profile = profile.set_index('Datetime')[['mean', 'upper', 'lower']]


    return profile.values.T