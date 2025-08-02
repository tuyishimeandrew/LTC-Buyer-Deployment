import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: using last 3 valid harvests
      - Global juice loss: most recent non-null value (×100, 2 decimals)
      - Average of individual yields: mean of (Dry_Output/Fresh_Purchased*100) over last 3 valid harvests
    """
    # Filter rows with numeric Fresh_Purchased and Dry_Output
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]

    # Take the last 3 harvests
    last_3 = valid.head(3)

    # Global yield across last 3 harvests
    total_fresh_3 = last_3["Fresh_Purchased"].sum()
    total_dry_3 = last_3["Dry_Output"].sum()
    global_yield = (total_dry_3 / total_fresh_3) * 100 if total_fresh_3 > 0 else np.nan

    # Compute individual yields for each of the last 3 and average them
    if not last_3.empty:
        yields = last_3.apply(
            lambda r: (r["Dry_Output"] / r["Fresh_Purchased"] * 100) if r["Fresh_Purchased"] > 0 else np.nan,
            axis=1
        )
        avg_individual_yield = yields.mean()
    else:
        avg_individual_yield = np.nan

    # Most recent juice loss (unaffected by Dry_Output)
    latest_loss = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_loss.empty:
        jl = latest_loss["Juice_Loss_Kasese"].iloc[0]
        juice_loss_val = round(jl * 100, 2) if isinstance(jl, (int, float)) else np.nan
    else:
        juice_loss_val = np.nan

    return global_yield, juice_loss_val, avg_individual_yield


def main():
    st.title("LTC Buyer CP Deployment")

    # File uploads
    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")

    if buyer_file:
        # Dynamically load first sheet from buyer file
        xls = pd.ExcelFile(buyer_file)
        first_sheet = xls.sheet_names[0]
        df = pd.read_excel(buyer_file, sheet_name=first_sheet, header=4)

        # Rename and clean
        df.rename(columns={
            df.columns[0]: "Harvest_ID",
            df.columns[1]: "Buyer",
            df.columns[3]: "Collection_Point",
            df.columns[4]: "Fresh_Purchased",
            df.columns[7]: "Juice_Loss_Kasese",
            df.columns[15]: "Dry_Output"
        }, inplace=True)
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        # Drop invalid or empty harvest rows
        df = df[df["Harvest_ID"].notnull() & (df["Harvest_ID"] != 0)]
        # Drop rows missing Juice_Loss_Kasese only (preserve Dry_Output for juice history)
        df = df.dropna(subset=["Juice_Loss_Kasese"])
        df.sort_index(ascending=False, inplace=True)

        # Compute per-buyer stats
        stats = []
        for buyer, group in df.groupby("Buyer"):
            g_yield, g_juice, avg_yield3 = compute_buyer_stats(group)
            stats.append({
                "Buyer": buyer,
                "Global_Yield": g_yield,
                "Global_Juice_Loss": g_juice,
                "Avg_Yield_3": avg_yield3
            })
        global_df = pd.DataFrame(stats)

        # Overall yield across all records (drop rows with missing Fresh or Dry)
        valid_all = df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
        valid_all = valid_all[valid_all["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
        valid_all = valid_all[valid_all["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
        agg_all = (
            valid_all
            .groupby("Buyer")
            .agg(
                Total_Purchased=("Fresh_Purchased", "sum"),
                Total_Dry_Output=("Dry_Output", "sum")
            )
            .reset_index()
        )
        agg_all["Overall_Yield"] = np.where(
            agg_all["Total_Purchased"] > 0,
            agg_all["Total_Dry_Output"] / agg_all["Total_Purchased"] * 100,
            np.nan
        )

        # Merge and format
        perf_df = global_df.merge(agg_all, on="Buyer", how="left")
        perf_df["Yield three prior harvest(%)"] = perf_df["Global_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        perf_df["Yield three prior harvest(%) (Unweighted)"] = perf_df["Avg_Yield_3"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        perf_df["Juice loss at Kasese(%)"] = perf_df["Global_Juice_Loss"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        perf_df["Overall Yield (All)(%)"] = perf_df["Overall_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        perf_df["Total Purchased"] = perf_df["Total_Purchased"].fillna(0)

        # Display global performance
        st.subheader("Buyer Global Performance")
        cols = [
            "Buyer",
            "Total_Dry_Output",
            "Yield three prior harvest(%)",
            "Yield three prior harvest(%) (Unweighted)",
            "Juice loss at Kasese(%)",
            "Overall Yield (All)(%)",
            "Total Purchased"
        ]
        st.dataframe(perf_df[cols])
        st.download_button(
            label="Download Buyer Global Performance CSV",
            data=perf_df.to_csv(index=False).encode("utf-8"),
            file_name="buyer_global_performance.csv",
            mime="text/csv"
        )

        # Part 2: Allocation by schedule
        if schedule_file:
            sched_xls = pd.ExcelFile(schedule_file)
            sched_sheet = sched_xls.sheet_names[0]
            sched = pd.read_excel(schedule_file, sheet_name=sched_sheet)

            sched.rename(columns={sched.columns[0]: "Date", sched.columns[3]: "CP"}, inplace=True)
            sched = sched.dropna(subset=["Date", "CP"])
            sched["Date"] = pd.to_datetime(sched["Date"], errors="coerce")
            sched = sched.dropna(subset=["Date"])

            qualified = perf_df[(perf_df["Global_Yield"] >= 37) &
                                 (perf_df["Overall_Yield"] >= 37) &
                                 (perf_df["Global_Juice_Loss"] <= 20)].copy()

            cp_stats = (
                df.groupby(["Collection_Point", "Buyer"]).agg(
                    Fresh_Purchased=("Fresh_Purchased", "sum"),
                    Dry_Output=("Dry_Output", "sum")
                )
                .reset_index()
            )
            cp_stats["CP_Yield"] = cp_stats.apply(
                lambda r: (r["Dry_Output"] / r["Fresh_Purchased"] * 100) if r["Fresh_Purchased"] > 0 else np.nan,
                axis=1
            )
            candidates = cp_stats.merge(qualified, on="Buyer", how="inner")

            allocations = []
            for dt in_sched:**ERROR**
import streamlit as st
import pandas as pd
import numpy as np

**NOTE** The code above truncated—please check the last merge and allocation logic block to ensure it’s correctly closed and indented.
