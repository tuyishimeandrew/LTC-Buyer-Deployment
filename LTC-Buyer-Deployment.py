import streamlit as st
import pandas as pd
import numpy as np

def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: using last 3 valid harvests (if both Fresh_Purchased and Dry_Output are numeric)
      - Global juice loss: the most recent non-null value (multiplied by 100 and rounded to 2 decimals)
    """
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan
    latest_juice_loss_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_juice_loss_row.empty:
        juice_loss_val = latest_juice_loss_row["Juice_Loss_Kasese"].values[0]
        if pd.notnull(juice_loss_val) and isinstance(juice_loss_val, (int, float)):
            juice_loss_val = round(juice_loss_val * 100, 2)
    else:
        juice_loss_val = np.nan
    return global_yield, juice_loss_val

def main():
    st.title("LTC Buyer Performance Allocation")

    st.markdown("### Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")
    
    st.markdown("### Upload CP Schedule Excel")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")
    
    if buyer_file is not None and schedule_file is not None:
        # -------------------------------
        # Process Buyer Performance Excel
        # -------------------------------
        # Read the buyer performance file; assume headers are on row 5 (header=4)
        df = pd.read_excel(buyer_file, header=4)
        df.rename(columns={
            df.columns[0]: "Harvest_ID",        # Column A
            df.columns[1]: "Buyer",             # Column B
            df.columns[3]: "Collection_Point",  # Column D
            df.columns[4]: "Fresh_Purchased",   # Column E
            df.columns[7]: "Juice_Loss_Kasese", # Column H
            df.columns[15]: "Dry_Output"        # Column P
        }, inplace=True)
        df.sort_index(ascending=False, inplace=True)

        # Compute global stats for each buyer (grouped by Buyer)
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            g_yield, g_juice = compute_buyer_stats(bdf)
            global_list.append({
                "Buyer": buyer,
                "Global_Yield": g_yield,
                "Global_Juice_Loss": g_juice
            })
        global_stats_df = pd.DataFrame(global_list)
        # Format display columns
        global_stats_df["Global_Yield_Display"] = global_stats_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_stats_df["Global_Juice_Loss_Display"] = global_stats_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        # Compute CP-specific yields: group by Collection_Point and Buyer
        cp_stats = df.groupby(["Collection_Point", "Buyer"]).agg({
            "Fresh_Purchased": "sum",
            "Dry_Output": "sum"
        }).reset_index()
        cp_stats["CP_Yield"] = cp_stats.apply(
            lambda row: (row["Dry_Output"] / row["Fresh_Purchased"]) * 100 
                        if row["Fresh_Purchased"] > 0 else np.nan, axis=1
        )
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        # Merge global stats into cp_stats by Buyer
        candidate_df = pd.merge(cp_stats, global_stats_df, on="Buyer", how="left")
        # Filter: only keep candidates with Global Yield >= 36% and Global Juice Loss <= 18%
        candidate_df = candidate_df[
            (candidate_df["Global_Yield"] >= 36) & (candidate_df["Global_Juice_Loss"] <= 18)
        ].copy()

        # -------------------------------
        # Process CP Schedule Excel
        # -------------------------------
        # For schedule file, assume:
        # - Column A has Date (rename it 'Date') and we only take rows where Date is not null.
        # - Column D has the CP (rename it 'CP')
        sched_df = pd.read_excel(schedule_file)
        # Rename columns for clarity (assuming A and D by position)
        sched_df.rename(columns={
            sched_df.columns[0]: "Date",
            sched_df.columns[3]: "CP"
        }, inplace=True)
        # Keep only rows with non-null Date and CP
        sched_df = sched_df[sched_df["Date"].notnull() & sched_df["CP"].notnull()].copy()
        # Convert Date column to datetime
        sched_df["Date"] = pd.to_datetime(sched_df["Date"], errors="coerce")
        sched_df = sched_df[sched_df["Date"].notnull()]  # drop rows with conversion errors

        # -------------------------------
        # Allocation per Date:
        # For each unique date in schedule, we:
        #  a) Get the list of CPs for that date.
        #  b) From candidate_df, take only rows whose Collection_Point is in that list.
        #  c) For each buyer in this subset, choose the candidate row with maximum CP_Yield.
        #  d) Group these chosen allocations by CP and rank them by CP_Yield descending.
        #  e) Record Best, Second, and Third buyer per CP.
        # -------------------------------
        allocation_results = []  # will store dicts with Date, CP, and allocated buyers

        for dt in sched_df["Date"].unique():
            # For this date, get the CPs from schedule (unique)
            cp_list = sched_df[sched_df["Date"] == dt]["CP"].unique()
            # Filter candidate_df to those CPs
            candidates_date = candidate_df[candidate_df["Collection_Point"].isin(cp_list)].copy()
            # For each buyer, choose the candidate row with highest CP_Yield (if multiple CPs available)
            if not candidates_date.empty:
                best_candidates = candidates_date.loc[candidates_date.groupby("Buyer")["CP_Yield"].idxmax()]
            else:
                best_candidates = pd.DataFrame()
            # Now group by Collection_Point (each CP on this date) and sort by CP_Yield descending
            for cp in cp_list:
                cp_group = best_candidates[best_candidates["Collection_Point"] == cp].copy()
                cp_group.sort_values(by="CP_Yield", ascending=False, inplace=True)
                allocated = cp_group.copy()
                # Only distinct buyers; by construction each buyer appears only once.
                # Now assign ranking if available.
                best_buyer = allocated.iloc[0]["Buyer"] if len(allocated) >= 1 else ""
                second_buyer = allocated.iloc[1]["Buyer"] if len(allocated) >= 2 else ""
                third_buyer = allocated.iloc[2]["Buyer"] if len(allocated) >= 3 else ""
                allocation_results.append({
                    "Date": dt.date(),
                    "Collection_Point": cp,
                    "Best Buyer for CP": best_buyer,
                    "Second Best Buyer for CP": second_buyer,
                    "Third Best Buyer for CP": third_buyer
                })

        allocation_df = pd.DataFrame(allocation_results)
        allocation_df.sort_values(by=["Date", "Collection_Point"], inplace=True)

        st.subheader("Buyer Allocation per CP per Date")
        st.dataframe(allocation_df)

if __name__ == "__main__":
    main()
