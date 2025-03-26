import streamlit as st
import pandas as pd
import numpy as np

def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: computed using the last 3 valid harvests (if both Fresh_Purchased and Dry_Output are numeric)
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
    st.title("LTC Buyer CP Deployment")

    st.markdown("### Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")
    
    st.markdown("### Upload CP Schedule Excel")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")
    
    if buyer_file is not None:
        # -------------------------------
        # PART 1: Buyer Global Performance (All Buyers)
        # -------------------------------
        df = pd.read_excel(buyer_file, header=4)
        df.rename(columns={
            df.columns[0]: "Harvest_ID",         # Column A
            df.columns[1]: "Buyer",              # Column B
            df.columns[3]: "Collection_Point",   # Column D
            df.columns[4]: "Fresh_Purchased",    # Column E
            df.columns[7]: "Juice_Loss_Kasese",  # Column H
            df.columns[15]: "Dry_Output"         # Column P
        }, inplace=True)
        
        # Ensure numeric conversion for Juice_Loss_Kasese
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)
        
        # Compute global stats for all buyers
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            g_yield, g_juice = compute_buyer_stats(bdf)
            global_list.append({
                "Buyer": buyer,
                "Global_Yield": g_yield,
                "Global_Juice_Loss": g_juice
            })
        global_performance_all = pd.DataFrame(global_list)
        global_performance_all["Yield three prior harvest(%)"] = global_performance_all["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_performance_all["Juice loss at Kasese(%)"] = global_performance_all["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        
        st.subheader("Buyer Global Performance")
        st.dataframe(global_performance_all[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
        csv_buyer_stats = global_performance_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Buyer Global Performance CSV",
            data=csv_buyer_stats,
            file_name="buyer_global_performance.csv",
            mime="text/csv",
        )
        
        # For allocation, only consider qualified buyers.
        filtered_global_stats_df = global_performance_all[
            (global_performance_all["Global_Yield"] >= 36) & (global_performance_all["Global_Juice_Loss"] <= 20)
        ].copy()
        
        # -------------------------------
        # PART 2: Allocation by CP (Display)
        # -------------------------------
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
        # For display purposes, only consider buyers in the filtered pool.
        candidate_df = pd.merge(cp_stats, filtered_global_stats_df, on="Buyer", how="inner")
        
        # For each CP, rank candidates by CP yield (descending) for display.
        ranking_list = []
        for cp, group in candidate_df.groupby("Collection_Point"):
            group_sorted = group.sort_values(by="CP_Yield", ascending=False)
            best = group_sorted.iloc[0]["Buyer"] if len(group_sorted) >= 1 else ""
            second = group_sorted.iloc[1]["Buyer"] if len(group_sorted) >= 2 else ""
            third = group_sorted.iloc[2]["Buyer"] if len(group_sorted) >= 3 else ""
            ranking_list.append({
                "Collection_Point": cp,
                "Best Buyer for CP": best,
                "Second Best Buyer for CP": second,
                "Third Best Buyer for CP": third
            })
        ranking_df = pd.DataFrame(ranking_list)
        
        display_df = pd.merge(candidate_df, ranking_df, on="Collection_Point", how="left")
        display_df["Best Buyer for CP"] = display_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Best Buyer for CP"] else "", axis=1
        )
        display_df["Second Best Buyer for CP"] = display_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Second Best Buyer for CP"] else "", axis=1
        )
        display_df["Third Best Buyer for CP"] = display_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Third Best Buyer for CP"] else "", axis=1
        )
        final_display = display_df[[
            "Collection_Point", "Buyer", 
            "Yield three prior harvest(%)", "Juice loss at Kasese(%)",
            "CP_Yield_Display",
            "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"
        ]].drop_duplicates().sort_values(by="Collection_Point")
        final_display.rename(columns={
            "CP_Yield_Display": "CP Yield(%)"
        }, inplace=True)
        
        st.subheader("Global Buyer Performance by CP")
        st.dataframe(final_display)
        csv_global = final_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Global Allocation CSV",
            data=csv_global,
            file_name="global_allocation.csv",
            mime="text/csv",
        )
        
        # -------------------------------
        # PART 3: Per Date Allocation with Dynamic Reallocation and Fallback for Blanks
        # -------------------------------
        if schedule_file is not None:
            sched_df = pd.read_excel(schedule_file)
            # Assume: Column A is Date and Column D is CP.
            sched_df.rename(columns={
                sched_df.columns[0]: "Date",
                sched_df.columns[3]: "CP"
            }, inplace=True)
            sched_df = sched_df[sched_df["Date"].notnull() & sched_df["CP"].notnull()].copy()
            sched_df["Date"] = pd.to_datetime(sched_df["Date"], errors="coerce")
            sched_df = sched_df[sched_df["Date"].notnull()]
            
            allocation_results = []
            # Process each unique buying day.
            for dt in sched_df["Date"].unique():
                cp_list = sched_df[sched_df["Date"] == dt]["CP"].unique()
                
                # Build candidate lists for each CP (from candidate_df), sorted by CP_Yield descending.
                candidates_by_cp = {}
                for cp in cp_list:
                    df_cp = candidate_df[candidate_df["Collection_Point"] == cp]
                    candidates_by_cp[cp] = df_cp.sort_values(by="CP_Yield", ascending=False).to_dict("records")
                
                # Initialize assignment for each CP as an empty list.
                assignment = {cp: [] for cp in cp_list}
                # Track buyers already assigned on this day.
                assigned_global = set()
                
                # Process allocation in three rounds: round 0 (best), round 1 (second best), round 2 (third best).
                # This ensures priority is given first to filling the best slot, then the second, then the third.
                for round_no in range(3):
                    # Proposals: For each CP, select a candidate not yet assigned.
                    proposals = {}  # cp -> (buyer, candidate_CP_Yield)
                    for cp in cp_list:
                        if len(assignment[cp]) > round_no:
                            continue  # already filled for this round
                        candidate_found = None
                        for candidate in candidates_by_cp.get(cp, []):
                            if candidate["Buyer"] not in assigned_global:
                                candidate_found = (candidate["Buyer"], candidate["CP_Yield"])
                                break
                        proposals[cp] = candidate_found  # may be None if no candidate available
                        
                    # Resolve conflicts: if a buyer is proposed for more than one CP in this round,
                    # keep the proposal with the highest CP_Yield and remove the others.
                    buyer_proposals = {}
                    for cp, proposal in proposals.items():
                        if proposal is not None:
                            buyer = proposal[0]
                            buyer_proposals.setdefault(buyer, []).append((cp, proposal[1]))
                    for buyer, cp_list_for_buyer in buyer_proposals.items():
                        if len(cp_list_for_buyer) > 1:
                            chosen_cp = max(cp_list_for_buyer, key=lambda x: x[1])[0]
                            for cp, _ in cp_list_for_buyer:
                                if cp != chosen_cp:
                                    proposals[cp] = None
                                    
                    # Assign the resolved proposals.
                    for cp in cp_list:
                        if proposals.get(cp) is not None:
                            buyer, _ = proposals[cp]
                            assignment[cp].append(buyer)
                            assigned_global.add(buyer)
                    
                    # Fallback: For any CP that still does not have a candidate in this round,
                    # fill from the remaining general pool (prioritizing second best before third).
                    fallback_pool = filtered_global_stats_df[~filtered_global_stats_df["Buyer"].isin(assigned_global)]
                    fallback_pool = fallback_pool.sort_values(by="Global_Yield", ascending=False)
                    for cp in cp_list:
                        if len(assignment[cp]) <= round_no:
                            if not fallback_pool.empty:
                                fallback_buyer = fallback_pool.iloc[0]["Buyer"]
                                assignment[cp].append(fallback_buyer)
                                assigned_global.add(fallback_buyer)
                                fallback_pool = fallback_pool.drop(fallback_pool.index[0])
                
                # Record final allocation for the day.
                for cp in cp_list:
                    best = assignment[cp][0] if len(assignment[cp]) >= 1 else ""
                    second = assignment[cp][1] if len(assignment[cp]) >= 2 else ""
                    third = assignment[cp][2] if len(assignment[cp]) >= 3 else ""
                    allocation_results.append({
                        "Date": dt.date(),
                        "Collection_Point": cp,
                        "Best Buyer for CP": best,
                        "Second Best Buyer for CP": second,
                        "Third Best Buyer for CP": third
                    })
            
            allocation_df = pd.DataFrame(allocation_results)
            allocation_df.sort_values(by=["Date", "Collection_Point"], inplace=True)
            st.subheader("Buyer Allocation according to CP schedule")
            st.dataframe(allocation_df)
            csv_date = allocation_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Per Date Allocation CSV",
                data=csv_date,
                file_name="per_date_allocation.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
