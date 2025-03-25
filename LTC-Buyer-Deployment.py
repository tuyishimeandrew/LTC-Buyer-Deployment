import streamlit as st
import pandas as pd
import numpy as np

def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: using the last 3 valid harvests (if both Fresh_Purchased and Dry_Output are numeric)
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
    
    st.markdown("### Upload CP Schedule Excel (for per date allocation)")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")
    
    if buyer_file is not None:
        # ===============================
        # PART 1: Global Allocation by CP
        # ===============================
        # Read buyer performance file (assume headers are on row 5, so header=4)
        df = pd.read_excel(buyer_file, header=4)
        df.rename(columns={
            df.columns[0]: "Harvest_ID",        # Column A
            df.columns[1]: "Buyer",             # Column B
            df.columns[3]: "Collection_Point",  # Column D
            df.columns[4]: "Fresh_Purchased",   # Column E
            df.columns[7]: "Juice_Loss_Kasese", # Column H
            df.columns[15]: "Dry_Output"        # Column P
        }, inplace=True)
        
        # Convert Juice_Loss_Kasese to numeric (non-numeric values become NaN)
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)
        
        # Compute global stats for each buyer
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            g_yield, g_juice = compute_buyer_stats(bdf)
            global_list.append({
                "Buyer": buyer,
                "Global_Yield": g_yield,
                "Global_Juice_Loss": g_juice
            })
        global_stats_df = pd.DataFrame(global_list)
        # Filter out buyers that don't meet the criteria so they are never considered
        global_stats_df = global_stats_df[
            (global_stats_df["Global_Yield"] >= 36) & (global_stats_df["Global_Juice_Loss"] <= 18)
        ].copy()
        
        # Prepare display columns for global stats
        global_stats_df["Yield three prior harvest(%)"] = global_stats_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_stats_df["Juice loss at Kasese(%)"] = global_stats_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        
        # Export buyer global performance
        st.subheader("Buyer Global Performance")
        st.dataframe(global_stats_df[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
        csv_buyer_stats = global_stats_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Buyer Global Performance CSV",
            data=csv_buyer_stats,
            file_name="buyer_global_performance.csv",
            mime="text/csv",
        )
        
        # Compute CP-specific yield for each CP–Buyer pair using the original df
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
        
        # Only consider buyers who meet global criteria.
        candidate_df = pd.merge(cp_stats, global_stats_df, on="Buyer", how="inner")
        
        # For each CP, rank the candidates by CP_Yield (descending)
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
        
        # Merge ranking info into candidate_df for display purposes
        display_df = pd.merge(candidate_df, ranking_df, on="Collection_Point", how="left")
        # Only show the buyer name in ranking column if it matches
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
        
        st.subheader("Global Buyer Performance by CP with Allocations")
        st.dataframe(final_display)
        csv_global = final_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Global Allocation CSV",
            data=csv_global,
            file_name="global_allocation.csv",
            mime="text/csv",
        )
        
        # ===============================
        # PART 2: Per Date Allocation
        # ===============================
        # Only proceed if a schedule file is uploaded.
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
            # Process allocation for each unique buying day
            for dt in sched_df["Date"].unique():
                # Get the CPs scheduled for that day.
                cp_list = sched_df[sched_df["Date"] == dt]["CP"].unique()
                # Consider only candidates whose CP is among those scheduled today.
                candidates_date = candidate_df[candidate_df["Collection_Point"].isin(cp_list)].copy()
                
                # ----- Step 1: Primary Allocation -----
                # For each buyer, choose their candidate row with maximum CP_Yield.
                if not candidates_date.empty:
                    best_candidates = candidates_date.loc[candidates_date.groupby("Buyer")["CP_Yield"].idxmax()]
                else:
                    best_candidates = pd.DataFrame()
                
                # Initialize allocation: dictionary with CP as key and list of allocated buyers.
                allocation = {cp: [] for cp in cp_list}
                assigned_buyers = set()
                
                # Primary allocation: assign each buyer from best_candidates to their CP if slot available (max 3 per CP).
                sorted_best = best_candidates.sort_values(by="CP_Yield", ascending=False)
                for _, row in sorted_best.iterrows():
                    cp = row["Collection_Point"]
                    buyer = row["Buyer"]
                    if buyer not in assigned_buyers and len(allocation[cp]) < 3:
                        allocation[cp].append(buyer)
                        assigned_buyers.add(buyer)
                
                # ----- Step 2: Secondary Allocation -----
                # Fill CPs with <3 slots using remaining candidate rows.
                remaining_candidates = candidates_date[~candidates_date["Buyer"].isin(assigned_buyers)].copy()
                remaining_candidates.sort_values(by="CP_Yield", ascending=False, inplace=True)
                for _, row in remaining_candidates.iterrows():
                    cp = row["Collection_Point"]
                    buyer = row["Buyer"]
                    if buyer not in assigned_buyers and len(allocation[cp]) < 3:
                        allocation[cp].append(buyer)
                        assigned_buyers.add(buyer)
                
                # ----- Step 3: Fallback Allocation -----
                # For any CP still with <3 slots, fill from global_stats_df (fallback candidates).
                fallback_candidates = global_stats_df[~global_stats_df["Buyer"].isin(assigned_buyers)].copy()
                fallback_candidates.sort_values(by="Global_Yield", ascending=False, inplace=True)
                for cp in cp_list:
                    while len(allocation[cp]) < 3 and not fallback_candidates.empty:
                        fallback_row = fallback_candidates.iloc[0]
                        buyer = fallback_row["Buyer"]
                        allocation[cp].append(buyer)
                        assigned_buyers.add(buyer)
                        fallback_candidates = fallback_candidates.drop(fallback_candidates.index[0])
                
                # ----- Step 4: Promotion for Empty CPs -----
                # If any CP remains completely unallocated, try to "promote" a candidate from another CP
                # who was originally second or third best (i.e. not the best allocated in their CP).
                def get_best_allocated(allocation):
                    """Return a dict mapping CP to its best (first allocated) buyer (or None)."""
                    return {cp: (buyers[0] if buyers else None) for cp, buyers in allocation.items()}
                
                best_allocated = get_best_allocated(allocation)
                empty_cps = [cp for cp in cp_list if len(allocation[cp]) == 0]
                # Continue promotion until no empty CP can be filled or no candidate available.
                while empty_cps:
                    promoted = False
                    for cp_empty in empty_cps:
                        # Look among all candidate rows for today that are from CPs in cp_list.
                        # We consider those whose buyer is either not allocated at all OR
                        # allocated in a non-best position in their original CP.
                        available_candidates = []
                        for _, row in candidates_date.iterrows():
                            buyer = row["Buyer"]
                            orig_cp = row["Collection_Point"]
                            # If buyer is not allocated, they are available.
                            if buyer not in assigned_buyers:
                                available_candidates.append(row)
                            # Otherwise, if allocated but not as the best in their original CP, they can be promoted.
                            elif best_allocated.get(orig_cp) != buyer:
                                available_candidates.append(row)
                        if available_candidates:
                            # Choose the candidate with the highest CP_Yield.
                            best_candidate = max(available_candidates, key=lambda r: r["CP_Yield"])
                            buyer = best_candidate["Buyer"]
                            orig_cp = best_candidate["Collection_Point"]
                            # If this buyer is allocated in their original CP, remove them from that allocation.
                            if buyer in allocation.get(orig_cp, []):
                                allocation[orig_cp].remove(buyer)
                                # Update best_allocated for the original CP.
                                best_allocated[orig_cp] = allocation[orig_cp][0] if allocation[orig_cp] else None
                            # Assign the candidate to the empty CP as the best buyer.
                            allocation[cp_empty].insert(0, buyer)
                            assigned_buyers.add(buyer)
                            promoted = True
                    # Update empty_cps and best_allocated after a promotion round.
                    best_allocated = get_best_allocated(allocation)
                    empty_cps = [cp for cp in cp_list if len(allocation[cp]) == 0]
                    # If no promotions occurred in this round, break out.
                    if not promoted:
                        break
                
                # Record final allocation for each CP on this buying day.
                for cp in cp_list:
                    best_buyer = allocation[cp][0] if len(allocation[cp]) >= 1 else ""
                    second_buyer = allocation[cp][1] if len(allocation[cp]) >= 2 else ""
                    third_buyer = allocation[cp][2] if len(allocation[cp]) >= 3 else ""
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
            csv_date = allocation_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Per Date Allocation CSV",
                data=csv_date,
                file_name="per_date_allocation.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
