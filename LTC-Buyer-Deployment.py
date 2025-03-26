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
    st.title("LTC Buyer Performance Allocation")

    st.markdown("### Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")
    
    st.markdown("### Upload CP Schedule Excel (for per date allocation)")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")
    
    if buyer_file is not None:
        # ===============================
        # PART 1: Buyer Global Performance (All Buyers)
        # ===============================
        # Read buyer performance file (assume headers on row 5, so header=4)
        df = pd.read_excel(buyer_file, header=4)
        df.rename(columns={
            df.columns[0]: "Harvest_ID",         # Column A
            df.columns[1]: "Buyer",              # Column B
            df.columns[3]: "Collection_Point",   # Column D
            df.columns[4]: "Fresh_Purchased",    # Column E
            df.columns[7]: "Juice_Loss_Kasese",  # Column H
            df.columns[15]: "Dry_Output"         # Column P
        }, inplace=True)
        
        # Ensure that Juice_Loss_Kasese is numeric (non-numeric become NaN)
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)
        
        # Compute global stats for all buyers (using all available data)
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            g_yield, g_juice = compute_buyer_stats(bdf)
            global_list.append({
                "Buyer": buyer,
                "Global_Yield": g_yield,
                "Global_Juice_Loss": g_juice
            })
        global_performance_all = pd.DataFrame(global_list)
        # Create display columns
        global_performance_all["Yield three prior harvest(%)"] = global_performance_all["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_performance_all["Juice loss at Kasese(%)"] = global_performance_all["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        
        st.subheader("Buyer Global Performance (All Buyers)")
        st.dataframe(global_performance_all[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
        csv_buyer_stats = global_performance_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Buyer Global Performance CSV",
            data=csv_buyer_stats,
            file_name="buyer_global_performance.csv",
            mime="text/csv",
        )
        
        # For allocation, only consider buyers meeting thresholds:
        filtered_global_stats_df = global_performance_all[
            (global_performance_all["Global_Yield"] >= 36) & (global_performance_all["Global_Juice_Loss"] <= 18)
        ].copy()
        
        # ===============================
        # PART 2: Allocation by CP (Display)
        # ===============================
        # Compute CP-specific yield per CP–Buyer pair from the original data.
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
        # For display purposes, only consider buyers from the filtered pool.
        candidate_df = pd.merge(cp_stats, filtered_global_stats_df, on="Buyer", how="inner")
        
        # For each CP, rank the candidates by CP yield (descending)
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
        
        # Merge ranking info into candidate_df for display
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
        
        st.subheader("Global Buyer Performance by CP with Optional Backups")
        st.dataframe(final_display)
        csv_global = final_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Global Allocation CSV",
            data=csv_global,
            file_name="global_allocation.csv",
            mime="text/csv",
        )
        
        # ===============================
        # PART 3: Per Date Allocation with Dynamic Reallocation (Promotions)
        # ===============================
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
            
            # Process each unique buying day
            for dt in sched_df["Date"].unique():
                # Get CPs scheduled for the day.
                cp_list = sched_df[sched_df["Date"] == dt]["CP"].unique()
                
                # Build candidate lists for each CP (from candidate_df), as lists of dictionaries
                candidates_by_cp = {}
                for cp in cp_list:
                    df_cp = candidate_df[candidate_df["Collection_Point"] == cp]
                    # Sort by CP_Yield descending
                    candidates_by_cp[cp] = df_cp.sort_values(by="CP_Yield", ascending=False).to_dict("records")
                
                # Initialize assignment dictionary: each CP gets a list of assigned buyers.
                # Each assigned entry is a tuple: (Buyer, CP_Yield from that candidate row)
                assignment = {cp: [] for cp in cp_list}
                # Also maintain a global mapping: buyer -> (CP, CP_Yield)
                assigned_buyers = {}
                
                # --- Primary Allocation ---
                for cp in cp_list:
                    for candidate in candidates_by_cp[cp]:
                        buyer = candidate["Buyer"]
                        if buyer not in assigned_buyers and len(assignment[cp]) < 3:
                            assignment[cp].append((buyer, candidate["CP_Yield"]))
                            assigned_buyers[buyer] = (cp, candidate["CP_Yield"])
                        if len(assignment[cp]) == 3:
                            break
                
                # --- Dynamic Reallocation (Promotion) ---
                changed = True
                while changed:
                    changed = False
                    # Iterate over each CP and its candidate list
                    for cp in cp_list:
                        for candidate in candidates_by_cp[cp]:
                            buyer = candidate["Buyer"]
                            cand_yield = candidate["CP_Yield"]
                            # If buyer is already assigned elsewhere...
                            if buyer in assigned_buyers:
                                current_cp, current_yield = assigned_buyers[buyer]
                                # And if candidate row from current CP shows a higher CP_Yield than the one in the current assignment...
                                if current_cp != cp and cand_yield > current_yield:
                                    # Reassign buyer from current_cp to cp
                                    # Remove buyer from the old CP assignment:
                                    assignment[current_cp] = [entry for entry in assignment[current_cp] if entry[0] != buyer]
                                    # Add buyer to the new CP at the top (as best candidate)
                                    assignment[cp].insert(0, (buyer, cand_yield))
                                    assigned_buyers[buyer] = (cp, cand_yield)
                                    changed = True
                                    # Now, try to refill the vacancy in the old CP from its candidate list
                                    for cand in candidates_by_cp[current_cp]:
                                        if cand["Buyer"] not in assigned_buyers and len(assignment[current_cp]) < 3:
                                            assignment[current_cp].append((cand["Buyer"], cand["CP_Yield"]))
                                            assigned_buyers[cand["Buyer"]] = (current_cp, cand["CP_Yield"])
                                            break
                                    # Once a change is made for this candidate, break to re‐start the loop
                                    break
                            else:
                                # If buyer is not assigned and there is a vacancy in cp, assign candidate.
                                if len(assignment[cp]) < 3:
                                    assignment[cp].append((buyer, cand_yield))
                                    assigned_buyers[buyer] = (cp, cand_yield)
                                    changed = True
                                    break
                        if changed:
                            # If any change occurred, restart checking from the first CP.
                            break
                
                # --- Fallback Allocation ---
                # For any CP with fewer than 3 assigned buyers, fill slots from the fallback pool.
                fallback_candidates = filtered_global_stats_df[~filtered_global_stats_df["Buyer"].isin(assigned_buyers.keys())].copy()
                fallback_candidates.sort_values(by="Global_Yield", ascending=False, inplace=True)
                for cp in cp_list:
                    while len(assignment[cp]) < 3 and not fallback_candidates.empty:
                        candidate_row = fallback_candidates.iloc[0]
                        buyer = candidate_row["Buyer"]
                        # Use Global_Yield as the surrogate yield value for fallback.
                        assignment[cp].append((buyer, candidate_row["Global_Yield"]))
                        assigned_buyers[buyer] = (cp, candidate_row["Global_Yield"])
                        fallback_candidates = fallback_candidates.drop(fallback_candidates.index[0])
                
                # Record final allocation for each CP on this day.
                for cp in cp_list:
                    best = assignment[cp][0][0] if len(assignment[cp]) >= 1 else ""
                    second = assignment[cp][1][0] if len(assignment[cp]) >= 2 else ""
                    third = assignment[cp][2][0] if len(assignment[cp]) >= 3 else ""
                    allocation_results.append({
                        "Date": dt.date(),
                        "Collection_Point": cp,
                        "Best Buyer for CP": best,
                        "Second Best Buyer for CP": second,
                        "Third Best Buyer for CP": third
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
