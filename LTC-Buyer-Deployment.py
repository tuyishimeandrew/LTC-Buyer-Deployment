import streamlit as st
import pandas as pd
import numpy as np

def compute_buyer_stats(buyer_df):
    """Calculate global buyer statistics from harvest data"""
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    
    # Get last 3 valid entries (assuming reverse chronological order)
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    # Get most recent juice loss value
    juice_loss_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    juice_loss = round(juice_loss_row["Juice_Loss_Kasese"].values[0] * 100, 2) if not juice_loss_row.empty else np.nan
    
    return global_yield, juice_loss

def main():
    st.title("LTC Buyer CP Deployment")

    # File Upload Section
    st.markdown("### Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("", type=["xlsx"], key="buyer")
    st.markdown("### Upload CP Schedule Excel")
    schedule_file = st.file_uploader("", type=["xlsx"], key="schedule")

    if buyer_file:
        # ======================
        # PART 1: Global Performance
        # ======================
        df = pd.read_excel(buyer_file, header=4)
        df.columns = df.columns.str.replace('\n', ' ').str.strip()
        
        # Column mapping based on Excel structure
        col_mapping = {
            df.columns[0]: "Harvest_ID",
            df.columns[1]: "Buyer",
            df.columns[3]: "Collection_Point",
            df.columns[4]: "Fresh_Purchased",
            df.columns[7]: "Juice_Loss_Kasese",
            df.columns[15]: "Dry_Output"
        }
        df.rename(columns=col_mapping, inplace=True)
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)

        # Calculate global metrics
        global_stats = []
        for buyer, group in df.groupby("Buyer"):
            yield_val, juice_loss = compute_buyer_stats(group)
            global_stats.append({
                "Buyer": buyer,
                "Global_Yield": yield_val,
                "Global_Juice_Loss": juice_loss
            })
        global_df = pd.DataFrame(global_stats)
        
        # Format display columns
        global_df["Yield three prior harvest(%)"] = global_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_df["Juice loss at Kasese(%)"] = global_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        st.subheader("Buyer Global Performance")
        st.dataframe(global_df[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
        
        # ======================
        # PART 2: CP Allocation
        # ======================
        qualified_buyers = global_df[
            (global_df["Global_Yield"] >= 36) & 
            (global_df["Global_Juice_Loss"] <= 20)
        ].copy()

        # Calculate CP-level metrics
        cp_data = df.groupby(["Collection_Point", "Buyer"]).agg({
            "Fresh_Purchased": "sum",
            "Dry_Output": "sum"
        }).reset_index()
        cp_data["CP_Yield"] = cp_data.apply(
            lambda r: (r["Dry_Output"] / r["Fresh_Purchased"]) * 100 if r["Fresh_Purchased"] > 0 else np.nan,
            axis=1
        )
        
        # Merge with qualified buyers
        candidate_pool = cp_data.merge(qualified_buyers, on="Buyer", how="inner")
        
        # Determine rankings per CP
        ranked_allocations = []
        for cp, group in candidate_pool.groupby("Collection_Point"):
            sorted_group = group.sort_values("CP_Yield", ascending=False)
            ranked_allocations.append({
                "Collection_Point": cp,
                "Best Buyer for CP": sorted_group.iloc[0]["Buyer"] if len(sorted_group) >= 1 else "",
                "Second Best Buyer for CP": sorted_group.iloc[1]["Buyer"] if len(sorted_group) >= 2 else "",
                "Third Best Buyer for CP": sorted_group.iloc[2]["Buyer"] if len(sorted_group) >= 3 else ""
            })
        ranking_df = pd.DataFrame(ranked_allocations)

        # Format final display
        final_display = candidate_pool.merge(ranking_df, on="Collection_Point", how="left")
        for rank_col in ["Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]:
            final_display[rank_col] = final_display.apply(
                lambda r: r["Buyer"] if r["Buyer"] == r[rank_col] else "", axis=1
            )
        final_display = final_display[[
            "Collection_Point", "Buyer", "Yield three prior harvest(%)",
            "Juice loss at Kasese(%)", "CP_Yield", rank_cols
        ]].drop_duplicates()

        st.subheader("CP Allocation Rankings")
        st.dataframe(final_display)

        # ======================
        # PART 3: Schedule-Based Allocation
        # ======================
        if schedule_file:
            schedule_df = pd.read_excel(schedule_file)
            schedule_df.rename(columns={
                schedule_df.columns[0]: "Date",
                schedule_df.columns[3]: "CP"
            }, inplace=True)
            schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
            schedule_df = schedule_df.dropna(subset=["Date", "CP"])
            
            allocation_results = []
            for date in schedule_df["Date"].unique():
                date_cps = schedule_df[schedule_df["Date"] == date]["CP"].unique()
                
                # Initialize allocation structures
                cp_candidates = {cp: [] for cp in date_cps}
                assigned_buyers = set()
                
                # Three-round allocation process
                for round_num in range(3):
                    # Stage 1: Primary candidate selection
                    candidate_proposals = {}
                    for cp in date_cps:
                        candidates = candidate_pool[
                            (candidate_pool["Collection_Point"] == cp) &
                            (~candidate_pool["Buyer"].isin(assigned_buyers))
                        ].sort_values("CP_Yield", ascending=False)
                        
                        if not candidates.empty:
                            candidate_proposals[cp] = candidates.iloc[0]["Buyer"]
                        else:
                            candidate_proposals[cp] = None
                    
                    # Stage 2: Conflict resolution
                    buyer_conflicts = {}
                    for cp, buyer in candidate_proposals.items():
                        if buyer:
                            buyer_conflicts.setdefault(buyer, []).append(cp)
                    
                    # Resolve conflicts by keeping highest CP yield
                    for buyer, conflict_cps in buyer_conflicts.items():
                        if len(conflict_cps) > 1:
                            best_cp = conflict_cps[0]  # Simplified selection
                            for cp in conflict_cps[1:]:
                                candidate_proposals[cp] = None
                    
                    # Stage 3: Assign candidates
                    for cp, buyer in candidate_proposals.items():
                        if buyer and (cp not in cp_candidates or len(cp_candidates[cp]) <= round_num):
                            cp_candidates[cp].append(buyer)
                            assigned_buyers.add(buyer)
                    
                    # Stage 4: Fallback allocation
                    fallback_pool = qualified_buyers[
                        ~qualified_buyers["Buyer"].isin(assigned_buyers)
                    ].sort_values("Global_Yield", ascending=False)
                    
                    for cp in date_cps:
                        if len(cp_candidates[cp]) <= round_num and not fallback_pool.empty:
                            fallback_buyer = fallback_pool.iloc[0]["Buyer"]
                            cp_candidates[cp].append(fallback_buyer)
                            assigned_buyers.add(fallback_buyer)
                            fallback_pool = fallback_pool.iloc[1:]
                
                # Record results
                for cp in date_cps:
                    allocation_results.append({
                        "Date": date.date(),
                        "Collection_Point": cp,
                        "Best Buyer": cp_candidates[cp][0] if len(cp_candidates[cp]) > 0 else "",
                        "Second Buyer": cp_candidates[cp][1] if len(cp_candidates[cp]) > 1 else "",
                        "Third Buyer": cp_candidates[cp][2] if len(cp_candidates[cp]) > 2 else ""
                    })
            
            allocation_df = pd.DataFrame(allocation_results)
            st.subheader("Scheduled Allocations")
            st.dataframe(allocation_df)

if __name__ == "__main__":
    main()
