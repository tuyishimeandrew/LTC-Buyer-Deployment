import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("LTC Buyer Performance Allocation")

    uploaded_file = st.file_uploader("Upload your Excel", type=["xlsx"])
    if uploaded_file is not None:
        # 1. Read Excel and rename columns (assuming headers are on row 5 → header=4)
        df = pd.read_excel(uploaded_file, header=4)
        df.rename(columns={
            df.columns[0]: "Harvest_ID",        # Column A
            df.columns[1]: "Buyer",             # Column B
            df.columns[3]: "Collection_Point",  # Column D
            df.columns[4]: "Fresh_Purchased",   # Column E
            df.columns[7]: "Juice_Loss_Kasese", # Column H
            df.columns[15]: "Dry_Output"        # Column P
        }, inplace=True)
        # Sort descending so that head(3) picks the most recent valid harvests
        df.sort_index(ascending=False, inplace=True)

        # 2. Calculate Global Stats per Buyer
        #    Global Yield is computed from the last 3 valid harvests (only if both Fresh and Dry are valid)
        #    Global Juice Loss is taken as the most recent non-null value, multiplied by 100 and rounded to 2 decimals.
        global_stats = []
        for buyer, buyer_df in df.groupby("Buyer"):
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

            global_stats.append({
                "Buyer": buyer,
                "Global_Yield": global_yield,
                "Global_Juice_Loss": juice_loss_val
            })
        global_stats_df = pd.DataFrame(global_stats)
        global_stats_df["Global_Yield_Display"] = global_stats_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_stats_df["Global_Juice_Loss_Display"] = global_stats_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        # 3. Compute CP-Specific Yield for each Buyer at each CP
        #    CP_Yield = (sum of Dry_Output at CP) / (sum of Fresh_Purchased at CP) * 100
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

        # 4. Merge CP stats with Global stats for each CP–Buyer candidate
        candidate_df = pd.merge(cp_stats, global_stats_df, on="Buyer", how="left")

        # 5. Filter candidates based on global conditions only (global yield >= 36% and global juice loss <= 18%)
        candidate_df = candidate_df[
            (candidate_df["Global_Yield"] >= 36) & (candidate_df["Global_Juice_Loss"] <= 18)
        ].copy()

        # 6. For each CP, rank candidates by CP_Yield (descending order)
        ranking_list = []
        for cp, group in candidate_df.groupby("Collection_Point"):
            group_sorted = group.sort_values(by="CP_Yield", ascending=False)
            best = group_sorted.iloc[0]["Buyer"] if len(group_sorted) >= 1 else ""
            second = group_sorted.iloc[1]["Buyer"] if len(group_sorted) >= 2 else ""
            third = group_sorted.iloc[2]["Buyer"] if len(group_sorted) >= 3 else ""
            ranking_list.append({
                "Collection_Point": cp,
                "Best_Buyer": best,
                "Second_Buyer": second,
                "Third_Buyer": third
            })
        ranking_df = pd.DataFrame(ranking_list)

        # 7. Merge ranking info back into the candidate table (for display)
        display_df = pd.merge(candidate_df, ranking_df, on="Collection_Point", how="left")
        display_df["Best Buyer for CP"] = display_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Best_Buyer"] else "", axis=1
        )
        display_df["Second Best Buyer for CP"] = display_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Second_Buyer"] else "", axis=1
        )
        display_df["Third Best Buyer for CP"] = display_df.apply(
            lambda row: row["Buyer"] if row["Buyer"] == row["Third_Buyer"] else "", axis=1
        )

        # 8. Final Display: select and rename columns
        final_display = display_df[[
            "Collection_Point", "Buyer", 
            "Global_Yield_Display", "Global_Juice_Loss_Display",
            "CP_Yield_Display",
            "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"
        ]].drop_duplicates().sort_values(by="Collection_Point")

        final_display.rename(columns={
            "Global_Yield_Display": "Yield three prior harvest(%)",
            "Global_Juice_Loss_Display": "Juice loss at Kasese(%)",
            "CP_Yield_Display": "CP Yield(%)"
        }, inplace=True)

        st.subheader("Buyer Performance by CP with Allocations")
        st.dataframe(final_display)

if __name__ == "__main__":
    main()
