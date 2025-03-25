import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("LTC Buyer Performance Allocation")

    uploaded_file = st.file_uploader("Upload your Excel", type=["xlsx"])
    if uploaded_file is not None:
        # -------------------------------
        # 1. Read Excel and rename columns
        # -------------------------------
        # Assume row 5 has headers so we use header=4
        df = pd.read_excel(uploaded_file, header=4)
        df.rename(columns={
            df.columns[0]: "Harvest_ID",        # Column A
            df.columns[1]: "Buyer",             # Column B
            df.columns[3]: "Collection_Point",  # Column D
            df.columns[4]: "Fresh_Purchased",   # Column E
            df.columns[7]: "Juice_Loss_Kasese", # Column H
            df.columns[15]: "Dry_Output"        # Column P
        }, inplace=True)
        df.sort_index(ascending=False, inplace=True)

        # -------------------------------
        # 2. Calculate Global Stats for each Buyer
        #    - Global Yield: From the last 3 valid harvests (if both Fresh and Dry are valid)
        #    - Global Juice Loss: Most recent non-null value (multiplied by 100 to convert to %, rounded to 2 decimals)
        # -------------------------------
        global_stats = []
        grouped_buyer = df.groupby("Buyer")
        for buyer, buyer_df in grouped_buyer:
            valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
            valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
            valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
            last_3 = valid.head(3)
            total_fresh = last_3["Fresh_Purchased"].sum()
            total_dry = last_3["Dry_Output"].sum()
            yield_percentage = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

            latest_juice_loss_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
            if not latest_juice_loss_row.empty:
                juice_loss_val = latest_juice_loss_row["Juice_Loss_Kasese"].values[0]
                if pd.notnull(juice_loss_val) and isinstance(juice_loss_val, (int, float)):
                    juice_loss_val = round(juice_loss_val * 100, 2)
            else:
                juice_loss_val = np.nan

            global_stats.append({
                "Buyer": buyer,
                "Global_Yield": yield_percentage,
                "Global_Juice_Loss": juice_loss_val
            })

        global_stats_df = pd.DataFrame(global_stats)
        global_stats_df["Global_Yield_Display"] = global_stats_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_stats_df["Global_Juice_Loss_Display"] = global_stats_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        # -------------------------------
        # 3. Compute CP-Specific Yield for each Buyer at each CP
        #    CP_Yield = (sum of Dry_Output at CP for that buyer) / (sum of Fresh_Purchased at CP for that buyer) * 100
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

        # -------------------------------
        # 4. Merge CP stats with Global stats so we have for each CP-Buyer candidate:
        #    - Global Yield, Global Juice Loss and their display versions
        #    - CP_Yield and its display version
        # -------------------------------
        candidate_df = pd.merge(cp_stats, global_stats_df, on="Buyer", how="left")

        # -------------------------------
        # 5. Filter candidates based on global conditions:
        #    Only consider buyers with Global Yield >= 36% and Global Juice Loss <= 18%
        # -------------------------------
        candidate_df = candidate_df[
            (candidate_df["Global_Yield"] >= 0.36) & (candidate_df["Global_Juice_Loss"] <= 0.18)
        ].copy()

        # -------------------------------
        # 6. Allocate Buyers to CPs using a Greedy Algorithm:
        #    - Sort candidate rows by CP_Yield (descending)
        #    - Each buyer can only be allocated once overall
        #    - Each CP can have up to 3 allocated buyers
        # -------------------------------
        candidate_df.sort_values(by="CP_Yield", ascending=False, inplace=True)
        allocated_buyers = set()
        cp_allocations = {cp: [] for cp in candidate_df["Collection_Point"].unique()}
        allocation_records = []

        for idx, row in candidate_df.iterrows():
            buyer = row["Buyer"]
            cp = row["Collection_Point"]
            if buyer in allocated_buyers:
                continue
            if len(cp_allocations[cp]) >= 3:
                continue
            allocated_buyers.add(buyer)
            cp_allocations[cp].append(row)
            round_number = len(cp_allocations[cp])
            record = row.copy()
            record["Allocation_Round"] = round_number
            allocation_records.append(record)

        allocated_df = pd.DataFrame(allocation_records)

        # -------------------------------
        # 7. Create a Ranking DataFrame by CP based on allocation rounds:
        #    For each CP, assign:
        #      - Best Buyer for CP = Allocation_Round 1
        #      - Second Best Buyer for CP = Allocation_Round 2
        #      - Third Best Buyer for CP = Allocation_Round 3
        # -------------------------------
        ranking_list = []
        for cp, group in allocated_df.groupby("Collection_Point"):
            group_sorted = group.sort_values(by="Allocation_Round")
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

        # -------------------------------
        # 8. Merge ranking info back into candidate table (for display):
        #    For each CP–Buyer row, add ranking columns that show the buyer's name only on the allocated row.
        # -------------------------------
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

        # -------------------------------
        # 9. Final Display:
        #    Select and rename columns as required:
        #      - Collection_Point, Buyer,
        #      - Yield three prior harvest(%) from Global_Yield_Display,
        #      - Juice loss at Kasese(%) from Global_Juice_Loss_Display,
        #      - CP Yield(%) from CP_Yield_Display,
        #      - Best Buyer for CP, Second Best Buyer for CP, Third Best Buyer for CP
        # -------------------------------
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
