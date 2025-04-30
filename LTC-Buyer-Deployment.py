import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: computed using the last 3 valid harvests
      - Global juice loss: the most recent non-null value (multiplied by 100 and rounded to 2 decimals)
    """
    # Filter valid numeric entries
    valid = (
        buyer_df
        .dropna(subset=["Fresh_Purchased", "Dry_Output"])  # non-null
        .loc[buyer_df["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
        .loc[buyer_df["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))],
    )[0]
    # Take the most recent 3
    last_3 = valid.head(3)
    total_fresh_3 = last_3["Fresh_Purchased"].sum()
    total_dry_3 = last_3["Dry_Output"].sum()
    global_yield = (total_dry_3 / total_fresh_3) * 100 if total_fresh_3 > 0 else np.nan

    # Latest juice loss
    latest_loss = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_loss.empty:
        jl = latest_loss["Juice_Loss_Kasese"].values[0]
        juice_loss_val = round(jl * 100, 2) if isinstance(jl, (int, float)) else np.nan
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
            df.columns[0]: "Harvest_ID",
            df.columns[1]: "Buyer",
            df.columns[3]: "Collection_Point",
            df.columns[4]: "Fresh_Purchased",
            df.columns[7]: "Juice_Loss_Kasese",
            df.columns[15]: "Dry_Output"
        }, inplace=True)
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)

        # 3-harvest metrics
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            g_yield, g_juice = compute_buyer_stats(bdf)
            global_list.append({
                "Buyer": buyer,
                "Global_Yield": g_yield,
                "Global_Juice_Loss": g_juice
            })
        global_performance_all = pd.DataFrame(global_list)

        # Overall metrics (all harvests)
        agg_all = (
            df.groupby("Buyer")
              .agg(
                  Total_Purchased=("Fresh_Purchased", "sum"),
                  Total_Dry_Output=("Dry_Output", "sum")
              )
              .reset_index()
        )
        agg_all["Overall_Yield"] = np.where(
            agg_all["Total_Purchased"] > 0,
            (agg_all["Total_Dry_Output"] / agg_all["Total_Purchased"]) * 100,
            np.nan
        )

        # Merge metrics
        global_performance_all = (
            global_performance_all
              .merge(agg_all, on="Buyer", how="left")
        )

        # Format for display
        global_performance_all["Yield three prior harvest(%)"] = (
            global_performance_all["Global_Yield"]
              .apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        )
        global_performance_all["Juice loss at Kasese(%)"] = (
            global_performance_all["Global_Juice_Loss"]
              .apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        )
        global_performance_all["Overall Yield (All)(%)"] = (
            global_performance_all["Overall_Yield"]
              .apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        )
        global_performance_all["Total Purchased"] = (
            global_performance_all["Total_Purchased"].fillna(0)
        )

        st.subheader("Buyer Global Performance")
        st.dataframe(
            global_performance_all[[
                "Buyer",
                "Yield three prior harvest(%)",
                "Juice loss at Kasese(%)",
                "Overall Yield (All)(%)",
                "Total Purchased"
            ]]
        )
        csv_buyer_stats = global_performance_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Buyer Global Performance CSV",
            data=csv_buyer_stats,
            file_name="buyer_global_performance.csv",
            mime="text/csv",
        )

        # Filter qualified buyers
        filtered_global_stats_df = global_performance_all[
            (global_performance_all["Global_Yield"] >= 37) &
            (global_performance_all["Global_Juice_Loss"] <= 20)
        ].copy()

        # -------------------------------
        # PART 2: Allocation by CP (Display)
        # -------------------------------
        cp_stats = (
            df.groupby(["Collection_Point", "Buyer"])  
              .agg({
                  "Fresh_Purchased": "sum",
                  "Dry_Output": "sum"
              })
              .reset_index()
        )
        cp_stats["CP_Yield"] = cp_stats.apply(
            lambda row: (row["Dry_Output"] / row["Fresh_Purchased"]) * 100
            if row["Fresh_Purchased"] > 0 else np.nan,
            axis=1
        )
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        candidate_df = pd.merge(
            cp_stats,
            filtered_global_stats_df,
            on="Buyer",
            how="inner"
        )

        ranking_list = []
        for cp, group in candidate_df.groupby("Collection_Point"):
            sorted_grp = group.sort_values(by="CP_Yield", ascending=False)
            buyers = sorted_grp["Buyer"].tolist()
            ranking_list.append({
                "Collection_Point": cp,
                "Best Buyer for CP": buyers[0] if len(buyers) > 0 else "",
                "Second Best Buyer for CP": buyers[1] if len(buyers) > 1 else "",
                "Third Best Buyer for CP": buyers[2] if len(buyers) > 2 else ""
            })
        ranking_df = pd.DataFrame(ranking_list)

        display_df = pd.merge(candidate_df, ranking_df, on="Collection_Point", how="left")
        for rank in ["Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]:
            display_df[rank] = display_df.apply(
                lambda r: r["Buyer"] if r["Buyer"] == r[rank] else "",
                axis=1
            )

        final_display = (
            display_df[[
                "Collection_Point", "Buyer",
                "Yield three prior harvest(%)", "Juice loss at Kasese(%)",
                "CP_Yield_Display",
                "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"
            ]]
            .drop_duplicates()
            .sort_values(by="Collection_Point")
            .rename(columns={"CP_Yield_Display": "CP Yield(%)"})
        )

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
        # PART 3: Per Date Allocation with Dynamic Reallocation and Fallback
        # -------------------------------
        if schedule_file is not None:
            sched_df = pd.read_excel(schedule_file)
            sched_df.rename(columns={sched_df.columns[0]: "Date", sched_df.columns[3]: "CP"}, inplace=True)
            sched_df = sched_df.dropna(subset=["Date", "CP"]).copy()
            sched_df["Date"] = pd.to_datetime(sched_df["Date"], errors="coerce")
            sched_df = sched_df.dropna(subset=["Date"])

            allocation_results = []
            for dt in sched_df["Date"].unique():
                cp_list = sched_df[sched_df["Date"] == dt]["CP"].unique()
                # candidates by CP sorted
                candidates_by_cp = {}
                for cp in cp_list:
                    df_cp = candidate_df[candidate_df["Collection_Point"] == cp]
                    df_cp = df_cp.sort_values("CP_Yield", ascending=False).drop_duplicates("Buyer")
                    candidates_by_cp[cp] = df_cp.to_dict("records")

                assignment = {cp: [] for cp in cp_list}
                assigned_global = set()

                for round_no in range(3):
                    proposals = {}
                    for cp in cp_list:
                        if len(assignment[cp]) > round_no:
                            continue
                        for cand in candidates_by_cp.get(cp, []):
                            if cand["Buyer"] not in assigned_global:
                                proposals[cp] = (cand["Buyer"], cand["CP_Yield"])
                                break
                        else:
                            proposals[cp] = None

                    # resolve conflicts
                    buyer_proposals = {}
                    for cp, prop in proposals.items():
                        if prop:
                            buyer_proposals.setdefault(prop[0], []).append((cp, prop[1]))
                    for buyer, cps in buyer_proposals.items():
                        if len(cps) > 1:
                            best_cp = max(cps, key=lambda x: x[1])[0]
                            for cp, _ in cps:
                                if cp != best_cp:
                                    proposals[cp] = None

                    for cp in cp_list:
                        if proposals.get(cp):
                            buyer = proposals[cp][0]
                            assignment[cp].append(buyer)
                            assigned_global.add(buyer)

                    # fallback
                    fallback = filtered_global_stats_df[~filtered_global_stats_df["Buyer"].isin(assigned_global)]
                    fallback = fallback.sort_values("Global_Yield", ascending=False)
                    for cp in cp_list:
                        if len(assignment[cp]) <= round_no and not fallback.empty:
                            fb = fallback.iloc[0]["Buyer"]
                            assignment[cp].append(fb)
                            assigned_global.add(fb)
                            fallback = fallback.iloc[1:]

                # record
                for cp in cp_list:
                    picks = assignment[cp]
                    allocation_results.append({
                        "Date": dt.date(),
                        "Collection_Point": cp,
                        "Best Buyer for CP": picks[0] if len(picks) > 0 else "",
                        "Second Best Buyer for CP": picks[1] if len(picks) > 1 else "",
                        "Third Best Buyer for CP": picks[2] if len(picks) > 2 else ""
                    })

            allocation_df = pd.DataFrame(allocation_results)
            allocation_df.sort_values(["Date", "Collection_Point"], inplace=True)
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
