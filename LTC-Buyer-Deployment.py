import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: last 3 valid harvests
      - Global juice loss: most recent non-null
    """
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"]).copy()
    valid = valid[
        valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float))) &
        valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))
    ]
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    latest = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest.empty:
        jl = latest["Juice_Loss_Kasese"].values[0]
        juice_loss = round(jl * 100, 2) if isinstance(jl, (int, float)) else np.nan
    else:
        juice_loss = np.nan

    return global_yield, juice_loss


def clean_and_rename(df):
    # Collapse all whitespace (including non-breaking spaces), then strip
    df.columns = (
        df.columns
          .str.replace(r"\s+", " ", regex=True)
          .str.replace("\xa0", " ")
          .str.strip()
    )

    # Map actual Excel headers to internal names
    rename_map = {
        "Harvest ID": "Harvest_ID",
        "Harvest date": "Harvest_ID",
        "Buyer Name": "Buyer",
        "Buyer": "Buyer",
        "Collection Point": "Collection_Point",
        "Fresh Purchased": "Fresh_Purchased",
        "Juice Loss Kasese": "Juice_Loss_Kasese",
        "Dry Output": "Dry_Output",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df


def main():
    st.title("LTC Buyer CP Deployment")

    st.markdown("### Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")

    st.markdown("### Upload CP Schedule Excel")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")

    if buyer_file is not None:
        # --------------------------------
        # PART 1: Buyer Global Performance
        # --------------------------------
        df = pd.read_excel(buyer_file, header=4)
        df = clean_and_rename(df)

        # Sanity check for required columns
        required = ["Harvest_ID", "Buyer", "Collection_Point", "Fresh_Purchased", "Dry_Output", "Juice_Loss_Kasese"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns after cleaning: {missing}")
            st.stop()

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
        global_performance_all = pd.DataFrame(global_list)
        global_performance_all["Yield three prior harvest(%)"] = (
            global_performance_all["Global_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        )
        global_performance_all["Juice loss at Kasese(%)"] = (
            global_performance_all["Global_Juice_Loss"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
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

        # Filter qualified buyers
        filtered_global_stats_df = global_performance_all[
            (global_performance_all["Global_Yield"] >= 36) &
            (global_performance_all["Global_Juice_Loss"] <= 20)
        ].copy()

        # --------------------------------
        # PART 2: Allocation by CP (Display)
        # --------------------------------
        cp_stats = df.groupby(["Collection_Point", "Buyer"]).agg({
            "Fresh_Purchased": "sum",
            "Dry_Output": "sum"
        }).reset_index()
        cp_stats["CP_Yield"] = cp_stats.apply(
            lambda row: (row["Dry_Output"] / row["Fresh_Purchased"]) * 100 if row["Fresh_Purchased"] > 0 else np.nan,
            axis=1
        )
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        candidate_df = pd.merge(cp_stats, filtered_global_stats_df, on="Buyer", how="inner")

        # Rank candidates per CP
        ranking_list = []
        for cp, group in candidate_df.groupby("Collection_Point"):
            sorted_grp = group.sort_values(by="CP_Yield", ascending=False)
            ranking_list.append({
                "Collection_Point": cp,
                "Best Buyer for CP": sorted_grp.iloc[0]["Buyer"] if len(sorted_grp) > 0 else "",
                "Second Best Buyer for CP": sorted_grp.iloc[1]["Buyer"] if len(sorted_grp) > 1 else "",
                "Third Best Buyer for CP": sorted_grp.iloc[2]["Buyer"] if len(sorted_grp) > 2 else ""
            })
        ranking_df = pd.DataFrame(ranking_list)

        display_df = pd.merge(candidate_df, ranking_df, on="Collection_Point", how="left")
        display_df["Best Buyer for CP"] = display_df.apply(
            lambda r: r["Buyer"] if r["Buyer"] == r["Best Buyer for CP"] else "", axis=1
        )
        display_df["Second Best Buyer for CP"] = display_df.apply(
            lambda r: r["Buyer"] if r["Buyer"] == r["Second Best Buyer for CP"] else "", axis=1
        )
        display_df["Third Best Buyer for CP"] = display_df.apply(
            lambda r: r["Buyer"] if r["Buyer"] == r["Third Best Buyer for CP"] else "", axis=1
        )

        final_display = display_df[[
            "Collection_Point", "Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)",
            "CP_Yield_Display", "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"
        ]].drop_duplicates().sort_values("Collection_Point")
        final_display.rename(columns={"CP_Yield_Display": "CP Yield(%)"}, inplace=True)

        st.subheader("Global Buyer Performance by CP")
        st.dataframe(final_display)
        csv_global = final_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Global Allocation CSV",
            data=csv_global,
            file_name="global_allocation.csv",
            mime="text/csv",
        )

        # --------------------------------
        # PART 3: Per-Date Allocation
        # --------------------------------
        if schedule_file is not None:
            sched_df = pd.read_excel(schedule_file)
            sched_df.rename(columns={
                sched_df.columns[0]: "Date",
                sched_df.columns[3]: "CP"
            }, inplace=True)
            sched_df = sched_df[sched_df["Date"].notnull() & sched_df["CP"].notnull()].copy()
            sched_df["Date"] = pd.to_datetime(sched_df["Date"], errors="coerce")
            sched_df.dropna(subset=["Date"], inplace=True)

            allocation_results = []
            for dt in sched_df["Date"].dt.date.unique():
                cps = sched_df[sched_df["Date"].dt.date == dt]["CP"].unique()

                # build candidate lists
                candidates_by_cp = {}
                for cp in cps:
                    sub = candidate_df[candidate_df["Collection_Point"] == cp]
                    sub_sorted = sub.sort_values(by="CP_Yield", ascending=False).drop_duplicates("Buyer")
                    candidates_by_cp[cp] = sub_sorted.to_dict("records")

                assignment = {cp: [] for cp in cps}
                assigned = set()

                # three rounds of selection
                for round_no in range(3):
                    proposals = {}
                    for cp in cps:
                        if len(assignment[cp]) > round_no:
                            continue
                        for cand in candidates_by_cp.get(cp, []):
                            if cand["Buyer"] not in assigned:
                                proposals[cp] = (cand["Buyer"], cand["CP_Yield"])
                                break
                        else:
                            proposals[cp] = None

                    # resolve conflicts
                    buyer_props = {}
                    for cp, prop in proposals.items():
                        if prop:
                            buyer_props.setdefault(prop[0], []).append((cp, prop[1]))
                    for buyer, lst in buyer_props.items():
                        if len(lst) > 1:
                            best_cp = max(lst, key=lambda x: x[1])[0]
                            for cp, _ in lst:
                                if cp != best_cp:
                                    proposals[cp] = None

                    # assign
                    for cp, prop in proposals.items():
                        if prop:
                            assignment[cp].append(prop[0])
                            assigned.add(prop[0])

                    # fallback
                    fallback = filtered_global_stats_df[~filtered_global_stats_df["Buyer"].isin(assigned)]
                    fallback = fallback.sort_values("Global_Yield", ascending=False)
                    for cp in cps:
                        if len(assignment[cp]) <= round_no and not fallback.empty:
                            fb = fallback.iloc[0]["Buyer"]
                            assignment[cp].append(fb)
                            assigned.add(fb)
                            fallback = fallback.iloc[1:]

                for cp in cps:
                    b1 = assignment[cp][0] if len(assignment[cp]) > 0 else ""
                    b2 = assignment[cp][1] if len(assignment[cp]) > 1 else ""
                    b3 = assignment[cp][2] if len(assignment[cp]) > 2 else ""
                    allocation_results.append({
                        "Date": dt,
                        "Collection_Point": cp,
                        "Best Buyer for CP": b1,
                        "Second Best Buyer for CP": b2,
                        "Third Best Buyer for CP": b3
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
