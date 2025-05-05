import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: using last 3 valid harvests
      - Global juice loss: most recent non-null value (×100, 2 decimals)
    """
    # Filter rows with numeric Fresh_Purchased and Dry_Output
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    # Take the last 3 harvests
    last_3 = valid.head(3)
    total_fresh_3 = last_3["Fresh_Purchased"].sum()
    total_dry_3 = last_3["Dry_Output"].sum()
    global_yield = (total_dry_3 / total_fresh_3) * 100 if total_fresh_3 > 0 else np.nan

    # Most recent juice loss
    latest_loss = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_loss.empty:
        jl = latest_loss["Juice_Loss_Kasese"].iloc[0]
        juice_loss_val = round(jl * 100, 2) if isinstance(jl, (int, float)) else np.nan
    else:
        juice_loss_val = np.nan

    return global_yield, juice_loss_val


def main():
    st.title("LTC Buyer CP Deployment")

    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")

    if not buyer_file:
        return

    # === Read and inspect columns ===
    df = pd.read_excel(buyer_file, header=4)
    st.write("### Columns before renaming:", df.columns.to_list())

    # === Rename columns ===
    df.rename(columns={
        df.columns[0]: "Harvest_ID",
        df.columns[1]: "Buyer",
        df.columns[3]: "Collection_Point",
        df.columns[4]: "Fresh_Purchased",
        # by index: rename the 16th column to Dry_Output
        df.columns[15]: "Dry_Output",
        df.columns[7]: "Juice_Loss_Kasese",
    }, inplace=True)

    st.write("### Columns after renaming:", df.columns.to_list())

    # Coerce juice loss to numeric
    df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")

    # Reverse chronological sort
    df.sort_index(ascending=False, inplace=True)

    # --- Part 1: Buyer Global Performance ---
    global_stats = []
    for buyer, bdf in df.groupby("Buyer"):
        g_yield, g_juice = compute_buyer_stats(bdf)
        global_stats.append({
            "Buyer": buyer,
            "Global_Yield": g_yield,
            "Global_Juice_Loss": g_juice
        })
    global_df = pd.DataFrame(global_stats)

    # Overall stats (all valid rows)
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
        (agg_all["Total_Dry_Output"] / agg_all["Total_Purchased"]) * 100,
        np.nan
    )

    # Merge and format
    perf_df = global_df.merge(agg_all, on="Buyer", how="left")
    perf_df["Yield three prior harvest(%)"] = perf_df["Global_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    perf_df["Juice loss at Kasese(%)"]   = perf_df["Global_Juice_Loss"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    perf_df["Overall Yield (All)(%)"]    = perf_df["Overall_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    perf_df["Total Purchased"]           = perf_df["Total_Purchased"].fillna(0)

    st.subheader("Buyer Global Performance")
    st.dataframe(perf_df[[
        "Buyer",
        "Yield three prior harvest(%)",
        "Juice loss at Kasese(%)",
        "Overall Yield (All)(%)",
        "Total Purchased"
    ]])
    st.download_button(
        label="Download Buyer Global Performance CSV",
        data=perf_df.to_csv(index=False).encode("utf-8"),
        file_name="buyer_global_performance.csv",
        mime="text/csv"
    )

    # Filter for qualified buyers
    qualified = perf_df[
        (perf_df["Global_Yield"] >= 37) &
        (perf_df["Global_Juice_Loss"] <= 20)
    ].copy()

    # --- Part 2: Allocation by CP ---
    cp_stats = (
        df.groupby(["Collection_Point", "Buyer"])
          .agg({"Fresh_Purchased": "sum", "Dry_Output": "sum"})
          .reset_index()
    )
    cp_stats["CP_Yield"] = cp_stats.apply(
        lambda r: (r["Dry_Output"] / r["Fresh_Purchased"]) * 100 if r["Fresh_Purchased"] > 0 else np.nan,
        axis=1
    )
    cp_stats["CP Yield(%)"] = cp_stats["CP_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    candidates = cp_stats.merge(qualified, on="Buyer", how="inner")

    # Top-3 per CP
    rankings = []
    for cp, grp in candidates.groupby("Collection_Point"):
        sorted_grp = grp.sort_values(by="CP_Yield", ascending=False)
        buyers = sorted_grp["Buyer"].tolist()
        rankings.append({
            "Collection_Point": cp,
            "Best Buyer for CP": buyers[0] if len(buyers) > 0 else "",
            "Second Best Buyer for CP": buyers[1] if len(buyers) > 1 else "",
            "Third Best Buyer for CP": buyers[2] if len(buyers) > 2 else ""
        })
    rank_df = pd.DataFrame(rankings)

    display = candidates.merge(rank_df, on="Collection_Point")
    for col in ["Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]:
        display[col] = display.apply(lambda r: r["Buyer"] if r["Buyer"] == r[col] else "", axis=1)

    final_display = (
        display[[
            "Collection_Point", "Buyer",
            "Yield three prior harvest(%)", "Juice loss at Kasese(%)", "CP Yield(%)",
            "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"
        ]]
        .drop_duplicates()
        .sort_values(by="Collection_Point")
    )

    st.subheader("Global Buyer Performance by CP")
    st.dataframe(final_display)
    st.download_button(
        label="Download Global Allocation CSV",
        data=final_display.to_csv(index=False).encode("utf-8"),
        file_name="global_allocation.csv",
        mime="text/csv"
    )

    # --- Part 3: Per-Date Allocation ---
    if schedule_file:
        sched = pd.read_excel(schedule_file)
        sched.rename(columns={
            sched.columns[0]: "Date",
            sched.columns[3]: "CP"
        }, inplace=True)
        sched = sched.dropna(subset=["Date", "CP"]).copy()
        sched["Date"] = pd.to_datetime(sched["Date"], errors="coerce")
        sched = sched.dropna(subset=["Date"])

        allocations = []
        for dt in sched["Date"].dt.date.unique():
            cps = sched[sched["Date"].dt.date == dt]["CP"].unique()
            # build candidate pools by CP
            pool_by_cp = {}
            for cp in cps:
                df_cp = candidates[candidates["Collection_Point"] == cp]
                df_cp = df_cp.sort_values(by="CP_Yield", ascending=False).drop_duplicates(subset="Buyer")
                pool_by_cp[cp] = df_cp.to_dict("records")

            assignment = {cp: [] for cp in cps}
            used_buyers = set()

            # three rounds of picks
            for round_no in range(3):
                props = {}
                for cp in cps:
                    if len(assignment[cp]) > round_no:
                        continue
                    props[cp] = next(
                        (c for c in pool_by_cp[cp] if c["Buyer"] not in used_buyers),
                        None
                    )

                # resolve conflicts (if same buyer is proposed to multiple CPs)
                proposals = {}
                for cp, c in props.items():
                    if c:
                        proposals.setdefault(c["Buyer"], []).append((cp, c["CP_Yield"]))
                for buyer, reps in proposals.items():
                    if len(reps) > 1:
                        # keep the CP where they have the highest yield
                        best_cp = max(reps, key=lambda x: x[1])[0]
                        for cp, _ in reps:
                            if cp != best_cp:
                                props[cp] = None

                # assign picks
                for cp, c in props.items():
                    if c:
                        assignment[cp].append(c["Buyer"])
                        used_buyers.add(c["Buyer"])

                # fallback for any CP still missing a pick
                fallback = qualified[~qualified["Buyer"].isin(used_buyers)].sort_values(by="Global_Yield", ascending=False)
                for cp in cps:
                    if len(assignment[cp]) <= round_no and not fallback.empty:
                        fb = fallback.iloc[0]["Buyer"]
                        assignment[cp].append(fb)
                        used_buyers.add(fb)
                        fallback = fallback.iloc[1:]

            # record all three picks per CP
            for cp in cps:
                picks = assignment[cp]
                allocations.append({
                    "Date": dt,
                    "Collection_Point": cp,
                    "Best Buyer for CP": picks[0] if len(picks) > 0 else "",
                    "Second Best Buyer for CP": picks[1] if len(picks) > 1 else "",
                    "Third Best Buyer for CP": picks[2] if len(picks) > 2 else ""
                })

        out_df = pd.DataFrame(allocations).sort_values(by=["Date", "Collection_Point"])
        st.subheader("Buyer Allocation according to CP schedule")
        st.dataframe(out_df)
        st.download_button(
            label="Download Per Date Allocation CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name="per_date_allocation.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
