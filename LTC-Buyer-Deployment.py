import streamlit as st
import pandas as pd
import numpy as np

def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: last 3 valid harvests
      - Global juice loss: most recent non-null
    """
    # only keep rows where both Fresh_Purchased and Dry_Output are numeric
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[
        valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float))) &
        valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))
    ]
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    # most recent non-null Juice_Loss_Kasese
    latest = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest.empty:
        jl = latest["Juice_Loss_Kasese"].values[0]
        juice_loss = round(jl * 100, 2) if isinstance(jl, (int, float)) else np.nan
    else:
        juice_loss = np.nan

    return global_yield, juice_loss

def clean_and_rename_buyer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggressively clean whitespace (incl. NBSP) from headers,
    then rename the key columns by matching their Excel names.
    """
    # collapse any sequence of whitespace into one space, remove NBSP, strip
    df.columns = (
        df.columns
          .str.replace(r'\s+', ' ', regex=True)
          .str.replace('\xa0', ' ')
          .str.strip()
    )

    # map the actual Excel header names to our internal names
    rename_map = {
        "Harvest date": "Harvest_ID",              # if you need it
        "Buyer Name": "Buyer",
        "Collection Point": "Collection_Point",
        "Purchased at CP (KG)": "Fresh_Purchased",
        "PB Dry Output (KG)": "Dry_Output",
        "Losses Kasese %": "Juice_Loss_Kasese",
    }
    # only rename those keys that actually exist in df
    to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=to_rename)
    return df

def main():
    st.title("LTC Buyer CP Deployment")

    st.markdown("### 1) Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("", type=["xlsx"], key="buyer")

    st.markdown("### 2) Upload CP Schedule Excel")
    schedule_file = st.file_uploader("", type=["xlsx"], key="schedule")

    if buyer_file is not None:
        # --- PART 1: Buyer Global Performance ---
        raw = pd.read_excel(buyer_file, header=4)
        df = clean_and_rename_buyer(raw)

        # sanity check: ensure we have the columns we need
        required = ["Buyer", "Collection_Point", "Fresh_Purchased", "Dry_Output", "Juice_Loss_Kasese"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns after cleaning: {missing}")
            st.stop()

        # convert juice loss to numeric (it might have been read as string)
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)

        # compute global stats
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            gy, gj = compute_buyer_stats(bdf)
            global_list.append({
                "Buyer": buyer,
                "Global_Yield": gy,
                "Global_Juice_Loss": gj
            })
        global_performance_all = pd.DataFrame(global_list)
        global_performance_all["Yield three prior harvest(%)"] = global_performance_all["Global_Yield"]\
            .apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        global_performance_all["Juice loss at Kasese(%)"] = global_performance_all["Global_Juice_Loss"]\
            .apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        st.subheader("Buyer Global Performance")
        st.dataframe(
            global_performance_all[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]]
        )
        csv1 = global_performance_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Buyer Global Performance CSV",
            data=csv1,
            file_name="buyer_global_performance.csv",
            mime="text/csv"
        )

        # filter qualified buyers
        qualified = global_performance_all[
            (global_performance_all["Global_Yield"] >= 36) &
            (global_performance_all["Global_Juice_Loss"] <= 20)
        ].copy()

        # --- PART 2: Allocation by CP ---
        cp_stats = df.groupby(["Collection_Point", "Buyer"])\
                     .agg({"Fresh_Purchased": "sum", "Dry_Output": "sum"})\
                     .reset_index()
        cp_stats["CP_Yield"] = cp_stats.apply(
            lambda r: (r["Dry_Output"] / r["Fresh_Purchased"]) * 100
            if r["Fresh_Purchased"] > 0 else np.nan, axis=1
        )
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"]\
            .apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        # only keep qualified buyers
        candidate_df = cp_stats.merge(
            qualified[["Buyer", "Global_Yield", "Global_Juice_Loss"]],
            on="Buyer", how="inner"
        )

        # rank top 3 per CP
        rankings = []
        for cp, grp in candidate_df.groupby("Collection_Point"):
            top = grp.sort_values("CP_Yield", ascending=False)["Buyer"].tolist()
            rankings.append({
                "Collection_Point": cp,
                "Best Buyer for CP": top[0] if len(top) > 0 else "",
                "Second Best Buyer for CP": top[1] if len(top) > 1 else "",
                "Third Best Buyer for CP": top[2] if len(top) > 2 else "",
            })
        rank_df = pd.DataFrame(rankings)

        display = candidate_df.merge(rank_df, on="Collection_Point", how="left")
        # only keep the matching buyer in each ranking column
        for col in ["Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]:
            display[col] = display.apply(
                lambda r: r["Buyer"] if r["Buyer"] == r[col] else "", axis=1
            )

        final_display = (
            display[[
                "Collection_Point", "Buyer",
                "CP_Yield_Display", "Best Buyer for CP",
                "Second Best Buyer for CP", "Third Best Buyer for CP"
            ]]
            .drop_duplicates()
            .sort_values("Collection_Point")
            .rename(columns={"CP_Yield_Display": "CP Yield(%)"})
        )

        st.subheader("Global Buyer Performance by CP")
        st.dataframe(final_display)
        csv2 = final_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Global Allocation CSV",
            data=csv2,
            file_name="global_allocation.csv",
            mime="text/csv"
        )

        # --- PART 3: Per-Date Allocation ---
        if schedule_file is not None:
            sched_df = pd.read_excel(schedule_file)
            # clean schedule headers similarly
            sched_df.columns = (
                sched_df.columns
                  .str.replace(r'\s+', ' ', regex=True)
                  .str.replace('\xa0', ' ')
                  .str.strip()
            )
            # rename by position or by known names
            sched_df = sched_df.rename(columns={
                sched_df.columns[0]: "Date",
                sched_df.columns[3]: "CP"
            })
            sched_df = sched_df[sched_df["Date"].notnull() & sched_df["CP"].notnull()]
            sched_df["Date"] = pd.to_datetime(sched_df["Date"], errors="coerce")
            sched_df = sched_df[sched_df["Date"].notnull()]

            allocation_results = []
            for dt in sched_df["Date"].unique():
                cp_list = sched_df[sched_df["Date"] == dt]["CP"].unique()
                # build candidates by CP
                candidates_by_cp = {}
                for cp in cp_list:
                    sub = candidate_df[candidate_df["Collection_Point"] == cp]
                    sub = sub.sort_values("CP_Yield", ascending=False).drop_duplicates("Buyer")
                    candidates_by_cp[cp] = sub.to_dict("records")

                assignment = {cp: [] for cp in cp_list}
                assigned = set()

                # three rounds
                for rnd in range(3):
                    # proposals
                    proposals = {}
                    for cp in cp_list:
                        if len(assignment[cp]) > rnd:
                            continue
                        for cand in candidates_by_cp.get(cp, []):
                            if cand["Buyer"] not in assigned:
                                proposals[cp] = (cand["Buyer"], cand["CP_Yield"])
                                break
                        else:
                            proposals[cp] = None

                    # resolve conflicts
                    by_buyer = {}
                    for cp, prop in proposals.items():
                        if prop:
                            buyer, yld = prop
                            by_buyer.setdefault(buyer, []).append((cp, yld))
                    for buyer, prefs in by_buyer.items():
                        if len(prefs) > 1:
                            chosen = max(prefs, key=lambda x: x[1])[0]
                            for cp, _ in prefs:
                                if cp != chosen:
                                    proposals[cp] = None

                    # assign
                    for cp, prop in proposals.items():
                        if prop:
                            b, _ = prop
                            assignment[cp].append(b)
                            assigned.add(b)

                    # fallback
                    fallback = qualified[~qualified["Buyer"].isin(assigned)]
                    fallback = fallback.sort_values("Global_Yield", ascending=False)
                    for cp in cp_list:
                        if len(assignment[cp]) <= rnd and not fallback.empty:
                            fb = fallback.iloc[0]["Buyer"]
                            assignment[cp].append(fb)
                            assigned.add(fb)
                            fallback = fallback.drop(fallback.index[0])

                for cp in cp_list:
                    lst = assignment[cp]
                    allocation_results.append({
                        "Date": dt.date(),
                        "Collection_Point": cp,
                        "Best Buyer for CP": lst[0] if len(lst) > 0 else "",
                        "Second Best Buyer for CP": lst[1] if len(lst) > 1 else "",
                        "Third Best Buyer for CP": lst[2] if len(lst) > 2 else ""
                    })

            alloc_df = pd.DataFrame(allocation_results)
            alloc_df = alloc_df.sort_values(["Date", "Collection_Point"])

            st.subheader("Buyer Allocation according to CP schedule")
            st.dataframe(alloc_df)
            csv3 = alloc_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Per-Date Allocation CSV",
                data=csv3,
                file_name="per_date_allocation.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
