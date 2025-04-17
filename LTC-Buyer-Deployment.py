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
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"]).copy()
    valid = valid[
        valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float))) &
        valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))
    ]
    # take the first three (most recent) entries
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
    df = df.copy()
    df.columns = (
        df.columns
          .str.replace(r"\s+", " ", regex=True)
          .str.replace('\xa0', ' ')
          .str.strip()
    )

    # map Excel header names to our internal names
    rename_map = {
        "Harvest date": "Harvest_ID",
        "Buyer Name": "Buyer",
        "Collection Point": "Collection_Point",
        "Purchased at CP (KG)": "Fresh_Purchased",
        "PB Dry Output (KG)": "Dry_Output",
        "Losses Kasese %": "Juice_Loss_Kasese",
    }
    to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=to_rename)


def main():
    st.title("LTC Buyer CP Deployment")

    st.markdown("### 1) Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("", type=["xlsx"], key="buyer")

    st.markdown("### 2) Upload CP Schedule Excel")
    schedule_file = st.file_uploader("", type=["xlsx"], key="schedule")

    if buyer_file is None:
        st.info("Please upload the Buyer Performance file to begin.")
        return

    # --- PART 1: Buyer Global Performance ---
    raw = pd.read_excel(buyer_file, header=4)
    df = clean_and_rename_buyer(raw)

    required = ["Buyer", "Collection_Point", "Fresh_Purchased", "Dry_Output", "Juice_Loss_Kasese"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns after cleaning: {missing}")
        st.stop()

    df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
    df.sort_index(ascending=False, inplace=True)

    global_list = []
    for buyer, bdf in df.groupby("Buyer"):
        gy, gj = compute_buyer_stats(bdf)
        global_list.append({"Buyer": buyer, "Global_Yield": gy, "Global_Juice_Loss": gj})
    global_perf = pd.DataFrame(global_list)
    global_perf["Yield three prior harvest(%)"] = global_perf["Global_Yield"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
    )
    global_perf["Juice loss at Kasese(%)"] = global_perf["Global_Juice_Loss"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
    )

    st.subheader("Buyer Global Performance")
    st.dataframe(global_perf[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])

    # --- PART 2: CP Allocation Based on Global Criteria ---
    # filter qualified buyers
    qualified = global_perf[(global_perf["Global_Yield"] >= 36) & (global_perf["Global_Juice_Loss"] <= 20)].copy()

    # CP-level yield
    cp_data = df.groupby(["Collection_Point", "Buyer"]).agg({
        "Fresh_Purchased": "sum",
        "Dry_Output": "sum"
    }).reset_index()
    cp_data["CP_Yield"] = cp_data.apply(
        lambda r: (r["Dry_Output"] / r["Fresh_Purchased"]) * 100 if r["Fresh_Purchased"] > 0 else np.nan,
        axis=1
    )

    pool = cp_data.merge(qualified, on="Buyer", how="inner")

    # ranking per CP
    ranked = []
    for cp, grp in pool.groupby("Collection_Point"):
        sorted_grp = grp.sort_values("CP_Yield", ascending=False)
        ranked.append({
            "Collection_Point": cp,
            "Best Buyer for CP": sorted_grp.iloc[0]["Buyer"] if len(sorted_grp) > 0 else "",
            "Second Best Buyer for CP": sorted_grp.iloc[1]["Buyer"] if len(sorted_grp) > 1 else "",
            "Third Best Buyer for CP": sorted_grp.iloc[2]["Buyer"] if len(sorted_grp) > 2 else ""
        })
    ranking_df = pd.DataFrame(ranked)
    rank_cols = ["Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]

    final = pool.merge(ranking_df, on="Collection_Point", how="left")
    for col in rank_cols:
        final[col] = final.apply(lambda r: r["Buyer"] if r["Buyer"] == r[col] else "", axis=1)

    st.subheader("CP Allocation Rankings")
    st.dataframe(final[["Collection_Point", "Buyer", "CP_Yield"] + rank_cols])

    # --- PART 3: Schedule-Based Allocation ---
    if schedule_file is not None:
        sched = pd.read_excel(schedule_file)
        sched = sched.rename(columns={sched.columns[0]: "Date", sched.columns[3]: "CP"})
        sched["Date"] = pd.to_datetime(sched["Date"], errors="coerce")
        sched = sched.dropna(subset=["Date", "CP"])

        alloc_results = []
        for date in sorted(sched["Date"].dt.date.unique()):
            cps = sched[sched["Date"].dt.date == date]["CP"].unique()
            cp_cands = {cp: [] for cp in cps}
            assigned = set()
            # three rounds
            for rnd in range(3):
                proposals = {}
                for cp in cps:
                    cands = pool[(pool["Collection_Point"] == cp) & (~pool["Buyer"].isin(assigned))]
                    cands = cands.sort_values("CP_Yield", ascending=False)
                    proposals[cp] = cands.iloc[0]["Buyer"] if not cands.empty else None
                # conflict resolution: remove duplicates keeping first
                seen = {}
                for cp, buyer in proposals.items():
                    if buyer:
                        seen.setdefault(buyer, []).append(cp)
                for buyer, cps_conf in seen.items():
                    if len(cps_conf) > 1:
                        # keep highest yield
                        yields = {cp: pool[(pool["Collection_Point"] == cp) & (pool["Buyer"] == buyer)]["CP_Yield"].values[0] for cp in cps_conf}
                        best_cp = max(yields, key=yields.get)
                        for cp in cps_conf:
                            if cp != best_cp:
                                proposals[cp] = None
                # assign
                for cp, buyer in proposals.items():
                    if buyer and buyer not in assigned:
                        cp_cands[cp].append(buyer)
                        assigned.add(buyer)
                # fallback
                fallback = qualified[~qualified["Buyer"].isin(assigned)].sort_values("Global_Yield", ascending=False)
                for cp in cps:
                    if len(cp_cands[cp]) <= rnd and not fallback.empty:
                        fb = fallback.iloc[0]["Buyer"]
                        cp_cands[cp].append(fb)
                        assigned.add(fb)
                        fallback = fallback.iloc[1:]
            # record
            for cp in cps:
                buyers = cp_cands[cp]
                alloc_results.append({
                    "Date": date,
                    "Collection_Point": cp,
                    "Best Buyer": buyers[0] if len(buyers) > 0 else "",
                    "Second Buyer": buyers[1] if len(buyers) > 1 else "",
                    "Third Buyer": buyers[2] if len(buyers) > 2 else ""
                })
        alloc_df = pd.DataFrame(alloc_results)
        st.subheader("Scheduled Allocations")
        st.dataframe(alloc_df)

if __name__ == "__main__":
    main()
