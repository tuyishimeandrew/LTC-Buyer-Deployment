"""LTC Buyer CP Deployment – Streamlit app

Key update: Robust whitespace cleanup is applied to both column names and
string-valued columns (Buyer, Collection Point / CP) as soon as the data is read.
All other logic remains unchanged.
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
WS_RE = re.compile(r"\s+")  # collapse any run of whitespace (tabs, new‑lines, 2+ spaces, etc.)

def clean_series(s: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace, collapse internal whitespace to one space."""
    return (
        s.astype(str)
        .str.replace("\n", " ", regex=False)  # turn hard new‑lines into spaces first
        .str.replace(WS_RE, " ", regex=True)   # collapse tabs / multiple spaces / etc.
        .str.strip()
    )

# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_buyer_stats(buyer_df: pd.DataFrame) -> Tuple[float, float]:
    """Compute global yield & most‑recent juice‑loss for a single buyer."""
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    latest_jl_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_jl_row.empty:
        jl_val = latest_jl_row["Juice_Loss_Kasese"].values[0]
        if pd.notnull(jl_val) and isinstance(jl_val, (int, float)):
            jl_val = round(jl_val * 100, 2)
    else:
        jl_val = np.nan
    return global_yield, jl_val

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.title("LTC Buyer CP Deployment")

    st.markdown("### Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")

    st.markdown("### Upload CP Schedule Excel")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")

    # -------------------------------------------------------------------
    # PART 1 – Buyer Global Performance (All Buyers)
    # -------------------------------------------------------------------
    if buyer_file is not None:
        df = pd.read_excel(buyer_file, header=4)

        # 1️⃣ Advanced whitespace cleanup on column names
        df.columns = (
            df.columns.astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace(WS_RE, " ", regex=True)
            .str.strip()
        )

        # 2️⃣ Column renames
        df.rename(
            columns={
                df.columns[0]: "Harvest_ID",        # Column A
                df.columns[1]: "Buyer",             # Column B
                df.columns[3]: "Collection_Point",  # Column D
                df.columns[4]: "Fresh_Purchased",   # Column E
                df.columns[7]: "Juice_Loss_Kasese", # Column H
                df.columns[15]: "Dry_Output",       # Column P
            },
            inplace=True,
        )

        # 3️⃣ Whitespace cleanup on string‑valued columns
        df[["Buyer", "Collection_Point"]] = df[["Buyer", "Collection_Point"]].apply(clean_series)

        # Ensure numeric conversion for Juice_Loss_Kasese
        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)

        # Compute global stats for all buyers
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            g_yield, g_juice = compute_buyer_stats(bdf)
            global_list.append({
                "Buyer": buyer,
                "Global_Yield": g_yield,
                "Global_Juice_Loss": g_juice,
            })
        global_df = pd.DataFrame(global_list)
        global_df["Yield three prior harvest(%)"] = global_df["Global_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )
        global_df["Juice loss at Kasese(%)"] = global_df["Global_Juice_Loss"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        st.subheader("Buyer Global Performance")
        st.dataframe(global_df[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
        st.download_button(
            "Download Buyer Global Performance CSV",
            global_df.to_csv(index=False).encode("utf‑8"),
            "buyer_global_performance.csv",
            "text/csv",
        )

        # Filter buyers who meet the thresholds
        qualified_df = global_df[
            (global_df["Global_Yield"] >= 36) & (global_df["Global_Juice_Loss"] <= 20)
        ].copy()

        # ----------------------------------------------------------------
        # PART 2 – Allocation by CP (Display)
        # ----------------------------------------------------------------
        cp_stats = (
            df.groupby(["Collection_Point", "Buyer"])
            .agg({"Fresh_Purchased": "sum", "Dry_Output": "sum"})
            .reset_index()
        )
        cp_stats["CP_Yield"] = cp_stats.apply(
            lambda r: (r["Dry_Output"] / r["Fresh_Purchased"]) * 100 if r["Fresh_Purchased"] > 0 else np.nan,
            axis=1,
        )
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        # Join only qualified buyers
        candidate_df = cp_stats.merge(qualified_df, on="Buyer", how="inner")

        # Rank candidates within each CP by CP_Yield
        ranking_rows = []
        for cp, grp in candidate_df.groupby("Collection_Point"):
            sorted_grp = grp.sort_values("CP_Yield", ascending=False)
            ranking_rows.append(
                {
                    "Collection_Point": cp,
                    "Best Buyer for CP": sorted_grp.iat[0, sorted_grp.columns.get_loc("Buyer")] if len(sorted_grp) >= 1 else "",
                    "Second Best Buyer for CP": sorted_grp.iat[1, sorted_grp.columns.get_loc("Buyer")] if len(sorted_grp) >= 2 else "",
                    "Third Best Buyer for CP": sorted_grp.iat[2, sorted_grp.columns.get_loc("Buyer")] if len(sorted_grp) >= 3 else "",
                }
            )
        ranking_df = pd.DataFrame(ranking_rows)

        disp_df = candidate_df.merge(ranking_df, on="Collection_Point", how="left")
        for col in ["Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]:
            disp_df[col] = disp_df.apply(lambda r, c=col: r["Buyer"] if r["Buyer"] == r[c] else "", axis=1)

        final_display = (
            disp_df[
                [
                    "Collection_Point",
                    "Buyer",
                    "Yield three prior harvest(%)",
                    "Juice loss at Kasese(%)",
                    "CP_Yield_Display",
                    "Best Buyer for CP",
                    "Second Best Buyer for CP",
                    "Third Best Buyer for CP",
                ]
            ]
            .drop_duplicates()
            .sort_values("Collection_Point")
            .rename(columns={"CP_Yield_Display": "CP Yield(%)"})
        )

        st.subheader("Global Buyer Performance by CP")
        st.dataframe(final_display)
        st.download_button(
            "Download Global Allocation CSV",
            final_display.to_csv(index=False).encode("utf‑8"),
            "global_allocation.csv",
            "text/csv",
        )

        # ----------------------------------------------------------------
        # PART 3 – Per‑Date Allocation (Dynamic)
        # ----------------------------------------------------------------
        if schedule_file is not None:
            sched_df = pd.read_excel(schedule_file)
            sched_df.rename(columns={sched_df.columns[0]: "Date", sched_df.columns[3]: "CP"}, inplace=True)
            sched_df = sched_df[sched_df["Date"].notnull() & sched_df["CP"].notnull()].copy()
            sched_df["Date"] = pd.to_datetime(sched_df["Date"], errors="coerce")
            sched_df = sched_df[sched_df["Date"].notnull()]

            # Clean CP whitespace too
            sched_df["CP"] = clean_series(sched_df["CP"])

            allocation_rows = []
            for dt in sched_df["Date"].unique():
                cp_list = sched_df.loc[sched_df["Date"] == dt, "CP"].unique()

                # Build candidate lists for each CP (sorted by CP_Yield desc)
                cand_by_cp = {}
                for cp in cp_list:
                    sub = candidate_df[candidate_df["Collection_Point"] == cp]
                    sub = sub.sort_values("CP_Yield", ascending=False).drop_duplicates("Buyer", keep="first")
                    cand_by_cp[cp] = sub.to_dict("records")

                assigned_global: set[str] = set()
                assignment: dict[str, list[str]] = {cp: [] for cp in cp_list}

                # Three rounds (best/second/third)
                for rnd in range(3):
                    proposals: dict[str, Tuple[str, float] | None] = {}
                    for cp in cp_list:
                        if len(assignment[cp]) > rnd:
                            continue  # already have a buyer this round
                        found = None
                        for cand in cand_by_cp.get(cp, []):
                            if cand["Buyer"] not in assigned_global:
                                found = (cand["Buyer"], cand["CP_Yield"])
                                break
                        proposals[cp] = found

                    # Resolve conflicts (same buyer proposed to multiple CPs)
                    buyer_to_cps: dict[str, list[Tuple[str, float]]] = {}
                    for cp, prop in proposals.items():
                        if prop is not None:
                            buyer = prop[0]
                            buyer_to_cps.setdefault(buyer, []).append((cp, prop[1]))
                    for buyer, cps in buyer_to_cps.items():
                        if len(cps) > 1:
                            keep_cp = max(cps, key=lambda x: x[1])[0]
                            for cp, _ in cps:
                                if cp != keep_cp:
                                    proposals[cp] = None

                    # Make the assignments for this round
                    for cp, prop in proposals.items():
                        if prop is not None:
                            buyer, _ = prop
                            assignment[cp].append(buyer)
                            assigned_global.add(buyer)

                    # Fallback from remaining qualified buyers if still blank
                    fallback_pool = qualified_df[~qualified_df["Buyer"].isin(assigned_global)].sort_values(
                        "Global_Yield", ascending=False
                    )
                    for cp in cp_list:
                        if len(assignment[cp]) <= rnd and not fallback_pool.empty:
                            fb_buyer = fallback_pool.iloc[0]["Buyer"]
                            assignment[cp].append(fb_buyer)
                            assigned_global.add(fb_buyer)
                            fallback_pool = fallback_pool.drop(fallback_pool.index[0])

                # Record
                for cp in cp_list:
                    row = {
                        "Date": dt.date(),
                        "Collection_Point": cp,
                        "Best Buyer for CP": assignment[cp][0] if len(assignment[cp]) >= 1 else "",
                        "Second Best Buyer for CP": assignment[cp][1] if len(assignment[cp]) >= 2 else "",
                        "Third Best Buyer for CP": assignment[cp][2] if len(assignment[cp]) >= 3 else "",
                    }
                    allocation_rows.append(row)

            alloc_df = pd.DataFrame(allocation_rows).sort_values(["Date", "Collection_Point"])
            st.subheader("Buyer Allocation according to CP schedule")
            st.dataframe(alloc_df)
            st.download_button(
                "Download Per Date Allocation CSV",
                alloc_df.to_csv(index=False).encode("utf‑8"),
                "per_date_allocation.csv",
                "text/csv",
            )


if __name__ == "__main__":
    main()
