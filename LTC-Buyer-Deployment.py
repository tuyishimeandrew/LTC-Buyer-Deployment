import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df: pd.DataFrame) -> tuple[float, float]:
    """
    Compute buyer-level stats:
      - Global yield (%) from last 3 valid harvests.
      - Latest juice loss (%) at Kasese.
    """
    # Filter valid numeric rows
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]

    # Compute yield
    last_three = valid.head(3)
    total_fresh = last_three["Fresh_Purchased"].sum()
    total_dry = last_three["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    # Compute latest juice loss
    latest_loss = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_loss.empty:
        val = latest_loss["Juice_Loss_Kasese"].iloc[0]
        juice_loss = round(val * 100, 2) if isinstance(val, (int, float)) else np.nan
    else:
        juice_loss = np.nan

    return global_yield, juice_loss


def main():
    st.title("LTC Buyer CP Deployment")

    buyer_file = st.file_uploader("Upload Buyer Performance (xlsx)", type="xlsx")
    schedule_file = st.file_uploader("Upload CP Schedule (xlsx)", type="xlsx")

    if not buyer_file:
        st.info("Please upload the Buyer Performance Excel to proceed.")
        return

    # Read buyer data
    df = pd.read_excel(buyer_file, header=4)
    df.columns = df.columns.str.replace("\n", " ").str.strip()

    # Positional rename: A->Harvest_ID, B->Buyer, D->Collection_Point,
    # E->Fresh_Purchased, H->Juice_Loss_Kasese, P->Dry_Output
    if len(df.columns) < 16:
        st.error(f"Expected at least 16 columns in Buyer file; found {len(df.columns)}.")
        return

    rename_map = {
        df.columns[0]: "Harvest_ID",
        df.columns[1]: "Buyer",
        df.columns[3]: "Collection_Point",
        df.columns[4]: "Fresh_Purchased",
        df.columns[7]: "Juice_Loss_Kasese",
        df.columns[15]: "Dry_Output",
    }
    df.rename(columns=rename_map, inplace=True)

    # Check for required columns
    required_cols = set(rename_map.values())
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.write("Available columns:", df.columns.tolist())
        return

    # Normalize Juice_Loss_Kasese: strip '%', convert to fraction
    loss = df["Juice_Loss_Kasese"].astype(str).str.rstrip('%').replace('', np.nan)
    df["Juice_Loss_Kasese"] = pd.to_numeric(loss, errors="coerce")
    df["Juice_Loss_Kasese"] = df["Juice_Loss_Kasese"].apply(lambda x: x/100 if pd.notnull(x) and abs(x) > 1 else x)

    # Ensure numeric Dry_Output and Fresh_Purchased
    df["Fresh_Purchased"] = pd.to_numeric(df["Fresh_Purchased"], errors="coerce")
    df["Dry_Output"] = pd.to_numeric(df["Dry_Output"], errors="coerce")

    # Sort descending to get latest rows first
    df.sort_index(ascending=False, inplace=True)

    # Part 1: Global stats per buyer
    stats = []
    for buyer, group in df.groupby("Buyer"):
        gy, gj = compute_buyer_stats(group)
        stats.append({"Buyer": buyer, "Global_Yield": gy, "Global_Juice_Loss": gj})
    global_df = pd.DataFrame(stats)
    global_df["Yield three prior harvest(%)"] = global_df["Global_Yield"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    global_df["Juice loss at Kasese(%)"] = global_df["Global_Juice_Loss"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

    st.subheader("Buyer Global Performance")
    st.dataframe(global_df[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
    st.download_button(
        "Download Buyer Global Performance CSV",
        global_df.to_csv(index=False).encode("utf-8"),
        file_name="buyer_global_performance.csv",
        mime="text/csv"
    )

    # Filter qualified buyers
    qualified = global_df[(global_df.Global_Yield >= 36) & (global_df.Global_Juice_Loss <= 20)]

    # Part 2: Allocation by Collection Point
    cp_stats = df.groupby(["Collection_Point", "Buyer"]).agg({
        "Fresh_Purchased": "sum",
        "Dry_Output": "sum"
    }).reset_index()
    cp_stats["CP_Yield"] = np.where(
        cp_stats["Fresh_Purchased"] > 0,
        (cp_stats["Dry_Output"] / cp_stats["Fresh_Purchased"]) * 100,
        np.nan
    )
    cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
    candidates = cp_stats.merge(qualified, on="Buyer", how="inner")

    ranking = []
    for cp, grp in candidates.groupby("Collection_Point"):
        ordered = grp.sort_values(by="CP_Yield", ascending=False)["Buyer"].tolist()
        ranking.append({
            "Collection_Point": cp,
            "Best Buyer for CP": ordered[0] if len(ordered) > 0 else "",
            "Second Best Buyer for CP": ordered[1] if len(ordered) > 1 else "",
            "Third Best Buyer for CP": ordered[2] if len(ordered) > 2 else "",
        })
    rank_df = pd.DataFrame(ranking)
    display_df = candidates.merge(rank_df, on="Collection_Point").drop_duplicates(["Collection_Point", "Buyer"]) 

    display_df = display_df.rename(columns={"CP_Yield_Display": "CP Yield(%)"})
    st.subheader("Global Buyer Performance by CP")
    st.dataframe(display_df[[
        "Collection_Point", "Buyer", "Yield three prior harvest(%)",
        "Juice loss at Kasese(%)", "CP Yield(%)",
        "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"
    ]])
    st.download_button(
        "Download Global Allocation CSV",
        display_df.to_csv(index=False).encode("utf-8"),
        file_name="global_allocation.csv",
        mime="text/csv"
    )

    # Part 3: Per-Date Allocation with dynamic fallback
    if schedule_file:
        sched = pd.read_excel(schedule_file)
        sched.rename(columns={sched.columns[0]: "Date", sched.columns[3]: "CP"}, inplace=True)
        sched = sched[sched.Date.notnull() & sched.CP.notnull()].copy()
        sched["Date"] = pd.to_datetime(sched.Date, errors="coerce")
        sched = sched[sched.Date.notnull()]

        allocation = []
        for dt in sorted(sched.Date.dt.date.unique()):
            cps = sched[sched.Date.dt.date == dt]["CP"].unique()
            # Prepare candidate lists
            cand_by_cp = {
                cp: candidates[candidates.Collection_Point == cp]
                         .sort_values(by="CP_Yield", ascending=False)
                         .to_dict("records")
                for cp in cps
            }
            assign = {cp: [] for cp in cps}
            used = set()
            # Three allocation rounds
            for rnd in range(3):
                # Propose
                proposals = {}
                for cp in cps:
                    if len(assign[cp]) > rnd:
                        continue
                    for c in cand_by_cp.get(cp, []):
                        if c["Buyer"] not in used:
                            proposals[cp] = (c["Buyer"], c["CP_Yield"])
                            break
                    else:
                        proposals[cp] = None
                # Resolve conflicts
                by_buyer = {}
                for cp, prop in proposals.items():
                    if prop:
                        by_buyer.setdefault(prop[0], []).append((cp, prop[1]))
                for buyer, lst in by_buyer.items():
                    if len(lst) > 1:
                        best_cp = max(lst, key=lambda x: x[1])[0]
                        for cp, _ in lst:
                            if cp != best_cp:
                                proposals[cp] = None
                # Assign and fallback
                for cp, prop in proposals.items():
                    if prop:
                        assign[cp].append(prop[0])
                        used.add(prop[0])
                fallback = qualified[~qualified.Buyer.isin(used)].sort_values(by="Global_Yield", ascending=False)
                for cp in cps:
                    if len(assign[cp]) <= rnd and not fallback.empty:
                        fb = fallback.iloc[0].Buyer
                        assign[cp].append(fb)
                        used.add(fb)
                        fallback = fallback.iloc[1:]
            # Record
            for cp in cps:
                sel = assign[cp] + [None, None, None]
                allocation.append({
                    "Date": dt,
                    "Collection_Point": cp,
                    "Best Buyer for CP": sel[0] or "",
                    "Second Best Buyer for CP": sel[1] or "",
                    "Third Best Buyer for CP": sel[2] or "",
                })
        alloc_df = pd.DataFrame(allocation).sort_values(["Date", "Collection_Point"])
        st.subheader("Buyer Allocation by Date and CP")
        st.dataframe(alloc_df)
        st.download_button(
            "Download Per-Date Allocation CSV",
            alloc_df.to_csv(index=False).encode("utf-8"),
            file_name="per_date_allocation.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
