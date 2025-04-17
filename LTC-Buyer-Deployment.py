import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: computed using the last 3 valid harvests (if both Fresh_Purchased and Dry_Output are numeric)
      - Global juice loss: the most recent non-null value (multiplied by 100 and rounded to 2 decimals)
    """
    # Filter rows with valid Fresh_Purchased and Dry_Output
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float))) &
                  valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    # Take the last three entries
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    # Most recent non-null Juice_Loss_Kasese
    latest_juice_loss_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_juice_loss_row.empty:
        jl_val = latest_juice_loss_row["Juice_Loss_Kasese"].values[0]
        if isinstance(jl_val, (int, float)):
            juice_loss_val = round(jl_val * 100, 2)
        else:
            juice_loss_val = np.nan
    else:
        juice_loss_val = np.nan

    return global_yield, juice_loss_val


def main():
    st.title("LTC Buyer CP Deployment")

    st.markdown("### 1) Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("", type=["xlsx"], key="buyer")

    st.markdown("### 2) Upload CP Schedule Excel")
    schedule_file = st.file_uploader("", type=["xlsx"], key="schedule")

    if buyer_file is not None:
        # PART 1: Buyer Global Performance
        raw = pd.read_excel(buyer_file, header=4)
        # Clean column names: remove newlines and strip spaces
        raw.columns = raw.columns.str.replace(r"\s+", " ", regex=True).str.replace("\n", " ").str.strip()

        # Rename based on positions matching the Excel layout
        raw.rename(columns={
            raw.columns[0]: "Harvest_ID",         # Column A
            raw.columns[1]: "Buyer",              # Column B
            raw.columns[3]: "Collection_Point",   # Column D
            raw.columns[4]: "Fresh_Purchased",    # Column E
            raw.columns[7]: "Juice_Loss_Kasese",  # Column H
            raw.columns[15]: "Dry_Output"         # Column P
        }, inplace=True)

        # Convert Juice_Loss_Kasese to numeric
        raw["Juice_Loss_Kasese"] = pd.to_numeric(raw["Juice_Loss_Kasese"], errors="coerce")
        raw.sort_index(ascending=False, inplace=True)

        # Compute global stats per buyer
        stats = []
        for buyer, grp in raw.groupby("Buyer"):
            gy, jl = compute_buyer_stats(grp)
            stats.append({"Buyer": buyer, "Global_Yield": gy, "Global_Juice_Loss": jl})
        global_df = pd.DataFrame(stats)
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
            data=global_df.to_csv(index=False).encode('utf-8'),
            file_name="buyer_global_performance.csv",
            mime="text/csv"
        )

        # Filter qualified buyers
        qualified = global_df[(global_df["Global_Yield"] >= 36) &
                              (global_df["Global_Juice_Loss"] <= 20)].copy()

        # PART 2: Allocation by CP (Global Allocation)
        cp_stats = raw.groupby(["Collection_Point", "Buyer"]).agg({
            "Fresh_Purchased": "sum", "Dry_Output": "sum"
        }).reset_index()
        cp_stats["CP_Yield"] = cp_stats.apply(
            lambda r: (r["Dry_Output"] / r["Fresh_Purchased"]) * 100 if r["Fresh_Purchased"] > 0 else np.nan,
            axis=1
        )
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else ""
        )

        candidate_df = cp_stats.merge(
            qualified[["Buyer", "Global_Yield", "Global_Juice_Loss"]],
            on="Buyer", how="inner"
        )

        # Determine top 3 per CP
        rankings = []
        for cp, grp in candidate_df.groupby("Collection_Point"):
            sorted_grp = grp.sort_values("CP_Yield", ascending=False)
            buyers = sorted_grp["Buyer"].tolist()
            rankings.append({
                "Collection_Point": cp,
                "Best Buyer for CP": buyers[0] if len(buyers) > 0 else "",
                "Second Best Buyer for CP": buyers[1] if len(buyers) > 1 else "",
                "Third Best Buyer for CP": buyers[2] if len(buyers) > 2 else ""
            })
        rank_df = pd.DataFrame(rankings)

        display = candidate_df.merge(rank_df, on="Collection_Point", how="left")
        for col in ["Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]:
            display[col] = display.apply(
                lambda r: r["Buyer"] if r["Buyer"] == r[col] else "", axis=1
            )
        final_display = (display[["Collection_Point", "Buyer", "CP_Yield_Display",
                                  "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]]
                         .drop_duplicates().sort_values("Collection_Point")
                         .rename(columns={"CP_Yield_Display": "CP Yield(%)"}))

        st.subheader("Global Buyer Performance by CP")
        st.dataframe(final_display)
        st.download_button(
            "Download Global Allocation CSV",
            data=final_display.to_csv(index=False).encode('utf-8'),
            file_name="global_allocation.csv",
            mime="text/csv"
        )

        # PART 3: Per-Date Allocation
        if schedule_file is not None:
            sched = pd.read_excel(schedule_file)
            sched.columns = sched.columns.str.replace(r"\s+", " ", regex=True).str.strip()
            sched.rename(columns={sched.columns[0]: "Date", sched.columns[3]: "CP"}, inplace=True)
            sched = sched[sched["Date"].notnull() & sched["CP"].notnull()]
            sched["Date"] = pd.to_datetime(sched["Date"], errors="coerce")
            sched = sched[sched["Date"].notnull()]

            allocations = []
            for dt in sched["Date"].unique():
                cp_list = sched[sched["Date"] == dt]["CP"].unique()
                candidates_by_cp = {}
                for cp in cp_list:
                    sub = candidate_df[candidate_df["Collection_Point"] == cp]
                    sub = sub.sort_values("CP_Yield", ascending=False).drop_duplicates("Buyer")
                    candidates_by_cp[cp] = sub.to_dict("records")

                assignment = {cp: [] for cp in cp_list}
                assigned = set()

                for rnd in range(3):
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

                    # Resolve conflicts
                    by_buyer = {}
                    for cp, prop in proposals.items():
                        if prop:
                            by_buyer.setdefault(prop[0], []).append((cp, prop[1]))
                    for buyer, prefs in by_buyer.items():
                        if len(prefs) > 1:
                            chosen = max(prefs, key=lambda x: x[1])[0]
                            for cp, _ in prefs:
                                if cp != chosen:
                                    proposals[cp] = None

                    # Assign proposals
                    for cp, prop in proposals.items():
                        if prop:
                            b, _ = prop
                            assignment[cp].append(b)
                            assigned.add(b)

                    # Fallback
                    fallback = qualified[~qualified["Buyer"].isin(assigned)].sort_values("Global_Yield", ascending=False)
                    for cp in cp_list:
                        if len(assignment[cp]) <= rnd and not fallback.empty:
                            fb = fallback.iloc[0]["Buyer"]
                            assignment[cp].append(fb)
                            assigned.add(fb)
                            fallback = fallback.drop(fallback.index[0])

                for cp in cp_list:
                    lst = assignment[cp]
                    allocations.append({
                        "Date": dt.date(),
                        "Collection_Point": cp,
                        "Best Buyer for CP": lst[0] if len(lst) > 0 else "",
                        "Second Best Buyer for CP": lst[1] if len(lst) > 1 else "",
                        "Third Best Buyer for CP": lst[2] if len(lst) > 2 else ""
                    })

            alloc_df = pd.DataFrame(allocations).sort_values(["Date", "Collection_Point"])
            st.subheader("Buyer Allocation according to CP schedule")
            st.dataframe(alloc_df)
            st.download_button(
                "Download Per-Date Allocation CSV",
                data=alloc_df.to_csv(index=False).encode('utf-8'),
                file_name="per_date_allocation.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
