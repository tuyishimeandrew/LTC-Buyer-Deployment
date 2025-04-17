import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: computed using the last 3 valid harvests (if both Fresh_Purchased and Dry_Output are numeric)
      - Global juice loss: the most recent non-null value (multiplied by 100 and rounded to 2 decimals)
    """
    # Keep only rows with both metrics present and numeric
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    # Latest juice loss (Kasese)
    latest_juice_loss_row = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest_juice_loss_row.empty:
        val = latest_juice_loss_row["Juice_Loss_Kasese"].values[0]
        juice_loss = round(val * 100, 2) if isinstance(val, (int, float)) and pd.notnull(val) else np.nan
    else:
        juice_loss = np.nan

    return global_yield, juice_loss


def main():
    st.title("LTC Buyer CP Deployment")

    st.markdown("### Upload Buyer Performance Excel")
    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"], key="buyer")

    st.markdown("### Upload CP Schedule Excel")
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"], key="schedule")

    if buyer_file is not None:
        # --- Read & clean Buyer data ---
        df = pd.read_excel(buyer_file, header=4)
        df.columns = df.columns.str.replace('\n', ' ').str.strip()

        df.rename(columns={
            "Harvest date": "Harvest_ID",
            "Buyer Name": "Buyer",
            "Collection Point": "Collection_Point",
            "Purchased at CP (KG)": "Fresh_Purchased",
            "Losses Kasese %": "Juice_Loss_Kasese",
            "PB Dry Output (KG)": "Dry_Output",
        }, inplace=True)

        required = {"Harvest_ID", "Buyer", "Collection_Point", "Fresh_Purchased", "Juice_Loss_Kasese", "Dry_Output"}
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing columns in Buyer file: {missing}")
            return

        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)

        # --- Part 1: Global stats per Buyer ---
        global_list = []
        for buyer, group in df.groupby("Buyer"):
            gy, gj = compute_buyer_stats(group)
            global_list.append({"Buyer": buyer, "Global_Yield": gy, "Global_Juice_Loss": gj})
        global_df = pd.DataFrame(global_list)
        global_df["Yield three prior harvest(%)"] = global_df["Global_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        global_df["Juice loss at Kasese(%)"] = global_df["Global_Juice_Loss"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        st.subheader("Buyer Global Performance")
        st.dataframe(global_df[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
        st.download_button("Download Buyer Global Performance CSV",
                           global_df.to_csv(index=False).encode('utf-8'),
                           file_name="buyer_global_performance.csv",
                           mime="text/csv")

        # Filter qualified buyers
        qualified = global_df[(global_df.Global_Yield >= 36) & (global_df.Global_Juice_Loss <= 20)].copy()

        # --- Part 2: Allocation by Collection Point ---
        cp_stats = df.groupby(["Collection_Point", "Buyer"]).agg({"Fresh_Purchased": "sum", "Dry_Output": "sum"}).reset_index()
        cp_stats["CP_Yield"] = cp_stats.apply(lambda r: (r.Dry_Output / r.Fresh_Purchased)*100 if r.Fresh_Purchased>0 else np.nan, axis=1)
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        candidates = pd.merge(cp_stats, qualified, on="Buyer", how="inner")

        ranking = []
        for cp, grp in candidates.groupby("Collection_Point"):
            sorted_grp = grp.sort_values('CP_Yield', ascending=False)
            buyers = sorted_grp.Buyer.tolist() + ["", "", ""]
            ranking.append({
                "Collection_Point": cp,
                "Best Buyer for CP": buyers[0],
                "Second Best Buyer for CP": buyers[1],
                "Third Best Buyer for CP": buyers[2],
            })
        rank_df = pd.DataFrame(ranking)
        display = pd.merge(candidates, rank_df, on="Collection_Point", how="left")
        display = display.drop_duplicates(subset=["Collection_Point", "Buyer"]).sort_values("Collection_Point")
        display.rename(columns={"CP_Yield_Display": "CP Yield(%)"}, inplace=True)

        st.subheader("Global Buyer Performance by CP")
        st.dataframe(display[["Collection_Point", "Buyer", "Yield three prior harvest(%)", 
                              "Juice loss at Kasese(%)", "CP Yield(%)", 
                              "Best Buyer for CP", "Second Best Buyer for CP", "Third Best Buyer for CP"]])
        st.download_button("Download Global Allocation CSV",
                           display.to_csv(index=False).encode('utf-8'),
                           file_name="global_allocation.csv",
                           mime="text/csv")

        # --- Part 3: Per-Date Allocation ---
        if schedule_file is not None:
            sched = pd.read_excel(schedule_file)
            sched.rename(columns={sched.columns[0]: "Date", sched.columns[3]: "CP"}, inplace=True)
            sched = sched[sched.Date.notnull() & sched.CP.notnull()].copy()
            sched["Date"] = pd.to_datetime(sched.Date, errors='coerce')
            sched = sched[sched.Date.notnull()]

            allocation = []
            for date in sorted(sched.Date.dt.date.unique()):
                cps = sched[sched.Date.dt.date==date].CP.unique()
                # candidates by CP
                cand_by_cp = {cp: candidates[candidates.Collection_Point==cp]
                              .sort_values('CP_Yield', ascending=False)
                              .to_dict('records') for cp in cps}
                assign = {cp: [] for cp in cps}
                used = set()
                # three rounds
                for rnd in range(3):
                    # proposals
                    proposals = {}
                    for cp in cps:
                        if len(assign[cp])>rnd: continue
                        for cand in cand_by_cp.get(cp, []):
                            if cand['Buyer'] not in used:
                                proposals[cp] = (cand['Buyer'], cand['CP_Yield'])
                                break
                        else:
                            proposals[cp] = None
                    # resolve conflicts
                    by_buyer = {}
                    for cp, prop in proposals.items():
                        if prop:
                            by_buyer.setdefault(prop[0], []).append((cp, prop[1]))
                    for buyer, plist in by_buyer.items():
                        if len(plist)>1:
                            best_cp = max(plist, key=lambda x: x[1])[0]
                            for cp, _ in plist:
                                if cp!=best_cp: proposals[cp]=None
                    # assign
                    for cp, prop in proposals.items():
                        if prop:
                            assign[cp].append(prop[0]); used.add(prop[0])
                    # fallback
                    pool = qualified[~qualified.Buyer.isin(used)].sort_values('Global_Yield', ascending=False)
                    for cp in cps:
                        if len(assign[cp])<=rnd and not pool.empty:
                            fb = pool.iloc[0].Buyer
                            assign[cp].append(fb); used.add(fb)
                            pool = pool.iloc[1:]
                # record
                for cp in cps:
                    buyers = assign[cp] + ["", "", ""]
                    allocation.append({
                        "Date": date,
                        "Collection_Point": cp,
                        "Best Buyer for CP": buyers[0],
                        "Second Best Buyer for CP": buyers[1],
                        "Third Best Buyer for CP": buyers[2],
                    })
            alloc_df = pd.DataFrame(allocation).sort_values(["Date", "Collection_Point"])
            st.subheader("Buyer Allocation according to CP schedule")
            st.dataframe(alloc_df)
            st.download_button("Download Per Date Allocation CSV",
                               alloc_df.to_csv(index=False).encode('utf-8'),
                               file_name="per_date_allocation.csv",
                               mime="text/csv")

if __name__ == "__main__":
    main()
