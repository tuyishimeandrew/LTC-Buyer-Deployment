import streamlit as st
import pandas as pd
import numpy as np


def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: computed using the last 3 valid harvests (if both Fresh_Purchased and Dry_Output are numeric)
      - Global juice loss: the most recent non-null value (multiplied by 100 and rounded to 2 decimals)
    """
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    last_3 = valid.head(3)
    total_fresh = last_3["Fresh_Purchased"].sum()
    total_dry = last_3["Dry_Output"].sum()
    global_yield = (total_dry / total_fresh) * 100 if total_fresh > 0 else np.nan

    latest = buyer_df.dropna(subset=["Juice_Loss_Kasese"]).head(1)
    if not latest.empty:
        val = latest["Juice_Loss_Kasese"].values[0]
        juice_loss = round(val * 100, 2) if isinstance(val, (int, float)) and pd.notnull(val) else np.nan
    else:
        juice_loss = np.nan
    return global_yield, juice_loss


def detect_columns(df):
    # Build mapping dynamically based on keywords
    mapping = {}
    for col in df.columns:
        lc = col.lower()
        if 'harvest' in lc and 'date' in lc:
            mapping[col] = 'Harvest_ID'
        elif 'buyer' in lc and 'name' in lc:
            mapping[col] = 'Buyer'
        elif 'collection' in lc and 'point' in lc:
            mapping[col] = 'Collection_Point'
        elif 'purchased' in lc or 'fresh loaded' in lc:
            mapping[col] = 'Fresh_Purchased'
        elif 'kasese' in lc and 'loss' in lc:
            mapping[col] = 'Juice_Loss_Kasese'
        elif 'dry' in lc and 'output' in lc:
            mapping[col] = 'Dry_Output'
    return mapping


def main():
    st.title("LTC Buyer CP Deployment")

    buyer_file = st.file_uploader("Upload Buyer Performance Excel", type=["xlsx"])
    schedule_file = st.file_uploader("Upload CP Schedule Excel", type=["xlsx"])

    if buyer_file:
        df = pd.read_excel(buyer_file, header=4)
        df.columns = df.columns.str.replace('\n', ' ').str.strip()

        col_map = detect_columns(df)
        df.rename(columns=col_map, inplace=True)

        req = {"Harvest_ID", "Buyer", "Collection_Point", "Fresh_Purchased", "Juice_Loss_Kasese", "Dry_Output"}
        missing = req - set(df.columns)
        if missing:
            st.error(f"Missing columns in Buyer file: {missing}\nAvailable columns: {list(df.columns)}")
            return

        df["Juice_Loss_Kasese"] = pd.to_numeric(df["Juice_Loss_Kasese"], errors="coerce")
        df.sort_index(ascending=False, inplace=True)

        # Part 1
        global_list = []
        for buyer, grp in df.groupby("Buyer"):
            gy, gj = compute_buyer_stats(grp)
            global_list.append({"Buyer": buyer, "Global_Yield": gy, "Global_Juice_Loss": gj})
        global_df = pd.DataFrame(global_list)
        global_df["Yield three prior harvest(%)"] = global_df["Global_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        global_df["Juice loss at Kasese(%)"] = global_df["Global_Juice_Loss"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")

        st.subheader("Buyer Global Performance")
        st.dataframe(global_df[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)"]])
        st.download_button("Download Buyer Global Performance CSV", global_df.to_csv(index=False).encode('utf-8'), file_name="buyer_global_performance.csv")

        qualified = global_df[(global_df.Global_Yield >= 36) & (global_df.Global_Juice_Loss <= 20)].copy()

        # Part 2
        cp_stats = df.groupby(["Collection_Point", "Buyer"]).agg({"Fresh_Purchased": "sum", "Dry_Output": "sum"}).reset_index()
        cp_stats["CP_Yield"] = cp_stats.apply(lambda r: (r.Dry_Output / r.Fresh_Purchased) * 100 if r.Fresh_Purchased > 0 else np.nan, axis=1)
        cp_stats["CP_Yield_Display"] = cp_stats["CP_Yield"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        candidates = pd.merge(cp_stats, qualified, on="Buyer", how="inner")

        ranking = []
        for cp, grp in candidates.groupby("Collection_Point"):
            top = grp.sort_values('CP_Yield', ascending=False).Buyer.tolist()
            ranking.append({
                "Collection_Point": cp,
                "Best Buyer for CP": top[0] if top else "",
                "Second Best Buyer for CP": top[1] if len(top) > 1 else "",
                "Third Best Buyer for CP": top[2] if len(top) > 2 else "",
            })
        rank_df = pd.DataFrame(ranking)
        display = pd.merge(candidates, rank_df, on="Collection_Point").drop_duplicates(["Collection_Point","Buyer"]).sort_values("Collection_Point")
        display.rename(columns={"CP_Yield_Display": "CP Yield(%)"}, inplace=True)

        st.subheader("Global Buyer Performance by CP")
        st.dataframe(display[["Collection_Point","Buyer","Yield three prior harvest(%)","Juice loss at Kasese(%)","CP Yield(%)","Best Buyer for CP","Second Best Buyer for CP","Third Best Buyer for CP"]])
        st.download_button("Download Global Allocation CSV", display.to_csv(index=False).encode('utf-8'), file_name="global_allocation.csv")

        # Part 3
        if schedule_file:
            sched = pd.read_excel(schedule_file)
            sched.rename(columns={sched.columns[0]:"Date",sched.columns[3]:"CP"}, inplace=True)
            sched = sched[sched.Date.notnull() & sched.CP.notnull()]
            sched["Date"] = pd.to_datetime(sched.Date, errors='coerce')
            sched = sched[sched.Date.notnull()]

            alloc = []
            for dt in sorted(sched.Date.dt.date.unique()):
                cps = sched[sched.Date.dt.date == dt].CP.unique()
                cbcp = {cp: candidates[candidates.Collection_Point==cp].sort_values('CP_Yield',ascending=False).to_dict('records') for cp in cps}
                assign, used = {cp:[] for cp in cps}, set()
                for rnd in range(3):
                    props = {cp: next(((c['Buyer'],c['CP_Yield']) for c in cbcp.get(cp,[]) if c['Buyer'] not in used), None) for cp in cps}
                    by_buyer = {}
                    for cp, p in props.items():
                        if p: by_buyer.setdefault(p[0],[]).append((cp,p[1]))
                    for buyer, lst in by_buyer.items():
                        if len(lst)>1:
                            best = max(lst, key=lambda x: x[1])[0]
                            for cp,_ in lst:
                                if cp!=best: props[cp]=None
                    for cp,p in props.items():
                        if p:
                            assign[cp].append(p[0]); used.add(p[0])
                    pool = qualified[~qualified.Buyer.isin(used)].sort_values('Global_Yield',ascending=False)
                    for cp in cps:
                        if len(assign[cp])<=rnd and not pool.empty:
                            fb = pool.iloc[0].Buyer; assign[cp].append(fb); used.add(fb); pool=pool.iloc[1:]
                for cp in cps:
                    buyers = assign[cp]+["","",""]
                    alloc.append({"Date":dt,"Collection_Point":cp,
                                  "Best Buyer for CP":buyers[0],"Second Best Buyer for CP":buyers[1],"Third Best Buyer for CP":buyers[2]})
            alloc_df = pd.DataFrame(alloc).sort_values(["Date","Collection_Point"])
            st.subheader("Buyer Allocation according to CP schedule")
            st.dataframe(alloc_df)
            st.download_button("Download Per Date Allocation CSV", alloc_df.to_csv(index=False).encode('utf-8'), file_name="per_date_allocation.csv")

if __name__ == '__main__':
    main()
