import streamlit as st
import pandas as pd
import numpy as np

def compute_buyer_stats(buyer_df):
    """
    Compute global statistics for a single buyer:
      - Global yield: using last 3 valid harvests
      - Global juice loss: most recent non-null value (×100, 2 decimals)
    """
    valid = buyer_df.dropna(subset=["Fresh_Purchased", "Dry_Output"])
    valid = valid[valid["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
    valid = valid[valid["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
    last_3 = valid.head(3)
    total_fresh_3 = last_3["Fresh_Purchased"].sum()
    total_dry_3 = last_3["Dry_Output"].sum()
    global_yield = (total_dry_3 / total_fresh_3) * 100 if total_fresh_3 > 0 else np.nan

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

    if buyer_file:
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

        # Part 1: global stats
        global_list = []
        for buyer, bdf in df.groupby("Buyer"):
            g_yield, g_juice = compute_buyer_stats(bdf)
            global_list.append({"Buyer": buyer, "Global_Yield": g_yield, "Global_Juice_Loss": g_juice})
        global_df = pd.DataFrame(global_list)

        valid_all = df.dropna(subset=["Fresh_Purchased", "Dry_Output"]).
        valid_all = valid_all[valid_all["Fresh_Purchased"].apply(lambda x: isinstance(x, (int, float)))]
        valid_all = valid_all[valid_all["Dry_Output"].apply(lambda x: isinstance(x, (int, float)))]
        agg_all = valid_all.groupby("Buyer").agg(
            Total_Purchased=("Fresh_Purchased", "sum"),
            Total_Dry_Output=("Dry_Output", "sum")
        ).reset_index()
        agg_all["Overall_Yield"] = np.where(
            agg_all["Total_Purchased"] > 0,
            agg_all["Total_Dry_Output"] / agg_all["Total_Purchased"] * 100,
            np.nan
        )

        global_df = global_df.merge(agg_all, on="Buyer", how="left")
        # format
        global_df["Yield three prior harvest(%)"] = global_df["Global_Yield"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        global_df["Juice loss at Kasese(%)"] = global_df["Global_Juice_Loss"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        global_df["Overall Yield (All)(%)"] = global_df["Overall_Yield"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        global_df["Total Purchased"] = global_df["Total_Purchased"].fillna(0)

        st.subheader("Buyer Global Performance")
        st.dataframe(global_df[["Buyer", "Yield three prior harvest(%)", "Juice loss at Kasese(%)", "Overall Yield (All)(%)", "Total Purchased"]])
        st.download_button("Download Buyer Global Performance CSV", global_df.to_csv(index=False).encode(), "buyer_global_performance.csv", "text/csv")

        # filter
        pool = global_df[(global_df["Global_Yield"] >= 37) & (global_df["Global_Juice_Loss"] <= 20)].copy()

        # Part 2: allocation by CP
        cp = df.groupby(["Collection_Point", "Buyer"]).agg({"Fresh_Purchased": "sum", "Dry_Output": "sum"}).reset_index()
        cp["CP_Yield"] = cp.apply(lambda r: r["Dry_Output"]/r["Fresh_Purchased"]*100 if r["Fresh_Purchased"]>0 else np.nan, axis=1)
        cp["CP Yield(%)"] = cp["CP_Yield"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "")
        cand = cp.merge(pool, on="Buyer")
        ranks=[]
        for c, grp in cand.groupby("Collection_Point"):
            s=grp.sort_values("CP_Yield", ascending=False)["Buyer"].tolist()
            ranks.append({"Collection_Point": c, "Best": s[0] if len(s)>0 else "", "Second": s[1] if len(s)>1 else "", "Third": s[2] if len(s)>2 else ""})
        rank_df=pd.DataFrame(ranks)
        disp = cand.merge(rank_df, on="Collection_Point")
        for col, label in [("Best","Best Buyer for CP"), ("Second","Second Best Buyer for CP"), ("Third","Third Best Buyer for CP")]:
            disp[label] = disp.apply(lambda r: r["Buyer"] if r["Buyer"]==r[col] else "", axis=1)
        final = disp[["Collection_Point","Buyer","Yield three prior harvest(%)","Juice loss at Kasese(%)","CP Yield(%)","Best Buyer for CP","Second Best Buyer for CP","Third Best Buyer for CP"]].drop_duplicates()
        st.subheader("Global Buyer Performance by CP")
        st.dataframe(final)
        st.download_button("Download Global Allocation CSV", final.to_csv(index=False).encode(), "global_allocation.csv", "text/csv")

        # Part 3: per-date allocation
        if schedule_file:
            sd=pd.read_excel(schedule_file)
            sd.rename(columns={sd.columns[0]:"Date", sd.columns[3]:"CP"}, inplace=True)
            sd=sd.dropna(subset=["Date","CP"])
            sd["Date"]=pd.to_datetime(sd["Date"], errors="coerce")
            sd=sd.dropna(subset=["Date"])
            results=[]
            for date in sd["Date"].dt.date.unique():
                cps=sd[sd["Date"].dt.date==date]["CP"].unique()
                cands={cp: cand[cand["Collection_Point")==cp].sort_values("CP_Yield",False).to_dict("records") for cp in cps}
                assign={cp:[] for cp in cps}
                used=set()
                for rnd in range(3):
                    props={}
                    for cp in cps:
                        if len(assign[cp])>rnd: continue
                        props[cp]=next(((c["Buyer"],c["CP_Yield"])for c in cands[cp] if c["Buyer"] not in used),None)
                    # conflict
                    by={} ;
                    for cp,val in props.items():
                        if val: by.setdefault(val[0],[]).append((cp,val[1]))
                    for buyer,prs in by.items():
                        if len(prs)>1:
                            best=min(prs,key=lambda x:-x[1])[0]
                            for cp,_ in prs:
                                if cp!=best: props[cp]=None
                    for cp,val in props.items():
                        if val:
                            assign[cp].append(val[0]); used.add(val[0])
                    fb=pool[~pool["Buyer"].isin(used)].sort_values("Global_Yield",False)
                    for cp in cps:
                        if len(assign[cp])<=rnd and not fb.empty:
                            buy=fb.iloc[0]["Buyer"]
                            assign[cp].append(buy); used.add(buy); fb=fb.iloc[1:]
                for cp in cps:
                    lst=assign[cp]
                    results.append({"Date":date,"Collection_Point":cp,"Best Buyer for CP":lst[0],"Second Best Buyer for CP":(lst[1] if len(lst)>1 else ""),"Third Best Buyer for CP":(lst[2] if len(lst)>2 else "")})
            out=pd.DataFrame(results).sort_values(["Date","Collection_Point"])
            st.subheader("Buyer Allocation according to CP schedule")
            st.dataframe(out)
            st.download_button("Download Per Date Allocation CSV", out.to_csv(index=False).encode(), "per_date_allocation.csv", "text/csv")

if __name__ == "__main__":
    main()
