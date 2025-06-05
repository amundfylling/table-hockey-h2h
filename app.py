"""
app.py – Table-hockey H2H viewer
────────────────────────────────
• Mobile-first selectors.
• Two stat-cards stay side-by-side on all screen sizes.
• Tables: green winner-cell, 🔗 link, desired columns only, newest-first.
• Light theme forced; theme switcher hidden.
"""

import re, numpy as np, pandas as pd, streamlit as st

# ── PAGE CONFIG & THEME LOCK ────────────────────────────────────────
st.set_page_config(page_title="Table-hockey H2H", layout ="wide")

st.markdown(
    """
    <style>
      /* 1️⃣  KEEP earlier rules …  */
      button[kind="theme"], .stThemeSwitcherPopoverTarget{visibility:hidden;}
      @media (max-width:768px){
        div[data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;}
        div[data-testid="stHorizontalBlock"]>div[data-testid="column"]:nth-child(-n+2){
          flex:0 0 50%!important;max-width:50%!important;min-width:0!important;
        }
      }
      span[data-testid="stMetricValue"]{font-size:clamp(22px,6vw,40px);font-weight:600;}
      div[data-testid="stMetricLabel"]{font-size:clamp(12px,3.5vw,18px);}
      @media (max-width:500px){
        div[data-testid="stMarkdownContainer"] h4{
          font-size:clamp(14px,5vw,18px);margin:0 0 2px 0;
        }
      }

      /* 2️⃣  NEW: enlarge the tab buttons */
      div[data-testid="stTabs"] button{
          font-size:3rem;          /* bigger text            */
          padding:0.6rem 1.4rem;      /* taller / wider target  */
      }
      /* Active tab – make text a bit bolder if you like */
      div[data-testid="stTabs"] button[aria-selected="true"]{
          font-weight:600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── CONSTANTS ────────────────────────────────────────────────────────
DATA_PATH   = "th_matches.parquet"
PLAYER_COLS = ["Player1", "Player2"]
SCORE_COLS  = ["GoalsPlayer1", "GoalsPlayer2"]
HIDE_COLS   = ["StageID", "Player1ID", "Player2ID", "TournamentID", "StageSequence"]
MIN_MATCHES = 50
URL_FMT     = "https://th.sportscorpion.com/eng/tournament/stage/{}/matches/"

ROUNDNUMBER_TO_STAGE = {
    1/64: "1/64 final", 1/32: "1/32 final", 1/16: "1/16 final", 1/8: "1/8 final",
    1/4:  "Quarter-final", 1/2: "Semi-final", 1.0: "Final", 0.9: "3rd-place match",
}

# ── HELPERS ──────────────────────────────────────────────────────────
def has_alpha(x:str)->bool: return bool(re.search(r"[A-Za-zÆØÅæøå]", str(x)))

def orient(row:pd.Series,left:str)->pd.Series:
    if row[PLAYER_COLS[0]]!=left:
        row[PLAYER_COLS[0]],row[PLAYER_COLS[1]]=row[PLAYER_COLS[1]],row[PLAYER_COLS[0]]
        row[SCORE_COLS[0]],row[SCORE_COLS[1]]=row[SCORE_COLS[1]],row[SCORE_COLS[0]]
    return row

def add_url(df):
    if "StageID" in df.columns:
        df["URL"]=df["StageID"].astype(int).map(lambda x:URL_FMT.format(x))
    return df

def prep_table(df,p1,p2):
    df=add_url(df.copy()).rename(columns={SCORE_COLS[0]:p1,SCORE_COLS[1]:p2})
    return df.drop(columns=[c for c in HIDE_COLS+PLAYER_COLS if c in df.columns])\
             .sort_values("Date",ascending=False)

def hl_winner(p1,p2):
    def _s(r):
        css=['']*len(r)
        if r[p1]>r[p2]: css[r.index.get_loc(p1)]='background-color:#c6f6d5'
        elif r[p2]>r[p1]: css[r.index.get_loc(p2)]='background-color:#c6f6d5'
        return css
    return _s

def stats(df,left,right):
    d=df.apply(orient,axis=1,left=left)
    gp=len(d)
    wins=(d[SCORE_COLS[0]]>d[SCORE_COLS[1]]).sum()
    loss=(d[SCORE_COLS[1]]>d[SCORE_COLS[0]]).sum()
    draw=gp-wins-loss
    gf,ga=(d[SCORE_COLS[0]].mean() if gp else 0,
           d[SCORE_COLS[1]].mean() if gp else 0)
    wr= wins/gp*100 if gp else 0
    return {"GP":gp,"W":wins,"L":loss,"D":draw,
            "GF":round(gf,2),"GA":round(ga,2),"WR":round(wr,1)}

# ── LOAD DATA ────────────────────────────────────────────────────────
@st.cache_data
def load(): return pd.read_parquet(DATA_PATH)
df=load()

# ── PLAYER CHOICE UI ────────────────────────────────────────────────
eligible=(pd.concat([df[c] for c in PLAYER_COLS]).dropna()
          .loc[lambda s:s.apply(has_alpha)].value_counts()
          .loc[lambda s:s>=MIN_MATCHES].index.sort_values().tolist())
st.markdown("### Select players")
p1=st.selectbox("Player 1",eligible,placeholder="Select Player 1")
if p1 is None: st.stop()

opps_raw=pd.concat([df.loc[df[PLAYER_COLS[0]]==p1,PLAYER_COLS[1]],
                    df.loc[df[PLAYER_COLS[1]]==p1,PLAYER_COLS[0]]]).dropna()
opponents=sorted(set(opps_raw[opps_raw.apply(has_alpha)])&set(eligible)-{p1})
p2=st.selectbox("Player 2",opponents,placeholder="Select Player 2")
if p2 is None: st.stop()

view=st.radio("",["Round-robin", "Play-off", "Both"],horizontal=True,index=0)



# ── FILTER MATCHES ─────────────────────────────────────────────────-
pair=df[((df[PLAYER_COLS[0]]==p1)&(df[PLAYER_COLS[1]]==p2))|
        ((df[PLAYER_COLS[0]]==p2)&(df[PLAYER_COLS[1]]==p1))].copy()
pair = pair.apply(orient, axis=1, left=p1)
pair["Date"] = pd.to_datetime(pair["Date"], errors="coerce")   # ← NEW

# ── DATE FILTER (two pickers) ──────────────────────────────────────
min_date = pair["Date"].min()
max_date = pair["Date"].max()

col_start, col_end = st.columns(2)
with col_start:
    start = st.date_input(
        "Start date",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date",
    )
with col_end:
    end = st.date_input(
        "End date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date",
    )

# guard: if user flips them, swap so start ≤ end
if start > end:
    st.warning("Start date is after end date — swapping them.")
    start, end = end, start

# apply filter
pair = pair[(pair["Date"] >= pd.to_datetime(start)) &
            (pair["Date"] <= pd.to_datetime(end))]



rr=pair[pair["Stage"].str.contains(r"round[-\s]?robin",case=False,na=False)]
po=pair[~pair["Stage"].str.contains(r"round[-\s]?robin",case=False,na=False)]
if not po.empty and "RoundNumber" in po.columns:
    po["PlayoffStage"]=po["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown")

# ── ENHANCED METRICS ────────────────────────────────────────────────
def player_stats(df_in: pd.DataFrame, pl: str, opp: str) -> dict:
    d = df_in.apply(orient, axis=1, left=pl)

    # ❶ ensure the Date column is datetime
    if pd.api.types.is_object_dtype(d["Date"]):
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        
    gp  = len(d)
    wins = (d[SCORE_COLS[0]] > d[SCORE_COLS[1]]).sum()
    loss = (d[SCORE_COLS[1]] > d[SCORE_COLS[0]]).sum()
    draw = gp - wins - loss

    # overtime subset (requires column 'Overtime' == 'Yes')
    ot = d[d["Overtime"].str.contains("yes", case=False, na=False)] if "Overtime" in d.columns else pd.DataFrame()
    ot_games = len(ot)
    ot_wins  = (ot[SCORE_COLS[0]] > ot[SCORE_COLS[1]]).sum()
    ot_loss  = (ot[SCORE_COLS[0]] < ot[SCORE_COLS[1]]).sum()

    goals_for  = d[SCORE_COLS[0]].sum()
    goals_against = d[SCORE_COLS[1]].sum()
    gd   = goals_for - goals_against
    avg_f = d[SCORE_COLS[0]].mean() if gp else 0
    avg_a = d[SCORE_COLS[1]].mean() if gp else 0

    # biggest win / loss
    # biggest win / loss  (tie-breakers added)
    d["Diff"] = abs(d[SCORE_COLS[0]] - d[SCORE_COLS[1]])
    wins_mask = d[SCORE_COLS[0]] > d[SCORE_COLS[1]]

    biggest_win  = (
        d[wins_mask]
        .sort_values(["Diff", SCORE_COLS[0]], ascending=[False, False])   # ← NEW
        .head(1)
    )

    biggest_loss = (
        d[~wins_mask]
        .sort_values(["Diff", SCORE_COLS[1]], ascending=[False, False])   # ← NEW
        .head(1)
    )

    def fmt_big(row):
        if row.empty:
            return "–"
        score = f"{row.iloc[0][SCORE_COLS[0]]}-{row.iloc[0][SCORE_COLS[1]]}"
        # ❷ safe date-string
        date_val = row.iloc[0]["Date"]
        if pd.isna(date_val):
            date_str = "unknown"
        elif isinstance(date_val, pd.Timestamp):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)[:10]        # fallback for leftovers
        return f"{score} ({date_str})"

    return dict(
        GP=gp,
        Wins=f"{wins} ({wins/gp*100:.1f} %)" if gp else "0",
        Losses=f"{loss} ({loss/gp*100:.1f} %)" if gp else "0",
        Draws=f"{draw} ({draw/gp*100:.1f} %)" if gp else "0",
        OT = f"{ot_games} ({ot_wins}/{ot_loss})" if ot_games else "–",
        GF=goals_for, GA=goals_against, GD=gd,
        AvgF=f"{avg_f:.2f}", AvgA=f"{avg_a:.2f}",
        BigWin=fmt_big(biggest_win), BigLoss=fmt_big(biggest_loss),
    )


# choose the data set for stats
if view == "Round-robin":
    current_df = rr
elif view == "Play-off":
    current_df = po
else:                                 # "Both"
    current_df = pd.concat([rr, po], ignore_index=True)


# ── MAIN CONTENT IN TABS ────────────────────────────────────────────
stats_tab, games_tab = st.tabs(["📊 Stats", "📑 All games"], )

with stats_tab:
    st.divider()  # visual separation from the selectors

    # ---- TOP metric: matches played (identical for both) ---------------
    matches_played = len(current_df)
    st.metric("Matches Played", matches_played if matches_played else "0")

    # ---- SIDE-BY-SIDE player cards -------------------------------------
    left_stats  = player_stats(current_df, p1, p2)
    right_stats = player_stats(current_df, p2, p1)

    c1, c2 = st.columns(2, gap="small", border=True)

    for col, pl, data in [(c1, p1, left_stats), (c2, p2, right_stats)]:
        with col:
            with st.expander(pl, expanded=True):
                st.markdown(f"#### {pl}")
                st.metric("Wins",    data["Wins"])
                st.metric("Losses",  data["Losses"])
                st.metric("Draws",   data["Draws"])
                if view in ("Play-off", "Both"):
                    st.metric("Overtime (Won/Lost)", data["OT"])
                st.metric("Goals Scored",      data["GF"])
                st.metric("Goals Conceded",  data["GA"])
                st.metric("Goal Diff.",     data["GD"])
                st.metric("Avg Goals Scored",  data["AvgF"])
                st.metric("Avg Goals Conceded",  data["AvgA"])
                st.metric("Biggest Win",    data["BigWin"])
                st.metric("Biggest Loss",   data["BigLoss"])


with games_tab:
    # ── TABLES ──────────────────────────────────────────────────────────
    link_cfg = {"URL": st.column_config.LinkColumn(label="", display_text="🔗")}
    sty=hl_winner(p1,p2)

    # helper to show round-robin table (kept DRY)
    def show_rr():
        rr_tbl = (prep_table(rr, p1, p2)
                .drop(columns=[c for c in ["Overtime", "Stage", "PlayoffGameNumber"] if c in rr.columns]))
        if "RoundNumber" in rr_tbl.columns:
            rr_tbl["RoundNumber"] = rr_tbl["RoundNumber"].astype("Int64")
        st.subheader("Round-robin")
        st.dataframe(rr_tbl.style.apply(sty, axis=1),
                    hide_index=True, use_container_width=True,
                    column_config=link_cfg)

    if view=="Play-off":
        # series aggregation
        if not po.empty:
            po["WinnerName"]=np.where(po[SCORE_COLS[0]]>po[SCORE_COLS[1]],p1,
                                    np.where(po[SCORE_COLS[1]]>po[SCORE_COLS[0]],p2,"Tie"))
            series=(po.groupby(["RoundNumber","TournamentName","StageID"])
                    .agg(**{p1:("WinnerName",lambda x:(x==p1).sum()),
                            p2:("WinnerName",lambda x:(x==p2).sum()),
                            "Date":("Date","min")})
                    .reset_index()
                    .assign(PlayoffStage=lambda d:d["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown"))
                    .pipe(add_url)
                    .loc[:,[p1,p2,"PlayoffStage","Date","TournamentName","URL"]]
                    .sort_values("Date",ascending=False))
            st.subheader("Play-off series scores")
            st.dataframe(series.style.apply(sty,axis=1),
                        hide_index=True,use_container_width=True,column_config=link_cfg)

        po_tbl=prep_table(po,p1,p2).drop(columns=[c for c in ["RoundNumber","WinnerName", "Stage"] if c in po.columns])
        if "PlayoffGameNumber" in po_tbl.columns:
            po_tbl["PlayoffGameNumber"]=po_tbl["PlayoffGameNumber"].astype("Int64")
        st.subheader("Play-off individual games")
        st.dataframe(po_tbl.style.apply(sty,axis=1),
                    hide_index=True,use_container_width=True,column_config=link_cfg)
    elif view == "Round-robin":
        show_rr()
    else:  # "Both"  –– render everything
        if not po.empty:
            po["WinnerName"] = np.where(po[SCORE_COLS[0]] > po[SCORE_COLS[1]], p1,
                                        np.where(po[SCORE_COLS[1]] > po[SCORE_COLS[0]], p2, "Tie"))
            series = (po.groupby(["RoundNumber", "TournamentName", "StageID"])
                        .agg(**{p1: ("WinnerName", lambda x: (x == p1).sum()),
                                p2: ("WinnerName", lambda x: (x == p2).sum()),
                                "Date": ("Date", "min")})
                        .reset_index()
                        .assign(PlayoffStage=lambda d: d["RoundNumber"]
                            .map(ROUNDNUMBER_TO_STAGE).fillna("Unknown"))
                        .pipe(add_url)
                        .loc[:, [p1, p2, "PlayoffStage", "Date",
                                "TournamentName", "URL"]]
                        .sort_values("Date", ascending=False))
            st.subheader("Play-off series scores")
            st.dataframe(series.style.apply(sty, axis=1),
                        hide_index=True, use_container_width=True,
                        column_config=link_cfg)

        po_tbl = prep_table(po, p1, p2)\
                .drop(columns=[c for c in ["RoundNumber", "WinnerName", "Stage"] if c in po.columns])
        if "PlayoffGameNumber" in po_tbl.columns:
            po_tbl["PlayoffGameNumber"] = po_tbl["PlayoffGameNumber"].astype("Int64")
        st.subheader("Play-off individual games")
        st.dataframe(po_tbl.style.apply(sty, axis=1),
                    hide_index=True, use_container_width=True,
                    column_config=link_cfg)

        show_rr()
