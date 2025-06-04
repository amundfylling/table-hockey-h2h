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
st.set_page_config(page_title="Table-hockey H2H", layout="wide")

st.markdown(
    """
    <style>
      /* hide the theme-switcher (desktop + mobile) */
      button[kind="theme"], .stThemeSwitcherPopoverTarget {visibility:hidden;}

      /* ── keep exactly TWO metric columns side-by-side on narrow screens ── */
      @media (max-width: 768px){
          div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(-n+2){
              /*  take up half of the row, prevent wrapping  */
              flex: 0 0 50% !important;           
              max-width: 50% !important;
          }
          /* prevent the block itself from wrapping */
          div[data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;}
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

view=st.radio("",["Round-robin","Play-off"],horizontal=True,index=0)

# ── FILTER MATCHES ─────────────────────────────────────────────────-
pair=df[((df[PLAYER_COLS[0]]==p1)&(df[PLAYER_COLS[1]]==p2))|
        ((df[PLAYER_COLS[0]]==p2)&(df[PLAYER_COLS[1]]==p1))].copy()
pair=pair.apply(orient,axis=1,left=p1)

rr=pair[pair["Stage"].str.contains(r"round[-\s]?robin",case=False,na=False)]
po=pair[~pair["Stage"].str.contains(r"round[-\s]?robin",case=False,na=False)]
if not po.empty and "RoundNumber" in po.columns:
    po["PlayoffStage"]=po["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown")

# ── STAT CARDS ──────────────────────────────────────────────────────
current=rr if view=="Round-robin" else po
c1,c2=st.columns(2,gap="large")
for col,pl,st_dict in [(c1,p1,stats(current,p1,p2)),
                       (c2,p2,stats(current,p2,p1))]:
    with col:
        st.markdown(f"#### {pl}")
        st.metric("Games Played",f"{st_dict['GP']:,}")
        st.metric("Wins",f"{st_dict['W']:,}")
        st.metric("Losses",f"{st_dict['L']:,}")
        st.metric("Win Rate",f"{st_dict['WR']} %")
        st.metric("Avg Goals For",f"{st_dict['GF']:.2f}")
        st.metric("Avg Goals Against",f"{st_dict['GA']:.2f}")

# ── TABLES ──────────────────────────────────────────────────────────
link_cfg={"URL":st.column_config.LinkColumn(label="",display_text="🔗")}
sty=hl_winner(p1,p2)

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
else:

    # ── Round-robin table ───────────────────────────────────────────────
    rr_tbl = (prep_table(rr, p1, p2)
            .drop(columns=[c for c in ["Overtime", "Stage", "PlayoffGameNumber"] if c in rr.columns]))

    if "RoundNumber" in rr_tbl.columns:            # ← new: make it an int
        rr_tbl["RoundNumber"] = rr_tbl["RoundNumber"].astype("Int64")

    st.subheader("Round-robin")
    st.dataframe(rr_tbl.style.apply(sty, axis=1),
                hide_index=True, use_container_width=True,
                column_config=link_cfg)
