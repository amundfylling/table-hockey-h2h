"""
app.py — Stiga table-hockey head-to-head viewer
───────────────────────────────────────────────
• Sidebar: Player 1, Player 2, and a Play-off ↔ Round-robin switch.
• Wide layout; URL (🔗) column in all tables.
• Tables shown depend on the sidebar switch.
"""

import re, numpy as np
import pandas as pd
import streamlit as st

# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(page_title="Table-hockey H2H", layout="wide")

# ── CONSTANTS ────────────────────────────────────────────────────────
DATA_PATH   = "th_matches.parquet"
PLAYER_COLS = ["Player1", "Player2"]              # change if different
SCORE_COLS  = ["GoalsPlayer1", "GoalsPlayer2"]    # change if different
DROP_COLS   = ["StageID", "Player1ID", "Player2ID",
               "TournamentID", "StageSequence"]   # always hidden
MIN_MATCHES = 50

ROUNDNUMBER_TO_STAGE = {
    1/64: "1/64 final", 1/32: "1/32 final", 1/16: "1/16 final", 1/8: "1/8 final",
    1/4: "Quarter-final", 1/2: "Semi-final", 1.0: "Final", 0.9: "3rd-place match"
}

URL_FMT = "https://th.sportscorpion.com/eng/tournament/stage/{}/matches/"

# ── HELPERS ──────────────────────────────────────────────────────────
def has_alpha(name: str) -> bool:
    return bool(re.search(r"[A-Za-zÆØÅæøå]", str(name)))

def orient_row(row: pd.Series, left_player: str) -> pd.Series:
    if row[PLAYER_COLS[0]] != left_player:
        row[PLAYER_COLS[0]], row[PLAYER_COLS[1]] = row[PLAYER_COLS[1]], row[PLAYER_COLS[0]]
        if set(SCORE_COLS).issubset(row.index):
            row[SCORE_COLS[0]], row[SCORE_COLS[1]] = row[SCORE_COLS[1]], row[SCORE_COLS[0]]
    return row

def add_url(df: pd.DataFrame) -> pd.DataFrame:
    if "StageID" in df.columns:
        df["URL"] = df["StageID"].astype(int).map(lambda x: URL_FMT.format(x))
    else:
        df["URL"] = ""
    return df

def prep_df(df: pd.DataFrame, p1: str, p2: str) -> pd.DataFrame:
    df = add_url(df.copy())
    if set(SCORE_COLS).issubset(df.columns):
        df = df.rename(columns={SCORE_COLS[0]: p1, SCORE_COLS[1]: p2})
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop(columns=[c for c in PLAYER_COLS if c in df.columns])
    return df

# ── LOAD DATA ────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

df = load_data(DATA_PATH)

# ── BUILD ELIGIBLE PLAYER LIST ───────────────────────────────────────
all_players = pd.concat([df[c] for c in PLAYER_COLS]).dropna()
eligible = (all_players[all_players.apply(has_alpha)]
            .value_counts()
            .loc[lambda s: s >= MIN_MATCHES]
            .index.sort_values()
            .tolist())

if len(eligible) < 2:
    st.error("Not enough eligible players.")
    st.stop()

# ── SIDEBAR ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")

p1 = st.sidebar.selectbox("Player 1", eligible, index=0)

opps = pd.concat([
    df.loc[df[PLAYER_COLS[0]] == p1, PLAYER_COLS[1]],
    df.loc[df[PLAYER_COLS[1]] == p1, PLAYER_COLS[0]]
]).dropna()
opponents = sorted(set(opps[opps.apply(has_alpha)]) & set(eligible) - {p1})

if not opponents:
    st.error(f"No qualifying opponents for {p1}.")
    st.stop()

p2 = st.sidebar.selectbox("Player 2", opponents, index=0)

view = st.sidebar.radio(
    "View",
    ["Round-robin", "Play-off"],
    horizontal=True,
    index=0 
)


# ── FILTER & PREP MATCHES ───────────────────────────────────────────
mask = (((df[PLAYER_COLS[0]] == p1) & (df[PLAYER_COLS[1]] == p2)) |
        ((df[PLAYER_COLS[0]] == p2) & (df[PLAYER_COLS[1]] == p1)))

filt = df.loc[mask].copy().apply(orient_row, axis=1, left_player=p1)

rr_mask  = filt["Stage"].str.contains(r"round[-\s]?robin", case=False, na=False)
round_rb = filt[rr_mask].copy()
playoff  = filt[~rr_mask].copy()

if not playoff.empty and "RoundNumber" in playoff.columns:
    playoff["PlayoffStage"] = playoff["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown")

if not playoff.empty:
    if "Winner" in playoff.columns:
        playoff["WinnerName"] = playoff["Winner"]
    else:
        playoff["WinnerName"] = np.where(playoff[SCORE_COLS[0]] > playoff[SCORE_COLS[1]], p1, p2)

    series = (playoff
              .groupby(["RoundNumber", "TournamentName", "StageID"])
              .agg(**{p1: ("WinnerName", lambda x: (x == p1).sum()),
                      p2: ("WinnerName", lambda x: (x == p2).sum()),
                      "Date": ("Date", "min")})
              .reset_index()
              .assign(PlayoffStage=lambda d: d["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown"))
              .pipe(add_url)
              .loc[:, [p1, p2, "PlayoffStage", "Date", "TournamentName", "URL"]])

# --- Play-off individual games table ---
cols_po  = [p1, p2, "PlayoffStage", "PlayoffGameNumber", "Date", "TournamentName", "URL"]
po_show  = prep_df(playoff, p1, p2)               # processed df (has renamed goal cols)
po_show  = po_show[[c for c in cols_po if c in po_show.columns]]

# --- Round-robin table ---
cols_rr  = [p1, p2, "RoundNumber", "Date", "TournamentName", "URL"]
rr_show  = prep_df(round_rb, p1, p2)
rr_show  = rr_show[[c for c in cols_rr if c in rr_show.columns]]


# ── HEADER & W/L TALLY ──────────────────────────────────────────────
st.markdown(f"## {p1} vs {p2} — {len(filt)} match(es)")

if "Winner" in filt.columns and not filt.empty:
    w1, w2 = (filt["Winner"] == p1).sum(), (filt["Winner"] == p2).sum()
    st.write(f"**Wins:** {p1} {w1} – {w2} {p2}")

link_cfg = {"URL": st.column_config.LinkColumn(label="", display_text="🔗")}

# ── DISPLAY TABLES ──────────────────────────────────────────────────
if view == "Play-off":
    st.subheader("Play-off series scores")
    if playoff.empty:
        st.write("No play-off games between these players.")
    else:
        st.dataframe(series, hide_index=True, use_container_width=True,
                     column_config=link_cfg)

    st.subheader("Play-off individual games")
    if playoff.empty:
        st.write("No play-off games between these players.")
    else:
        st.dataframe(po_show, hide_index=True, use_container_width=True,
                     column_config=link_cfg)

else:  # Round-robin
    st.subheader("Round-robin")
    if round_rb.empty:
        st.write("No round-robin matches between these players.")
    else:
        st.dataframe(rr_show, hide_index=True, use_container_width=True,
                     column_config=link_cfg)
