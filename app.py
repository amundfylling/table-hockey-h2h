"""
app.py — Stiga table-hockey head-to-head viewer
───────────────────────────────────────────────
• Sidebar: Player 1, Player 2, and view switch (default = Round-robin).
• Tables show a 🔗 link plus cell-highlighting: the higher score in each row
  is shaded green.
• Wide layout.
"""

import re, numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pandas import IndexSlice



# ── PAGE CONFIG ──────────────────────────────────────────────────────
st.set_page_config(page_title="Table-hockey H2H", layout="wide")

# ── CONSTANTS ────────────────────────────────────────────────────────
DATA_PATH   = "th_matches.parquet"
PLAYER_COLS = ["Player1", "Player2"]
SCORE_COLS  = ["GoalsPlayer1", "GoalsPlayer2"]
DROP_COLS   = ["StageID", "Player1ID", "Player2ID", "TournamentID", "StageSequence"]
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
    """Rename goal cols → player names, drop IDs & name cols, add URL."""
    df = add_url(df.copy())
    if set(SCORE_COLS).issubset(df.columns):
        df = df.rename(columns={SCORE_COLS[0]: p1, SCORE_COLS[1]: p2})
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.drop(columns=[c for c in PLAYER_COLS if c in df.columns])
    return df

def winner_style(p1: str, p2: str):
    """Return a row-wise Styler function that highlights the higher value."""
    def _style(row):
        styles = [''] * len(row)
        try:
            a, b = float(row[p1]), float(row[p2])
        except Exception:
            return styles
        if a > b:
            styles[row.index.get_loc(p1)] = 'background-color: #c6f6d5'
        elif b > a:
            styles[row.index.get_loc(p2)] = 'background-color: #c6f6d5'
        return styles
    return _style

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

# ── FILTER PANEL (replace the whole sidebar section) ─────────────────
st.markdown("### Select players")

# ①  Player-1 selector with a blank placeholder
p1_options = ["— Select Player 1 —"] + eligible
p1 = st.selectbox("Player 1", p1_options, index=0, key="p1")

# If the user hasn’t picked anyone yet, stop here
if p1 == "— Select Player 1 —":
    st.info("Please choose both players to continue.")
    st.stop()

# ②  Build opponent list once Player 1 is chosen
opps = pd.concat([
    df.loc[df[PLAYER_COLS[0]] == p1, PLAYER_COLS[1]],
    df.loc[df[PLAYER_COLS[1]] == p1, PLAYER_COLS[0]]
]).dropna()
opponents = sorted(set(opps[opps.apply(has_alpha)]) & set(eligible) - {p1})

p2_options = ["— Select Player 2 —"] + opponents
p2 = st.selectbox("Player 2", p2_options, index=0, key="p2")

if p2 == "— Select Player 2 —":
    st.info("Please choose both players to continue.")
    st.stop()

# ③  View switch (Round-robin is now the default)
view = st.radio("View",
                ["Round-robin", "Play-off"],
                horizontal=True,
                index=0,
                key="view")


# --------------------------------------------------------------------


# ── FILTER & PREP MATCHES ───────────────────────────────────────────
mask = (((df[PLAYER_COLS[0]] == p1) & (df[PLAYER_COLS[1]] == p2)) |
        ((df[PLAYER_COLS[0]] == p2) & (df[PLAYER_COLS[1]] == p1)))
filt = df.loc[mask].copy().apply(orient_row, axis=1, left_player=p1)

rr_mask  = filt["Stage"].str.contains(r"round[-\s]?robin", case=False, na=False)
round_rb = filt[rr_mask].copy()
playoff  = filt[~rr_mask].copy()

# ----------  DYNAMIC PAGE TITLE ----------

if view == "Round-robin":
    match_count = len(round_rb)
    view_label  = "Round-robin"
else:  # Play-off view
    match_count = len(playoff)
    view_label  = "play-off"

plural = "match" if match_count == 1 else "matches"
title  = f"{p1} and {p2} — {match_count} {view_label} {plural}"


if not playoff.empty and "RoundNumber" in playoff.columns:
    playoff["PlayoffStage"] = playoff["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown")

# Play-off series aggregation
if not playoff.empty:
    playoff["WinnerName"] = (playoff["Winner"] if "Winner" in playoff.columns
                             else np.where(playoff[SCORE_COLS[0]] > playoff[SCORE_COLS[1]], p1, p2))
    series = (playoff.groupby(["RoundNumber", "TournamentName", "StageID"])
              .agg(**{p1: ("WinnerName", lambda x: (x == p1).sum()),
                       p2: ("WinnerName", lambda x: (x == p2).sum()),
                       "Date": ("Date", "min")})
              .reset_index()
              .assign(PlayoffStage=lambda d: d["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown"))
              .pipe(add_url)
              .loc[:, [p1, p2, "PlayoffStage", "Date", "TournamentName", "URL"]])

# Prepare display DataFrames
cols_po = [p1, p2, "PlayoffStage", "PlayoffGameNumber", "Date", "TournamentName", "URL"]
po_show = prep_df(playoff, p1, p2).loc[:, [c for c in cols_po if c in prep_df(playoff, p1, p2).columns]]

cols_rr = [p1, p2, "RoundNumber", "Date", "TournamentName", "URL"]
rr_show = prep_df(round_rb, p1, p2).loc[:, [c for c in cols_rr if c in prep_df(round_rb, p1, p2).columns]]

# Choose the subset that matches the sidebar view
if view == "Round-robin":
    win_source = round_rb.copy()
else:                              # Play-off view → count INDIVIDUAL games
    win_source = playoff.copy()

# Derive winners per game
if win_source.empty:
    w1 = w2 = 0
else:
    if "Winner" in win_source.columns:
        winners = win_source["Winner"]
    else:
        winners = np.where(
            win_source[SCORE_COLS[0]] > win_source[SCORE_COLS[1]], p1,
            np.where(win_source[SCORE_COLS[1]] > win_source[SCORE_COLS[0]], p2, "Tie")
        )
    w1 = (winners == p1).sum()
    w2 = (winners == p2).sum()



link_cfg = {"URL": st.column_config.LinkColumn(label="", display_text="🔗")}
row_style = winner_style(p1, p2)  # Styler function

# ----------  H2H SUMMARY REPORT  ------------------------------------
# Put this BEFORE you display the round-robin / play-off tables
# -------------------------------------------------------------

# ----------  CLEANER H2H SUMMARY  -----------------------------------

def stat_dict(df_sub: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    """Return stats for `left` (df_sub must still have raw goal columns)."""
    oriented = df_sub.apply(orient_row, axis=1, left_player=left)
    gp   = len(oriented)
    wins = (oriented[SCORE_COLS[0]] > oriented[SCORE_COLS[1]]).sum()
    loss = (oriented[SCORE_COLS[1]] > oriented[SCORE_COLS[0]]).sum()
    draw = gp - wins - loss
    avg_for     = oriented[SCORE_COLS[0]].mean() if gp else 0
    avg_against = oriented[SCORE_COLS[1]].mean() if gp else 0
    win_rate = wins / gp * 100 if gp else 0
    return dict(GP=gp, W=wins, L=loss, D=draw,
                GF=round(avg_for, 2), GA=round(avg_against, 2),
                WR=round(win_rate, 1))

# totals in current view
tot_p1 = stat_dict(win_source_total := (round_rb if view=="Round-robin" else playoff), p1, p2)
tot_p2 = stat_dict(win_source_total, p2, p1)

# H2H only
h2h_p1 = stat_dict(win_source, p1, p2)
h2h_p2 = stat_dict(win_source, p2, p1)

# -------- cards ------------
c1, c2 = st.columns(2, gap="large")
for col, player, stats in [(c1, p1, tot_p1), (c2, p2, tot_p2)]:
    with col:
        st.markdown(f"#### {player}")
        st.metric("Games Played",   f"{stats['GP']:,}")
        st.metric("Wins",           f"{stats['W']:,}")
        st.metric("Losses",         f"{stats['L']:,}")
        st.metric("Win Rate",       f"{stats['WR']} %")
        st.metric("Avg Goals For",  f"{stats['GF']:.2f}")
        st.metric("Avg Goals Against",  f"{stats['GA']:.2f}")




# ── DISPLAY TABLES ──────────────────────────────────────────────────
if view == "Play-off":
    st.subheader("Play-off series scores")
    if playoff.empty:
        st.write("No play-off games between these players.")
    else:
        st.dataframe(series.style.apply(row_style, axis=1),
                     hide_index=True, use_container_width=True,
                     column_config=link_cfg)

    st.subheader("Play-off individual games")
    if playoff.empty:
        st.write("No play-off games between these players.")
    else:
        st.dataframe(po_show.style.apply(row_style, axis=1),
                     hide_index=True, use_container_width=True,
                     column_config=link_cfg)

else:  # Round-robin
    st.subheader("Round-robin")
    if round_rb.empty:
        st.write("No round-robin matches between these players.")
    else:
        st.dataframe(rr_show.style.apply(row_style, axis=1),
                     hide_index=True, use_container_width=True,
                     column_config=link_cfg)
