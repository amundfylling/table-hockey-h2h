"""
app.py — Stiga table-hockey head-to-head viewer
──────────────────────────────────────────────
Tables (in order)
1. Play-off series scores      –  <P1Wins> | <P2Wins> | PlayoffStage | Date | TournamentName | URL
2. Play-off individual games   –  goals(P1) | goals(P2) | PlayoffStage | PlayoffGameNumber | Date | TournamentName | URL
3. Round-robin                 –  goals(P1) | goals(P2) | RoundNumber | Date | TournamentName | URL
"""

import re, numpy as np
import streamlit as st
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_PATH   = "th_matches.parquet"
PLAYER_COLS = ["Player1", "Player2"]              # adjust if different
SCORE_COLS  = ["GoalsPlayer1", "GoalsPlayer2"]    # adjust if different
DROP_COLS   = ["StageID", "Player1ID", "Player2ID",
               "TournamentID", "StageSequence"]   # always hide from view
MIN_MATCHES = 50

ROUNDNUMBER_TO_STAGE = {
    1/64: "1/64 final",
    1/32: "1/32 final",
    1/16: "1/16 final",
    1/8:  "1/8 final",
    1/4:  "Quarter-final",
    1/2:  "Semi-final",
    1.0:  "Final",
    0.9:  "3rd-place match",
}

URL_FMT = "https://th.sportscorpion.com/eng/tournament/stage/{}/matches/"

# ── HELPERS ───────────────────────────────────────────────────────────
def has_alpha(name: str) -> bool:
    return bool(re.search(r"[A-Za-zÆØÅæøå]", str(name)))

def orient_row(row: pd.Series, left_player: str) -> pd.Series:
    """Ensure `left_player` is always in PLAYER_COLS[0]; swap names & scores if needed."""
    if row[PLAYER_COLS[0]] == left_player:
        return row
    # swap names
    row[PLAYER_COLS[0]], row[PLAYER_COLS[1]] = row[PLAYER_COLS[1]], row[PLAYER_COLS[0]]
    # swap scores
    if set(SCORE_COLS).issubset(row.index):
        row[SCORE_COLS[0]], row[SCORE_COLS[1]] = row[SCORE_COLS[1]], row[SCORE_COLS[0]]
    return row

def add_url_column(df: pd.DataFrame) -> pd.DataFrame:
    """Append a URL column based on StageID (StageID must be present)."""
    if "StageID" in df.columns:
        df["URL"] = df["StageID"].astype(int).astype(str).apply(lambda x: URL_FMT.format(x))
    else:
        df["URL"] = ""
    return df

def prep_for_display(df_in: pd.DataFrame, p1: str, p2: str) -> pd.DataFrame:
    """Hide IDs, rename goal columns to player names, drop player-name cols, add URL."""
    df_out = df_in.copy()
    # build URL before StageID disappears
    df_out = add_url_column(df_out)
    # rename goal columns
    if set(SCORE_COLS).issubset(df_out.columns):
        df_out = df_out.rename(columns={SCORE_COLS[0]: p1, SCORE_COLS[1]: p2})
    # drop unwanted columns
    df_out = df_out.drop(columns=[c for c in DROP_COLS if c in df_out.columns])
    df_out = df_out.drop(columns=[c for c in PLAYER_COLS if c in df_out.columns])
    return df_out

# ── LOAD DATA ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

df = load_data(DATA_PATH)

# ── ELIGIBLE PLAYERS ─────────────────────────────────────────────────
all_players   = pd.concat([df[c] for c in PLAYER_COLS]).dropna()
valid_players = all_players[all_players.apply(has_alpha)]
player_counts = valid_players.value_counts()
eligible      = sorted(player_counts[player_counts >= MIN_MATCHES].index)

if len(eligible) < 2:
    st.error(f"Need ≥2 players with ≥{MIN_MATCHES} matches and alphabetic names.")
    st.stop()

# ── SIDEBAR (PLAYER-ONLY FILTERS) ────────────────────────────────────
st.sidebar.header("Filters")
p1 = st.sidebar.selectbox("Player 1", eligible, index=0)

mask_vs_p1 = (df[PLAYER_COLS[0]] == p1) | (df[PLAYER_COLS[1]] == p1)
opps_series = pd.concat([
    df.loc[df[PLAYER_COLS[0]] == p1, PLAYER_COLS[1]],
    df.loc[df[PLAYER_COLS[1]] == p1, PLAYER_COLS[0]]
]).dropna()
opps_series = opps_series[opps_series.apply(has_alpha)]
opponents   = sorted(set(opps_series) & set(eligible) - {p1})

if not opponents:
    st.error(f"No qualifying opponents found for {p1}.")
    st.stop()

p2 = st.sidebar.selectbox("Player 2", opponents, index=0)

# ── FILTER DATA FOR SELECTED PLAYERS ─────────────────────────────────
mask_players = (
    ((df[PLAYER_COLS[0]] == p1) & (df[PLAYER_COLS[1]] == p2)) |
    ((df[PLAYER_COLS[0]] == p2) & (df[PLAYER_COLS[1]] == p1))
)
filtered = df.loc[mask_players].copy().apply(orient_row, axis=1, left_player=p1)

# ── CLASSIFY STAGES ─────────────────────────────────────────────────
if "Stage" not in filtered.columns:
    st.error("No 'Stage' column to classify matches.")
    st.stop()

rr_mask     = filtered["Stage"].str.contains(r"round[-\s]?robin", case=False, na=False)
round_robin = filtered[rr_mask].copy()
play_off    = filtered[~rr_mask].copy()

# friendly stage text for play_off rows
if not play_off.empty and "RoundNumber" in play_off.columns:
    play_off["PlayoffStage"] = play_off["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown")

# ── PAGE TITLE & W/L TALLY ──────────────────────────────────────────
st.title("Table-hockey head-to-head")
st.markdown(f"### {p1} vs {p2} — {len(filtered)} match(es)")

if "Winner" in filtered.columns and not filtered.empty:
    w1 = (filtered["Winner"] == p1).sum()
    w2 = (filtered["Winner"] == p2).sum()
    st.write(f"**Wins:** {p1} {w1} – {w2} {p2}")

# Column-config for link-icon (Streamlit ≥ 1.25)
link_cfg = {"URL": st.column_config.LinkColumn(label="", display_text="🔗")}

# ── 1. PLAY-OFF SERIES SCORES ───────────────────────────────────────
st.subheader("Play-off series scores")

if play_off.empty:
    st.write("No play-off games between these players.")
else:
    # winner per game
    if "Winner" in play_off.columns:
        play_off["WinnerName"] = play_off["Winner"]
    else:
        play_off["WinnerName"] = np.where(
            play_off[SCORE_COLS[0]] > play_off[SCORE_COLS[1]], p1, p2
        )

    # aggregate wins & earliest date per series (StageID constant inside series)
    series_tbl = (
        play_off
        .groupby(["RoundNumber", "TournamentName", "StageID"])
        .agg(
            P1Wins=("WinnerName", lambda x: (x == p1).sum()),
            P2Wins=("WinnerName", lambda x: (x == p2).sum()),
            Date  =("Date", "min")
        )
        .reset_index()
        .rename(columns={"P1Wins": p1, "P2Wins": p2})
    )

    # add PlayoffStage & URL
    series_tbl["PlayoffStage"] = series_tbl["RoundNumber"].map(ROUNDNUMBER_TO_STAGE).fillna("Unknown")
    series_tbl = add_url_column(series_tbl)

    # arrange cols & drop RoundNumber
    series_tbl = series_tbl[[p1, p2, "PlayoffStage", "Date", "TournamentName", "URL"]]

    st.dataframe(series_tbl, hide_index=True, use_container_width=True,
                 column_config=link_cfg)

# ── 2. PLAY-OFF INDIVIDUAL GAMES ────────────────────────────────────
st.subheader("Play-off individual games")

if play_off.empty:
    st.write("No play-off games between these players.")
else:
    po_show = prep_for_display(play_off, p1, p2)
    po_cols = [p1, p2, "PlayoffStage", "PlayoffGameNumber", "Date", "TournamentName", "URL"]
    po_show = po_show[[c for c in po_cols if c in po_show.columns]]
    st.dataframe(po_show.reset_index(drop=True), hide_index=True, use_container_width=True,
                 column_config=link_cfg)

# ── 3. ROUND-ROBIN TABLE ────────────────────────────────────────────
st.subheader("Round-robin")

if round_robin.empty:
    st.write("No round-robin matches between these players.")
else:
    rr_show = prep_for_display(round_robin, p1, p2)
    rr_cols = [p1, p2, "RoundNumber", "Date", "TournamentName", "URL"]
    rr_show = rr_show[[c for c in rr_cols if c in rr_show.columns]]
    st.dataframe(rr_show.reset_index(drop=True), hide_index=True, use_container_width=True,
                 column_config=link_cfg)
