import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go



# ── PAGE CONFIG & THEME LOCK ────────────────────────────────────────
st.set_page_config(page_title="Table-hockey H2H", layout="wide")




# ── CONSTANTS ────────────────────────────────────────────────────────
DATA_PATH   = "combined_matches.parquet"
PLAYER_COLS = ["Player1", "Player2"]
SCORE_COLS  = ["GoalsPlayer1", "GoalsPlayer2"]
HIDE_COLS   = ["Player1", "Player2"]
MIN_MATCHES = 50
URL_FMT     = "https://th.sportscorpion.com/eng/tournament/stage/{}/matches/"


# ── HELPERS ──────────────────────────────────────────────────────────
def has_alpha(x: str) -> bool:
    return bool(re.search(r"[A-Za-zÆØÅæøå]", str(x)))


def orient(row: pd.Series, left: str) -> pd.Series:
    if row["Player1"] != left:
        row["Player1"], row["Player2"] = row["Player2"], row["Player1"]
        row["GoalsPlayer1"], row["GoalsPlayer2"] = row["GoalsPlayer2"], row["GoalsPlayer1"]
    return row


def add_url(df: pd.DataFrame) -> pd.DataFrame:
    if "StageID" in df.columns:
        df["URL"] = df["StageID"].apply(
            lambda x: URL_FMT.format(int(x)) if not pd.isna(x) else np.nan
        )
    return df


def prep_table(df: pd.DataFrame, p1: str, p2: str) -> pd.DataFrame:
    df2 = add_url(df.copy()).rename(columns={SCORE_COLS[0]: p1, SCORE_COLS[1]: p2})
    to_drop = [c for c in HIDE_COLS if c in df2.columns]
    return df2.drop(columns=to_drop).sort_values("Date", ascending=False)


def hl_winner(p1: str, p2: str):
    def _s(r: pd.Series):
        css = [""] * len(r)
        if r[p1] > r[p2]:
            css[r.index.get_loc(p1)] = "background-color:#c6f6d5"
        elif r[p2] > r[p1]:
            css[r.index.get_loc(p2)] = "background-color:#c6f6d5"
        return css
    return _s


def player_stats(df_in: pd.DataFrame, pl: str, opp: str) -> dict:
    d = df_in.apply(orient, axis=1, left=pl)

    gp   = len(d)
    wins = (d["GoalsPlayer1"] > d["GoalsPlayer2"]).sum()
    loss = (d["GoalsPlayer2"] > d["GoalsPlayer1"]).sum()
    draw = gp - wins - loss

    tight_wins = ((d["GoalsPlayer1"] - d["GoalsPlayer2"]) == 1).sum()

    ot = d[d["Overtime"].str.contains("yes", case=False, na=False)] if "Overtime" in d.columns else pd.DataFrame()
    ot_games = len(ot)
    ot_wins  = (ot["GoalsPlayer1"] > ot["GoalsPlayer2"]).sum()
    ot_loss  = (ot["GoalsPlayer1"] < ot["GoalsPlayer2"]).sum()

    goals_for     = d["GoalsPlayer1"].sum()
    goals_against = d["GoalsPlayer2"].sum()
    gd            = goals_for - goals_against
    avg_f         = d["GoalsPlayer1"].mean() if gp else 0
    avg_a         = d["GoalsPlayer2"].mean() if gp else 0

    d["Diff"] = abs(d["GoalsPlayer1"] - d["GoalsPlayer2"])
    wins_mask      = d["GoalsPlayer1"] > d["GoalsPlayer2"]

    biggest_win  = (
        d[wins_mask]
        .sort_values(["Diff", "GoalsPlayer1"], ascending=[False, False])
        .head(1)
    )
    biggest_loss = (
        d[~wins_mask]
        .sort_values(["Diff", "GoalsPlayer2"], ascending=[False, False])
        .head(1)
    )

    def fmt_big(row: pd.DataFrame) -> str:
        if row.empty:
            return "–"
        score = f"{row.iloc[0]['GoalsPlayer1']}-{row.iloc[0]['GoalsPlayer2']}"
        date_val = row.iloc[0]["Date"]
        date_str = date_val.strftime("%Y-%m-%d") if isinstance(date_val, pd.Timestamp) else "unknown"
        return f"{score} ({date_str})"

    return dict(
        GP        = gp,
        Wins      = f"{wins} ({wins/gp*100:.1f} %)" if gp else "0",
        Losses    = f"{loss} ({loss/gp*100:.1f} %)" if gp else "0",
        Draws     = f"{draw} ({draw/gp*100:.1f} %)" if gp else "0",
        TightWins = tight_wins,
        OT        = f"{ot_games} ({ot_wins}/{ot_loss})" if ot_games else "–",
        GF        = goals_for,
        GA        = goals_against,
        GD        = gd,
        AvgF      = f"{avg_f:.2f}",
        AvgA      = f"{avg_a:.2f}",
        BigWin    = fmt_big(biggest_win),
        BigLoss   = fmt_big(biggest_loss),
    )


# ── LOAD DATA ────────────────────────────────────────────────────────
@st.cache_data
def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # list of (TournamentName, Date) to remove
    exclusions = [
        ("EM", "2022-06-11"),
        ("Swedish Masters", "2016-02-13"),
    ]

    # iteratively filter out matching rows
    for name, date_str in exclusions:
        date = pd.to_datetime(date_str)
        df = df[~((df["TournamentName"] == name) & (df["Date"] == date))]

    return df

df = load()


# ── PLAYER CHOICE UI ────────────────────────────────────────────────
eligible = (
    pd.concat([df[c] for c in PLAYER_COLS])
      .dropna()
      .loc[lambda s: s.apply(has_alpha)]
      .value_counts()
      .loc[lambda s: s >= MIN_MATCHES]
      .index
      .sort_values()
      .tolist()
)

st.markdown("### Select players")
p1 = st.selectbox("Player 1", eligible, placeholder="Select Player 1")
if p1 is None:
    st.stop()

opps_raw = pd.concat([
    df.loc[df["Player1"] == p1, "Player2"],
    df.loc[df["Player2"] == p1, "Player1"],
]).dropna()
opponents = sorted(set(opps_raw.loc[opps_raw.apply(has_alpha)]) & set(eligible) - {p1})

p2 = st.selectbox("Player 2", opponents, placeholder="Select Player 2")
if p2 is None:
    st.stop()

view = st.radio("", ["Round-robin", "Play-off", "Both"], horizontal=True, index=2)


# ── FILTER MATCHES FOR SELECTED PAIR ─────────────────────────────────
pair = df[
    ((df["Player1"] == p1) & (df["Player2"] == p2)) |
    ((df["Player1"] == p2) & (df["Player2"] == p1))
].copy()
pair = pair.apply(orient, axis=1, left=p1)


# ── DATE FILTER ──────────────────────────────────────────────────────
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

if start > end:
    st.warning("Start date is after end date — swapping them.")
    start, end = end, start

pair = pair[
    (pair["Date"] >= pd.to_datetime(start)) &
    (pair["Date"] <= pd.to_datetime(end))
]


# ── SPLIT INTO ROUND-ROBIN vs PLAY-OFF ──────────────────────────────
rr = pair[pair["Stage"].str.contains(r"round[-\s]?robin", case=False, na=False)]
po = pair[~pair["Stage"].str.contains(r"round[-\s]?robin", case=False, na=False)]

if not po.empty and "RoundNumber" in po.columns:
    po = po.assign(PlayoffStage=po["RoundNumber"].fillna("Unknown"))


# ── CHOOSE CURRENT SUBSET FOR STATS ─────────────────────────────────
if view == "Round-robin":
    current_df = rr
elif view == "Play-off":
    current_df = po
else:  # Both
    current_df = pd.concat([rr, po], ignore_index=True)


# ── DISPLAY TABS ────────────────────────────────────────────────────
stats_tab, games_tab, charts_tab = st.tabs(["📊 Stats", "📑 All games", "📈 Charts"])



with stats_tab:
    st.divider()
    matches_played = len(current_df)
    st.metric("Matches Played", matches_played if matches_played else "0")

    left_stats  = player_stats(current_df, p1, p2)
    right_stats = player_stats(current_df, p2, p1)

    c1, c2 = st.columns(2, gap="small", border=True)
    for col, pl, data in [(c1, p1, left_stats), (c2, p2, right_stats)]:
        with col:
            with st.expander(pl, expanded=True):
                st.markdown(f"#### {pl}")
                st.metric("Wins",       data["Wins"])
                st.metric("Losses",     data["Losses"])
                st.metric("Draws",      data["Draws"])
                st.metric("Tight wins (by 1 goal)", data["TightWins"])
                if view in ("Play-off", "Both"):
                    st.metric("Overtime (Won/Lost)", data["OT"])
                st.metric("Goals Scored",       data["GF"])
                st.metric("Goals Conceded",     data["GA"])
                st.metric("Goal Diff.",         data["GD"])
                st.metric("Avg Goals Scored",   data["AvgF"])
                st.metric("Avg Goals Conceded", data["AvgA"])
                st.metric("Biggest Win",        data["BigWin"])
                st.metric("Biggest Loss",       data["BigLoss"])


with games_tab:
    link_cfg = {"URL": st.column_config.LinkColumn(label="URL", display_text="🔗")}
    sty = hl_winner(p1, p2)

    def show_rr():
        rr_tbl = (
            prep_table(rr, p1, p2)
            .drop(columns=[c for c in ["Overtime", "Stage", "PlayoffGameNumber", "StageID"] if c in rr.columns])
        )
        if "RoundNumber" in rr_tbl.columns:
            rr_tbl["RoundNumber"] = pd.to_numeric(rr_tbl["RoundNumber"], errors="coerce").astype("Int64")
        rr_tbl["Date"] = rr_tbl["Date"].dt.date

        st.subheader("Round-robin")
        st.dataframe(
            rr_tbl.style.apply(sty, axis=1),
            hide_index=True,
            use_container_width=True,
            column_config=link_cfg,
        )

    if view == "Play-off":
        # ── PLAY-OFF SERIES SCORES ────────────────────────────────────────
        if not po.empty:
            po = po.assign(
                WinnerName=np.where(
                    po["GoalsPlayer1"] > po["GoalsPlayer2"], p1,
                    np.where(po["GoalsPlayer2"] > po["GoalsPlayer1"], p2, "Tie")
                )
            )

            series = (
                po.groupby(["RoundNumber", "TournamentName", "Date"])
                  .agg(
                      **{
                          p1: ("WinnerName", lambda wins: (wins == p1).sum()),
                          p2: ("WinnerName", lambda wins: (wins == p2).sum()),
                          "StageID": ("StageID", lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan),
                      }
                  )
                  .reset_index()
                  .assign(
                      PlayoffStage=lambda df: df["RoundNumber"].fillna("Unknown"),
                      RoundNumber=lambda df: pd.to_numeric(df["RoundNumber"], errors="coerce").astype("Int64"),
                      URL=lambda df: df["StageID"].apply(lambda x: URL_FMT.format(int(x)) if not pd.isna(x) else np.nan)
                  )
                  .drop(columns=["StageID", "RoundNumber"])
            )

            series["Date"] = series["Date"].dt.date
            series = series.sort_values("Date", ascending=False)

            # Reorder so that TournamentName follows player columns
            cols = [p1, p2, "TournamentName", "PlayoffStage", "Date", "URL"]
            series = series.loc[:, cols]

            st.subheader("Play-off series scores")
            st.dataframe(
                series.style.apply(hl_winner(p1, p2), axis=1),
                hide_index=True,
                use_container_width=True,
                column_config=link_cfg,
            )

        # ── PLAY-OFF INDIVIDUAL GAMES ────────────────────────────────────
        po_tbl = add_url(po.copy()).rename(columns={SCORE_COLS[0]: p1, SCORE_COLS[1]: p2})
        po_tbl["RoundNumber"] = pd.to_numeric(po_tbl["RoundNumber"], errors="coerce").astype("Int64")
        po_tbl["PlayoffGameNumber"] = po_tbl["PlayoffGameNumber"].astype("Int64")
        po_tbl["Date"] = po_tbl["Date"].dt.date

        to_drop = [c for c in ["Player1", "Player2", "StageID", "RoundNumber", "WinnerName", "Stage"] if c in po_tbl.columns]
        po_tbl = po_tbl.drop(columns=to_drop)

        # Sort by Date DESC, then by PlayoffGameNumber DESC
        po_tbl = po_tbl.sort_values(["Date", "PlayoffGameNumber"], ascending=[False, False])

        st.subheader("Play-off individual games")
        st.dataframe(
            po_tbl.style.apply(sty, axis=1),
            hide_index=True,
            use_container_width=True,
            column_config=link_cfg,
        )

    elif view == "Round-robin":
        show_rr()

    else:  # Both
        # ── PLAY-OFF SERIES SCORES ────────────────────────────────────────
        if not po.empty:
            po = po.assign(
                WinnerName=np.where(
                    po["GoalsPlayer1"] > po["GoalsPlayer2"], p1,
                    np.where(po["GoalsPlayer2"] > po["GoalsPlayer1"], p2, "Tie")
                )
            )

            series = (
                po.groupby(["RoundNumber", "TournamentName", "Date"])
                  .agg(
                      **{
                          p1: ("WinnerName", lambda wins: (wins == p1).sum()),
                          p2: ("WinnerName", lambda wins: (wins == p2).sum()),
                          "StageID": ("StageID", lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan),
                      }
                  )
                  .reset_index()
                  .assign(
                      PlayoffStage=lambda df: df["RoundNumber"].fillna("Unknown"),
                      RoundNumber=lambda df: pd.to_numeric(df["RoundNumber"], errors="coerce").astype("Int64"),
                      URL=lambda df: df["StageID"].apply(lambda x: URL_FMT.format(int(x)) if not pd.isna(x) else np.nan)
                  )
                  .drop(columns=["StageID", "RoundNumber"])
            )

            series["Date"] = series["Date"].dt.date
            series = series.sort_values("Date", ascending=False)
            cols = [p1, p2, "TournamentName", "PlayoffStage", "Date", "URL"]
            series = series.loc[:, cols]

            st.subheader("Play-off series scores")
            st.dataframe(
                series.style.apply(hl_winner(p1, p2), axis=1),
                hide_index=True,
                use_container_width=True,
                column_config=link_cfg,
            )

        # ── PLAY-OFF INDIVIDUAL GAMES ────────────────────────────────────
        po_tbl = add_url(po.copy()).rename(columns={SCORE_COLS[0]: p1, SCORE_COLS[1]: p2})
        po_tbl["RoundNumber"] = pd.to_numeric(po_tbl["RoundNumber"], errors="coerce").astype("Int64")
        po_tbl["PlayoffGameNumber"] = po_tbl["PlayoffGameNumber"].astype("Int64")
        po_tbl["Date"] = po_tbl["Date"].dt.date

        to_drop = [c for c in ["Player1", "Player2", "StageID", "RoundNumber", "WinnerName", "Stage"] if c in po_tbl.columns]
        po_tbl = po_tbl.drop(columns=to_drop)
        po_tbl = po_tbl.sort_values(["Date", "PlayoffGameNumber"], ascending=[False, False])

        st.subheader("Play-off individual games")
        st.dataframe(
            po_tbl.style.apply(sty, axis=1),
            hide_index=True,
            use_container_width=True,
            column_config=link_cfg,
        )

        # ── ROUND-ROBIN TABLE ───────────────────────────────────────────
        show_rr()

with charts_tab:
    import plotly.express as px
    import numpy as np

    # NBHF colours
    BLUE  = "#233469"
    RED   = "#ff4344"
    YELLOW= "#f9c561"
    WHITE = "#ffffff"

    # 1) Goal‐Difference Distribution with NBHF blue bars
    diffs = (current_df["GoalsPlayer1"] - current_df["GoalsPlayer2"]).abs()
    hist = (
        diffs.value_counts()
             .sort_index()
             .rename_axis("Goal Difference")
             .reset_index(name="count")
    )
    hist["percent"] = hist["count"] / hist["count"].sum()

    fig1 = px.bar(
        hist,
        x="Goal Difference",
        y="count",
        text=hist.apply(lambda r: f"{int(r['count'])} ({r['percent']:.1%})", axis=1),
        title="Goal Difference Distribution",
        color_discrete_sequence=[BLUE]   # use NBHF blue
    )
    fig1.update_traces(textposition="outside", cliponaxis=False)
    fig1.update_layout(
        bargap=0.2,
        xaxis=dict(dtick=1),
        yaxis_title="Count",
        plot_bgcolor=WHITE,       # white background
        paper_bgcolor=WHITE
    )
    st.plotly_chart(fig1, use_container_width=True)


    # 2) Yearly Outcome Proportions with NBHF colours
    out = current_df.assign(
        Outcome=np.where(
            current_df["GoalsPlayer1"] > current_df["GoalsPlayer2"], p1,
            np.where(current_df["GoalsPlayer1"] < current_df["GoalsPlayer2"], p2, "Draw")
        ),
        Year=current_df["Date"].dt.year
    )
    grp_counts = (
        out.groupby(["Year", "Outcome"])
           .size()
           .unstack(fill_value=0)
    )
    for cat in [p1, "Draw", p2]:
        if cat not in grp_counts.columns:
            grp_counts[cat] = 0
    grp_counts = grp_counts[[p1, "Draw", p2]].sort_index()
    grp_prop = grp_counts.div(grp_counts.sum(axis=1), axis=0)

    df_counts = grp_counts.reset_index().melt(
        id_vars="Year", value_vars=[p1, "Draw", p2],
        var_name="Outcome", value_name="Count"
    )
    df_prop = grp_prop.reset_index().melt(
        id_vars="Year", value_vars=[p1, "Draw", p2],
        var_name="Outcome", value_name="Proportion"
    )
    df_bar = df_counts.merge(df_prop, on=["Year", "Outcome"])
    df_bar["label"] = df_bar.apply(lambda r: f"{int(r['Count'])}", axis=1)

    fig2 = px.bar(
        df_bar,
        x="Year",
        y="Proportion",
        color="Outcome",
        text="label",
        category_orders={"Outcome": [p1, "Draw", p2]},
        barmode="stack",
        title="Win Rate by Year",
        color_discrete_map={
            p1: BLUE,     # Player1 = blue
            "Draw": YELLOW,
            p2: RED       # Player2 = red
        }
    )
    fig2.update_traces(textposition="inside")
    fig2.update_layout(
        yaxis_tickformat=".0%",
        xaxis=dict(type="category"),
        yaxis_title="Proportion",
        plot_bgcolor=WHITE,
        paper_bgcolor=WHITE
    )
    st.plotly_chart(fig2, use_container_width=True)
