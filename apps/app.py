"""
Life After 50 by County - Streamlit app
---------------------------------------
Reads the data pack written by section 12b of Modeling.ipynb (the app_*.csv files).
It never runs the model.

Run it:
    pip install -r requirements.txt
    streamlit run app.py

Where the data is found (first match wins):
    1. the folder in the LE50_APP_DATA environment variable
    2. the folder this file is in
    3. an "app" or "data" folder next to this file
    4. C:/Users/14438/data/processed/app
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import copy

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------- look and feel
INK, MUTED, ACCENT, SURFACE, BORDER = "#0F172A", "#64748B", "#2563EB", "#F8FAFC", "#E2E8F0"
THEME = {"theme.base": "light", "theme.primaryColor": ACCENT, "theme.backgroundColor": "#FFFFFF",
         "theme.secondaryBackgroundColor": "#F1F5F9", "theme.textColor": INK}
_theme_changed = False
try:                                   # same effect as a .streamlit/config.toml, kept in one file
    from streamlit import config as _cfg
    for _k, _v in THEME.items():
        if _cfg.get_option(_k) != _v:
            _cfg.set_option(_k, _v)
            _theme_changed = True
except Exception:
    pass

st.set_page_config(page_title="Life After 50 by County", layout="wide")
if _theme_changed:                     # first run after start-up: rerun once so the colours take effect
    st.rerun()

pio.templates["le50"] = go.layout.Template(layout=dict(
    font=dict(family="Inter, 'Source Sans Pro', 'Segoe UI', sans-serif", color=INK, size=13),
    paper_bgcolor="white", plot_bgcolor="white", colorway=[ACCENT],
    xaxis=dict(gridcolor="#EEF2F6", zerolinecolor=BORDER, linecolor=BORDER, title_font=dict(color=MUTED, size=12)),
    yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=BORDER),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font=dict(color=INK, size=13))))
pio.templates.default = "le50"

CSS = """
<style>
[data-testid="stAppDeployButton"] {display: none;}
.block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1380px;}
h1, h2, h3, h4 {letter-spacing: -0.015em; color: #0F172A;}
section[data-testid="stSidebar"] {background: #F8FAFC; border-right: 1px solid #E2E8F0;}
section[data-testid="stSidebar"] .block-container {padding-top: 1.6rem;}

/* header */
.le-eyebrow {font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: #2563EB;}
.le-title {font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; color: #0F172A; margin: .15rem 0 .35rem 0;}
.le-sub {color: #64748B; font-size: 1rem; margin-bottom: 1.1rem;}
.le-brand {font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; color: #0F172A;}
.le-brand span {color: #2563EB;}

/* cards */
.le-kpis {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin-bottom: 1.1rem;}
.le-card {background: #fff; border: 1px solid #E2E8F0; border-radius: 16px; padding: 14px 18px;
          box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px -18px rgba(15,23,42,.25);}
.le-card .lab {font-size: .72rem; font-weight: 700; letter-spacing: .07em; text-transform: uppercase; color: #64748B;
               display: flex; align-items: center; gap: 7px;}
.le-card .val {font-size: 1.85rem; font-weight: 750; letter-spacing: -0.02em; color: #0F172A; line-height: 1.25;}
.le-card .sub {font-size: .8rem; color: #64748B;}
.le-dot {width: 10px; height: 10px; border-radius: 50%; display: inline-block;}
.le-legend {display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: .8rem; color: #475569; align-items: center;}
.le-legend span {display: inline-flex; align-items: center; gap: 6px;}
.le-hero {border-radius: 18px; padding: 18px 22px; color: #fff; height: 100%;}
.le-hero .lab {font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .9;}
.le-hero .val {font-size: 2.9rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05; margin: 4px 0;}
.le-hero .sub {font-size: .92rem; opacity: .95;}

[data-testid="stMetric"] {background: #fff; border: 1px solid #E2E8F0; border-radius: 14px; padding: 12px 16px;
                          box-shadow: 0 1px 2px rgba(15,23,42,.04);}
[data-testid="stMetricLabel"] p {font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #64748B;}
[data-testid="stMetricValue"] {font-weight: 700; font-size: 1.5rem; color: #0F172A;}

/* tabs as pills (covers both the newer and the older Streamlit tab markup) */
[role="tablist"] {gap: 4px !important; border-bottom: none !important; box-shadow: none !important; background: #F1F5F9;
                  padding: 5px; border-radius: 999px; width: fit-content; margin-bottom: .6rem;}
[role="tablist"]::after, [role="tablist"]::before {display: none !important;}
[role="tab"] {border-radius: 999px !important; padding: 6px 18px !important; height: auto !important; color: #475569 !important;
              border: none !important; cursor: pointer;}
[role="tab"] p {font-weight: 600; font-size: .9rem;}
[role="tab"][aria-selected="true"] {background: #fff !important; color: #0F172A !important; box-shadow: 0 1px 3px rgba(15,23,42,.16);}
.react-aria-SelectionIndicator, [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {display: none !important;}

/* keep charts at full strength while the app is working, so a click does not make the map fade */
[data-stale="true"]:has([data-testid="stPlotlyChart"]) {opacity: 1 !important; transition: none !important;}
[data-testid="stPlotlyChart"] {opacity: 1 !important;}

/* containers */
[data-testid="stPlotlyChart"] {border: 1px solid #E2E8F0; border-radius: 16px; overflow: hidden; background: #fff;}
[data-testid="stExpander"] details {border: 1px solid #E2E8F0; border-radius: 14px; background: #fff;}
[data-testid="stExpander"] summary p {font-weight: 600;}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
div[role="dialog"] {border-radius: 22px;}
[data-testid="stAlert"] {border-radius: 14px;}
.stButton button {border-radius: 999px; font-weight: 600;}
</style>
"""

# --------------------------------------------------------------------------- settings
BAND = 1.0          # years either side of the comparison that still count as "about the same"
MIN_GAIN = 0.1      # combined gains smaller than this are shown as "no measurable gain"
GREEN, YELLOW, RED, GREY, FAINT = "#22A565", "#F7C948", "#E5484D", "#D5DAE1", "#EEF1F5"
STATUS_COLOR = {"Green": GREEN, "Yellow": YELLOW, "Red": RED}
STATUS_DOT = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}
GRADE_COLOR = {"A": "#22A565", "B": "#8BCF5A", "C": "#F7C948", "D": "#FB923C", "F": "#E5484D"}
GEO_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
PLACE_TYPES = ["Large metro", "Small metro", "Town", "Rural"]
POP_BINS = [0, 10_000, 50_000, 250_000, np.inf]
POP_LABELS = ["Under 10,000", "10,000 to 49,999", "50,000 to 249,999", "250,000 or more"]
TARGET_ORDER = ["peer_median", "peer_top_quartile", "peer_best", "national_top_decile"]
TARGET_HELP = {
    "peer_median": "Catch up to the typical similar county.",
    "peer_top_quartile": "Reach what the top quarter of similar counties already reach.",
    "peer_best": "Match the single best similar county on each factor. Treat as a ceiling.",
    "national_top_decile": "Reach the national top 10%, ignoring peers. Treat as a ceiling.",
}
NOT_PERCENT = {"acs_gini", "chr_drinking_water_violations", "chr_food_environment_index"}
STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


# --------------------------------------------------------------------------- loading
def find_data_dir() -> Path | None:
    here = Path(__file__).resolve().parent
    candidates = [os.environ.get("LE50_APP_DATA"), here, here / "app", here / "data",
                  r"C:/Users/14438/data/processed/app"]
    for c in candidates:
        if c and (Path(c) / "app_counties.csv").exists():
            return Path(c)
    return None


@st.cache_data(show_spinner="Loading county data...")
def load_pack(folder: str):
    d = Path(folder)
    txt = {"fips": str, "peer_fips": str, "member_fips": str}
    counties = pd.read_csv(d / "app_counties.csv", dtype=txt)
    feats = pd.read_csv(d / "app_county_features.csv", dtype=txt, low_memory=False)
    scen = pd.read_csv(d / "app_county_scenarios.csv", dtype=txt)
    peers = pd.read_csv(d / "app_county_peers.csv", dtype=txt)
    guide = pd.read_csv(d / "app_feature_guide.csv")
    summary = json.loads((d / "app_summary.json").read_text()) if (d / "app_summary.json").exists() else {}

    r = counties["rucc"].round()
    counties["place_type"] = np.select([r == 1, r.between(2, 3), r.between(4, 7), r >= 8],
                                       PLACE_TYPES, default="Unknown")
    counties["pop_group"] = pd.cut(counties["population"], POP_BINS, labels=POP_LABELS,
                                   right=False).astype(object)
    counties["state_name"] = counties["state"].map(STATE_NAMES).fillna(counties["state"])
    counties["vs_peers_years"] = counties["le50"] - counties["peer_mean_le50"]
    for col in ("grade", "band", "vs_peers"):
        feats[col] = feats[col].where(feats[col].notna(), "")
    return counties, feats, scen, peers, guide, summary


@st.cache_data(show_spinner="Loading county boundaries...")
def load_geo(folder: str):
    d = Path(folder)
    for p in (d / "counties.geojson", d.parent / "counties.geojson"):
        if p.exists():
            return json.loads(p.read_text())
    try:                                             # first run only; saved for next time
        with urllib.request.urlopen(GEO_URL, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        try:
            (d / "counties.geojson").write_text(raw)
        except OSError:
            pass
        return json.loads(raw)
    except Exception:
        return None


def _merc(lat):
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


def _unmerc(y):
    return np.degrees(2 * np.arctan(np.exp(np.radians(y))) - np.pi / 2)


# state code -> (where it really is, where to draw it, how much to shrink it); (lon, lat) pairs
INSETS = {"02": ((-152.5, 63.5), (-118.6, 26.6), 0.20),      # Alaska: a fifth of its map size
          "15": ((-157.4, 20.6), (-107.3, 25.6), 1.00)}      # Hawaii: same size, moved


@st.cache_data(show_spinner=False)
def load_geo_inset(folder: str):
    """County boundaries with Alaska shrunk and both Alaska and Hawaii moved next to the
    lower 48. Shapes are scaled on the map's own projection so they keep their look."""
    geo = load_geo(folder)
    if geo is None:
        return None
    geo = copy.deepcopy(geo)

    def shift(ring, src, dst, k):
        a = np.asarray(ring, dtype=float)
        lon = np.where(a[:, 0] > 0, a[:, 0] - 360, a[:, 0])          # Aleutians past the date line
        x = dst[0] + (lon - src[0]) * k
        y = _merc(dst[1]) + (_merc(a[:, 1]) - _merc(src[1])) * k
        return np.column_stack([x, _unmerc(y)]).round(4).tolist()

    for ft in geo["features"]:
        spec = INSETS.get(str(ft["id"])[:2])
        if not spec:
            continue
        g = ft["geometry"]
        if g["type"] == "Polygon":
            g["coordinates"] = [shift(r, *spec) for r in g["coordinates"]]
        else:
            g["coordinates"] = [[shift(r, *spec) for r in poly] for poly in g["coordinates"]]
    return geo


def with_status(counties: pd.DataFrame, basis: str) -> pd.DataFrame:
    c = counties.copy()
    c["status_diff"] = c["years_vs_national_median"] if basis == "nation" else c["vs_peers_years"]
    c["status"] = np.select([c["status_diff"] >= BAND, c["status_diff"] <= -BAND],
                            ["Green", "Red"], default="Yellow")
    return c


# --------------------------------------------------------------------------- formatting
def fmt_value(feature: str, v, guide_row=None) -> str:
    if pd.isna(v):
        return "No data"
    if feature == "chr_drinking_water_violations":
        return "Yes" if v >= 0.5 else "No"
    if feature == "acs_log_pop":
        return f"{np.exp(v):,.0f} people"
    gmax = guide_row["national_max"] if guide_row is not None else np.nan
    gmin = guide_row["national_min"] if guide_row is not None else np.nan
    if feature not in NOT_PERCENT and pd.notna(gmax) and gmax <= 1.0 and gmin >= 0:
        return f"{100 * v:.1f}%"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:,.1f}" if abs(v) >= 10 else f"{v:,.2f}"


def years(v, signed=True) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{v:+.1f} years" if signed else f"{v:.1f} years"


def ordinal(n) -> str:
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n:,}{suffix}"


def chip(text: str, color: str, dark_text=False) -> str:
    fg = "#222" if dark_text else "#fff"
    return (f"<span style='background:{color};color:{fg};padding:2px 10px;border-radius:12px;"
            f"font-weight:600;font-size:0.9rem'>{text}</span>")


def dot(color: str) -> str:
    return f"<i class='le-dot' style='background:{color}'></i>"


def kpi(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    lead = dot(color) if color else ""
    return (f"<div class='le-card'><div class='lab'>{lead}{label}</div><div class='val'>{value}</div>"
            f"<div class='sub'>{sub}&nbsp;</div></div>")


def status_sentence(row, basis: str) -> str:
    what = "the national median" if basis == "nation" else "the average of its 25 similar counties"
    d = row["status_diff"]
    if row["status"] == "Yellow":
        return f"Within a year of {what} ({d:+.1f})."
    word = "above" if d > 0 else "below"
    return f"{abs(d):.1f} years {word} {what}."


# --------------------------------------------------------------------------- county page
def county_page(fips: str, C: pd.DataFrame, feats, scen, peers, guide, summary, basis: str):
    row = C[C["fips"] == fips].iloc[0]
    g = guide.set_index("feature")
    f = feats[feats["fips"] == fips]

    st.markdown(
        f"{chip(row['status'], STATUS_COLOR[row['status']], row['status'] == 'Yellow')}"
        f" &nbsp; {chip('Grade ' + str(row['le50_grade']), GRADE_COLOR.get(row['le50_grade'], GREY), row['le50_grade'] in ('B', 'C'))}",
        unsafe_allow_html=True)
    bits = [row["place_type"], row["rucc_label"] if isinstance(row["rucc_label"], str) else None,
            f"population {row['population']:,.0f}" if pd.notna(row["population"]) else None]
    st.caption(" · ".join(b for b in bits if b))
    if bool(row.get("combined_unit", False)):
        st.caption(f"Reported together with neighbouring counties as one unit: {row['member_fips'].replace(';', ', ')}")

    tint = {"Green": "linear-gradient(135deg,#16A34A,#22A565)", "Yellow": "linear-gradient(135deg,#D99A06,#F2B635)",
            "Red": "linear-gradient(135deg,#DC2626,#E5484D)"}[row["status"]]
    hero, grid = st.columns([1.05, 3])
    hero.markdown(
        f"<div class='le-hero' style='background:{tint}'><div class='lab'>Life expectancy at 50</div>"
        f"<div class='val'>{row['le50']:.1f}<span style='font-size:1.1rem;font-weight:600'> years</span></div>"
        f"<div class='sub'>Lives to about {row['expected_age_at_death_from_50']:.0f}<br>"
        f"{row['years_vs_national_median']:+.1f} years vs the national median</div></div>",
        unsafe_allow_html=True)
    with grid:
        m = st.columns(3)
        m[0].metric("National rank", f"{int(row['le50_national_rank']):,} of {len(C):,}",
                    help="1 is the longest lived county.")
        m[1].metric(f"Rank in {row['state']}", f"{int(row['le50_state_rank'])} of {int(row['counties_in_state'])}")
        m[2].metric("Among 25 similar counties", f"{ordinal(row['rank_among_peers'])} of {int(row['peer_group_size'])}",
                    f"{row['vs_peers_years']:+.1f} vs their average")
        n = st.columns(3)
        if "le50_male" in row and pd.notna(row.get("le50_male")):
            n[0].metric("Men / women at 50", f"{row['le50_male']:.1f} / {row['le50_female']:.1f}")
        elif pd.notna(row.get("le_birth")):
            n[0].metric("At birth", f"{row['le_birth']:.1f} years")
        n[1].metric("Model expected", f"{row['model_predicted_le50']:.1f} years",
                    f"{row['actual_minus_predicted']:+.1f} actual vs expected", delta_color="off")
        n[2].metric("Report card average", f"{row['gpa_overall']:.1f} of 4.0" if pd.notna(row["gpa_overall"]) else "n/a",
                    help="Average of the A to F grades below. A = 4, F = 0.")
    st.info(f"**{status_sentence(row, basis)}**  {row['vs_expectation']}.")

    t_change, t_card, t_profile, t_peers, t_drivers = st.tabs(
        ["What it could change", "Report card", "County profile", "Similar counties", "What shapes the estimate"])

    # ---- what it could change
    with t_change:
        labels = summary.get("targets", {k: k for k in TARGET_ORDER})
        keys = [k for k in TARGET_ORDER if f"gain_{k}" in row.index]
        left, right = st.columns([1, 2])
        with left:
            target = st.radio("Set the bar at", keys, index=keys.index("peer_top_quartile") if "peer_top_quartile" in keys else 0,
                              format_func=lambda k: labels.get(k, k), key=f"target_{fips}")
            st.caption(TARGET_HELP.get(target, ""))
            gain = row[f"gain_{target}"]
            n_behind = int(row[f"factors_behind_{target}"])
            if pd.isna(gain) or gain < MIN_GAIN:
                st.metric("All factors together", "No measurable gain")
                st.caption("This county is already close to that level on the factors it can change.")
            else:
                st.metric("All factors together", f"+{gain:.1f} years", f"{n_behind} factors behind the target",
                          delta_color="off")
                st.caption("Smaller than the bars added up, because the factors overlap.")
        s = scen[(scen["fips"] == fips) & (scen["target"] == target)].sort_values("years_gained", ascending=False)
        show = s[s["years_gained"] >= 0.01]
        with right:
            if show.empty:
                st.write("Nothing here moves the estimate by a measurable amount.")
            else:
                top = show.head(10).iloc[::-1]
                fig = go.Figure(go.Bar(
                    x=top["years_gained"], y=top["label"], orientation="h",
                    marker_color=[RED if b else GREEN for b in top["beyond_peers"]],
                    text=[f"+{v:.2f}" for v in top["years_gained"]], textposition="outside",
                    hovertemplate="%{y}<br>+%{x:.2f} years<extra></extra>"))
                fig.update_layout(height=max(260, 34 * len(top) + 90), margin=dict(l=10, r=40, t=30, b=40),
                                  xaxis_title="Years of life expectancy at 50, one factor at a time",
                                  title="Largest single changes")
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if not show.empty:
            tab = pd.DataFrame({
                "Factor": show["label"], "Area": show["theme"],
                "Now": [fmt_value(ft, v, g.loc[ft] if ft in g.index else None) for ft, v in zip(show["feature"], show["current"])],
                "Target": [fmt_value(ft, v, g.loc[ft] if ft in g.index else None) for ft, v in zip(show["feature"], show["target_value"])],
                "Years": show["years_gained"].round(2),
                "Beyond any similar county": np.where(show["beyond_peers"], "Yes", ""),
            })
            st.dataframe(tab, hide_index=True, width="stretch")
            if show["beyond_peers"].any():
                st.caption("Red bars are targets no similar county has reached. The model has hardly seen a county like that.")
        st.caption(summary.get("caveat", "Associations between counties, not causes."))

    # ---- report card
    with t_card:
        graded = f[f["grade"] != ""].copy()
        counts = st.columns(6)
        for k, L in enumerate("ABCDF"):
            counts[k].markdown(f"{chip(L, GRADE_COLOR[L], L in ('B', 'C'))} &nbsp; **{int(row.get(f'n_grade_{L}', 0))}**",
                               unsafe_allow_html=True)
        counts[5].caption("A is the top fifth of all counties. C is typical.")
        strengths = [f"{row[f'strength_{k}']} ({row[f'strength_{k}_grade']})" for k in (1, 2, 3)
                     if isinstance(row.get(f"strength_{k}"), str)]
        if strengths:
            st.write("**Strongest areas:** " + ", ".join(strengths))
        for cat, title in (("modifiable", "Things a county can act on"), ("structural", "Things that move over years")):
            part = graded[graded["category"] == cat].sort_values(["theme", "score_percentile"])
            if part.empty:
                continue
            st.markdown(f"#### {title}")
            tab = pd.DataFrame({
                "Area": part["theme"], "Measure": part["label"],
                "This county": [fmt_value(ft, v, g.loc[ft] if ft in g.index else None) for ft, v in zip(part["feature"], part["value"])],
                "Similar counties": [fmt_value(ft, v, g.loc[ft] if ft in g.index else None) for ft, v in zip(part["feature"], part["peer_median"])],
                "Better than % of counties": part["score_percentile"].round(0),
                "Grade": part["grade"], "Against similar counties": part["vs_peers"].str.capitalize(),
            })
            st.dataframe(
                tab.style.map(lambda v: f"background-color:{GRADE_COLOR.get(v, '')};color:#222;font-weight:700"
                              if v in GRADE_COLOR else "", subset=["Grade"]),
                hide_index=True, width="stretch",
                column_config={"Better than % of counties": st.column_config.ProgressColumn(
                    min_value=0, max_value=100, format="%d")})
        missing = int(row.get("features_without_data", 0))
        if missing:
            st.caption(f"{missing} measures have no data for this county and are not graded.")

    # ---- profile (context, not graded)
    with t_profile:
        st.caption("Who lives here and what kind of place it is. These are not graded. A county is not better "
                   "or worse for being older, more rural or more diverse. The band shows where it sits nationally.")
        ctx = f[f["grade"] == ""].sort_values(["theme", "label"])
        tab = pd.DataFrame({
            "Area": ctx["theme"], "Measure": ctx["label"],
            "This county": [fmt_value(ft, v, g.loc[ft] if ft in g.index else None) for ft, v in zip(ctx["feature"], ctx["value"])],
            "National percentile": ctx["national_percentile"].round(0), "Band": ctx["band"],
            "Similar counties": [fmt_value(ft, v, g.loc[ft] if ft in g.index else None) for ft, v in zip(ctx["feature"], ctx["peer_median"])],
            "National median": [fmt_value(ft, v, g.loc[ft] if ft in g.index else None) for ft, v in zip(ctx["feature"], ctx["national_median"])],
        })
        st.dataframe(tab, hide_index=True, width="stretch",
                     column_config={"National percentile": st.column_config.ProgressColumn(
                         min_value=0, max_value=100, format="%d")})

    # ---- peers
    with t_peers:
        p = peers[peers["fips"] == fips].copy()
        st.caption("The 25 counties most like this one on things it cannot change: size, rurality, age, race "
                   "and ethnicity, and migration. They are the fair comparison.")
        both = pd.concat([pd.DataFrame({"place": p["peer_place"], "le50": p["peer_le50"], "who": "Similar county"}),
                          pd.DataFrame({"place": [row["place"]], "le50": [row["le50"]], "who": ["This county"]})])
        both = both.sort_values("le50")
        fig = go.Figure(go.Bar(x=both["le50"], y=both["place"], orientation="h",
                               marker_color=np.where(both["who"] == "This county", ACCENT, "#CBD5E1"),
                               hovertemplate="%{y}<br>%{x:.1f} years<extra></extra>"))
        fig.add_vline(x=row["national_median_le50"], line_dash="dash", line_color="#666",
                      annotation_text="national median", annotation_position="top")
        lo = float(both["le50"].min()) - 1
        fig.update_layout(height=26 * len(both) + 90, margin=dict(l=10, r=20, t=30, b=40),
                          xaxis=dict(title="Life expectancy at 50 (years)", range=[lo, float(both["le50"].max()) + 1]))
        ev = st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        pick = st.selectbox("Open one of them", ["-"] + p.sort_values("peer_rank")["peer_place"].tolist(),
                            key=f"peerpick_{fips}")
        if pick != "-":
            st.session_state.fips = p.loc[p["peer_place"] == pick, "peer_fips"].iloc[0]
            st.rerun()                                # full rerun reopens the popup on that county

    # ---- drivers
    with t_drivers:
        sh = f[f["shap_years"].notna()].copy()
        if sh.empty:
            st.write("Not available.")
        else:
            st.caption("How each characteristic moves the model's estimate for this county away from the national "
                       "average, in years. Green pushes it up, red pulls it down. This describes the model, not causes.")
            top = sh.reindex(sh["shap_years"].abs().sort_values(ascending=False).index).head(14).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["shap_years"], y=top["label"], orientation="h",
                                   marker_color=np.where(top["shap_years"] >= 0, GREEN, RED),
                                   customdata=top["category"],
                                   hovertemplate="%{y}<br>%{x:+.2f} years (%{customdata})<extra></extra>"))
            fig.add_vline(x=0, line_color="#444", line_width=1)
            fig.update_layout(height=30 * len(top) + 90, margin=dict(l=10, r=20, t=20, b=40),
                              xaxis_title="Years added to or taken from the estimate")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --------------------------------------------------------------------------- map
def poly_table(geo, C: pd.DataFrame) -> pd.DataFrame:
    """One row per polygon. Counties IHME reports as one unit all point at that unit."""
    unit_of = {}
    for fips, members in zip(C["fips"], C["member_fips"].fillna("")):
        for mfips in (members.split(";") if members else [fips]):
            unit_of[mfips] = fips
        unit_of[fips] = fips
    ids = [ft["id"] for ft in geo["features"]]
    return pd.DataFrame({"poly": ids, "fips": [unit_of.get(i) for i in ids]})


@st.cache_data(show_spinner=False)
def poly_bounds(folder: str, inset: bool) -> pd.DataFrame:
    """Bounding box of every county shape, used to zoom the map to the chosen states."""
    geo = load_geo_inset(folder) if inset else load_geo(folder)
    rows = []
    for ft in geo["features"]:
        geom = ft["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else [r for poly in geom["coordinates"] for r in poly]
        xy = np.array([pt for ring in rings for pt in ring], dtype=float)
        xy = xy[xy[:, 0] < 0] if (xy[:, 0] < 0).any() else xy      # Aleutians cross the date line
        rows.append((ft["id"], xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()))
    return pd.DataFrame(rows, columns=["poly", "x0", "x1", "y0", "y1"])


def view_for(bounds: pd.DataFrame, state_abbrs: list[str]):
    """Centre and zoom that fit the chosen states; the lower 48 when none are chosen."""
    if not state_abbrs:
        return dict(lat=37.3, lon=-96.5), 3.42
    codes = {k for k, v in FIPS_OF_STATE.items() if v in state_abbrs}
    b = bounds[bounds["poly"].str[:2].isin(codes)]
    if b.empty:
        return dict(lat=37.3, lon=-96.5), 3.42
    x0, x1, y0, y1 = b["x0"].min(), b["x1"].max(), b["y0"].min(), b["y1"].max()
    lat = (y0 + y1) / 2
    # Largest zoom at which the states still fit a map about 850 x 500 pixels
    # (a web-map tile is 512 pixels wide at zoom 0 and doubles with each level)
    z_lon = np.log2(850 * 360 / (512 * max(x1 - x0, 0.3)))
    z_lat = np.log2(500 * 360 / (512 * max(y1 - y0, 0.3) / np.cos(np.radians(lat))))
    return dict(lat=lat, lon=(x0 + x1) / 2), float(np.clip(min(z_lon, z_lat), 2.2, 9))


@st.cache_resource(show_spinner=False, max_entries=24)
def cached_map(folder: str, basis: str, shown: tuple, zoom_states: tuple, basemap: bool):
    """The finished map for one combination of settings. Building it takes about a second,
    so it is kept and reused; opening a county then costs nothing on the map side."""
    C = with_status(load_pack(folder)[0], basis)
    geo = load_geo(folder) if basemap else load_geo_inset(folder)
    return draw_map(geo, poly_bounds(folder, not basemap), C, set(shown), basis, list(zoom_states), basemap)


def draw_map(geo, bounds, C: pd.DataFrame, shown: set, basis: str, zoom_states: list[str], basemap: bool):
    polys = poly_table(geo, C).merge(
        C[["fips", "place", "state", "le50", "le50_national_rank", "le50_grade", "status", "status_diff"]],
        on="fips", how="left")
    polys["layer"] = np.where(polys["fips"].isna(), "No data",
                              np.where(polys["fips"].isin(shown), polys["status"], "Not in filter"))
    order = ["Green", "Yellow", "Red", "Not in filter", "No data"]
    polys["layer"] = pd.Categorical(polys["layer"], order)
    polys = polys.sort_values("layer")
    ref = "national median" if basis == "nation" else "similar counties"
    center, zoom = view_for(bounds, zoom_states)
    fig = px.choropleth_map(
        polys, geojson=geo, locations="poly", color="layer",
        color_discrete_map={**STATUS_COLOR, "Not in filter": FAINT, "No data": GREY},
        category_orders={"layer": order}, center=center, zoom=zoom,
        map_style="carto-positron" if basemap else "white-bg", opacity=0.8 if basemap else 1.0,
        custom_data=["fips", "place", "le50", "le50_national_rank", "le50_grade", "status_diff"])
    fig.update_traces(
        marker_line_width=0.4, marker_line_color="white",
        hovertemplate=("<b>%{customdata[1]}</b><br>%{customdata[2]:.1f} years at 50"
                       "<br>%{customdata[5]:+.1f} vs " + ref +
                       "<br>Grade %{customdata[4]}, rank %{customdata[3]:,}<extra></extra>"),
        selector=lambda t: t.name in STATUS_COLOR)
    fig.update_traces(hoverinfo="skip", hovertemplate=None, selector=lambda t: t.name not in STATUS_COLOR)
    fig.update_traces(unselected=dict(marker=dict(opacity=0.8 if basemap else 1.0)))
    fig.update_traces(marker_line_width=0.4, marker_line_color="white",
                      selector=lambda t: t.name not in STATUS_COLOR)
    fig.update_layout(height=570, margin=dict(l=0, r=0, t=0, b=0), clickmode="event+select", showlegend=False)
    return fig


FIPS_OF_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


# --------------------------------------------------------------------------- popup
def close_county():
    """Runs when the popup is closed. Bumping view_key clears the map and list selections,
    so the same county can be clicked again."""
    st.session_state.fips = None
    st.session_state.view_key += 1


def open_county(fips: str, source: tuple):
    """Open a county once per click. `source` identifies the click so that a selection
    still sitting in the map or a list does not reopen the popup on a later rerun."""
    if st.session_state.get("last_click") != source:
        st.session_state.last_click = source
        st.session_state.fips = fips          # the popup is drawn at the end of this same run


def show_county_popup(C, feats, scen, peers, guide, summary, basis):
    fips = st.session_state.fips
    place = C.loc[C["fips"] == fips, "place"].iloc[0]
    popup = st.dialog(place, width="large", on_dismiss=close_county)(county_page)
    popup(fips, C, feats, scen, peers, guide, summary, basis)


# --------------------------------------------------------------------------- filters
FILTER_DEFAULTS = {"f_state": [], "f_status": [], "f_place": [], "f_pop": [], "f_grade": [], "f_expect": [],
                   "f_opp": "Any", "f_gain": 0.0, "f_name": ""}


def reset_filters():
    for key, default in FILTER_DEFAULTS.items():
        st.session_state[key] = copy.copy(default)


# --------------------------------------------------------------------------- app
def main():
    data_dir = find_data_dir()
    if data_dir is None:
        st.error("Could not find app_counties.csv. Put app.py in the folder that holds the app_*.csv files "
                 "(the `app` folder written by section 12b of the Modeling notebook), or set the "
                 "LE50_APP_DATA environment variable to that folder.")
        st.stop()
    counties, feats, scen, peers, guide, summary = load_pack(str(data_dir))

    st.session_state.setdefault("fips", None)
    st.session_state.setdefault("view_key", 0)

    # ---- sidebar
    st.markdown(CSS, unsafe_allow_html=True)
    sb = st.sidebar
    sb.markdown("<div class='le-brand'>Life After <span>50</span></div>", unsafe_allow_html=True)
    sb.caption("How long a 50 year old can expect to live, by county, 2019.")
    jump = sb.selectbox("Go to a county", ["-"] + counties.sort_values(["state", "county_name"])["place"].tolist(),
                        key=f"jump_{st.session_state.view_key}")
    if jump != "-":
        open_county(counties.loc[counties["place"] == jump, "fips"].iloc[0],
                    ("jump", st.session_state.view_key, jump))
    basis = sb.radio("Color counties against", ["nation", "peers"],
                     format_func=lambda k: {"nation": "The national median", "peers": "Similar counties"}[k],
                     help="Similar counties are the 25 most like each county in size, rurality, age, "
                          "race and ethnicity, and migration.")
    basemap = sb.checkbox("Show towns and roads on the map", value=False,
                          help="Draws a street map under the counties. Needs an internet connection.")
    C = with_status(counties, basis)

    for key, default in FILTER_DEFAULTS.items():
        st.session_state.setdefault(key, copy.copy(default))
    n_active = sum(st.session_state[k] != d for k, d in FILTER_DEFAULTS.items())
    sb.divider()
    head, reset = sb.columns([1.25, 1])
    head.markdown(f"#### Filters{f' · {n_active}' if n_active else ''}")
    reset.button("Reset all", on_click=reset_filters, disabled=n_active == 0, width="stretch")
    f_state = sb.multiselect("State", sorted(C["state_name"].unique()), key="f_state")
    f_status = sb.multiselect("Status", ["Green", "Yellow", "Red"], key="f_status",
                              format_func=lambda s_: f"{STATUS_DOT[s_]} {s_}")
    f_place = sb.multiselect("Kind of place", [p for p in PLACE_TYPES if p in set(C["place_type"])], key="f_place")
    f_pop = sb.multiselect("Population", POP_LABELS, key="f_pop")
    f_grade = sb.multiselect("Life expectancy grade", list("ABCDF"), key="f_grade")
    f_expect = sb.multiselect("Against what the model expects", sorted(C["vs_expectation"].dropna().unique()),
                              key="f_expect")
    opps = sorted(C["opportunity_1"].dropna().unique())
    f_opp = sb.selectbox("Biggest thing it could change", ["Any"] + opps, key="f_opp")
    f_gain = sb.slider("Possible gain of at least (years)", 0.0, 3.0, step=0.25, key="f_gain",
                       help="Years associated with reaching the top quarter of similar counties.")
    f_name = sb.text_input("County name contains", key="f_name")

    mask = pd.Series(True, index=C.index)
    if f_state:
        mask &= C["state_name"].isin(f_state)
    if f_status:
        mask &= C["status"].isin(f_status)
    if f_place:
        mask &= C["place_type"].isin(f_place)
    if f_pop:
        mask &= C["pop_group"].isin(f_pop)
    if f_grade:
        mask &= C["le50_grade"].isin(f_grade)
    if f_expect:
        mask &= C["vs_expectation"].isin(f_expect)
    if f_opp != "Any":
        mask &= C["opportunity_1"] == f_opp
    if f_gain > 0:
        mask &= C["gain_peer_top_quartile"] >= f_gain
    if f_name.strip():
        mask &= C["place"].str.contains(f_name.strip(), case=False, regex=False)
    F = C[mask]

    sb.divider()
    sb.caption(summary.get("caveat", ""))

    # ---- overview
    ref = "the national median" if basis == "nation" else "similar counties"
    st.markdown(
        "<div class='le-eyebrow'>County life expectancy · 2019</div>"
        "<div class='le-title'>Life After 50 by County</div>"
        f"<div class='le-sub'>How long a 50 year old can expect to live in each of {len(C):,} counties, "
        f"compared with {ref}. Click a county to open its page.</div>", unsafe_allow_html=True)
    counts = {s_: int((F["status"] == s_).sum()) for s_ in ("Green", "Yellow", "Red")}
    what = {"Green": f"more than {BAND:.0f} year above", "Yellow": f"within {BAND:.0f} year",
            "Red": f"more than {BAND:.0f} year below"}
    cards = [kpi("Counties shown", f"{len(F):,}", f"of {len(C):,}")]
    cards += [kpi(s_, f"{counts[s_]:,}", what[s_], STATUS_COLOR[s_]) for s_ in counts]
    cards += [kpi("Median at 50", f"{F['le50'].median():.1f} yrs" if len(F) else "n/a",
                  f"national median {C['national_median_le50'].iloc[0]:.1f}")]
    st.markdown(f"<div class='le-kpis'>{''.join(cards)}</div>", unsafe_allow_html=True)

    tab_map, tab_list = st.tabs(["Map", "List"])

    with tab_map:
        geo = load_geo(str(data_dir)) if basemap else load_geo_inset(str(data_dir))
        if geo is None:
            st.warning("County boundaries are not available. Copy counties.geojson (saved by the EDA notebook) "
                       "into the data folder, or connect to the internet once so it can be downloaded.")
        else:
            zoom = [s_ for s_ in C.loc[C["state_name"].isin(f_state), "state"].unique()] if f_state else []
            st.markdown(
                "<div class='le-legend'>"
                f"<span>{dot(GREEN)}More than {BAND:.0f} year above</span><span>{dot(YELLOW)}Within {BAND:.0f} year</span>"
                f"<span>{dot(RED)}More than {BAND:.0f} year below</span><span>{dot(FAINT)}Outside the filters</span>"
                f"<span>{dot(GREY)}No data</span></div>", unsafe_allow_html=True)
            fig = cached_map(str(data_dir), basis, tuple(F["fips"]), tuple(zoom), basemap)
            ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                                 key=f"map_{st.session_state.view_key}",
                                 config={"scrollZoom": True, "displayModeBar": False})
            pts = ev.selection.points if ev and getattr(ev, "selection", None) else []
            if pts:
                cd = pts[0].get("customdata") or []
                picked = cd[0] if cd else None
                if isinstance(picked, str) and picked in set(C["fips"]):
                    open_county(picked, ("map", st.session_state.view_key, picked))
            st.caption("Scroll to zoom and drag to move. " +
                       ("Alaska and Hawaii are in their true positions, off to the left."
                        if basemap else "Alaska and Hawaii are drawn at the lower left, and Alaska is shrunk to fit."))

    with tab_list:
        if F.empty:
            st.write("No counties match the filters.")
        sort_by = st.selectbox("Sort counties within each state by",
                               ["Life expectancy, longest first", "Life expectancy, shortest first",
                                "Possible gain, largest first", "County name", "Population, largest first"])
        sort_col, asc = {"Life expectancy, longest first": ("le50", False),
                         "Life expectancy, shortest first": ("le50", True),
                         "Possible gain, largest first": ("gain_peer_top_quartile", False),
                         "County name": ("county_name", True),
                         "Population, largest first": ("population", False)}[sort_by]
        st.caption("Click any cell in a row to open the county.")
        states = sorted(F["state_name"].unique())
        for sname in states:
            part = F[F["state_name"] == sname].sort_values(sort_col, ascending=asc)
            tally = " ".join(f"{STATUS_DOT[s_]} {int((part['status'] == s_).sum())}" for s_ in ("Green", "Yellow", "Red"))
            with st.expander(f"{sname} · {len(part)} counties · median {part['le50'].median():.1f} years · {tally}",
                             expanded=len(states) <= 3):
                gain = part["gain_peer_top_quartile"]
                tab = pd.DataFrame({
                    "": part["status"].map(STATUS_DOT), "County": part["county_name"],
                    "Life expectancy at 50": part["le50"].round(1),
                    "Vs national median": part["years_vs_national_median"].round(1),
                    "Vs similar counties": part["vs_peers_years"].round(1),
                    "Grade": part["le50_grade"], "National rank": part["le50_national_rank"],
                    "Kind of place": part["place_type"], "Population": part["population"],
                    "Biggest thing it could change": part["opportunity_1"],
                    "Possible gain (years)": gain.where(gain >= MIN_GAIN).round(1),
                })
                ev = st.dataframe(tab, hide_index=True, width="stretch", on_select="rerun",
                                  selection_mode=["single-cell", "single-row"],
                                  key=f"list_{sname}_{st.session_state.view_key}",
                                  column_config={"Population": st.column_config.NumberColumn(format="localized"),
                                                 "National rank": st.column_config.NumberColumn(format="localized")})
                sel = ev.selection if ev and getattr(ev, "selection", None) else {}
                cells, rows = sel.get("cells") or [], sel.get("rows") or []
                hit = cells[0][0] if cells else (rows[0] if rows else None)
                if hit is not None:
                    picked = part["fips"].iloc[int(hit)]
                    open_county(picked, ("list", st.session_state.view_key, picked))

    # ---- county popup, drawn over whichever view is showing
    if st.session_state.fips and st.session_state.fips in set(C["fips"]):
        show_county_popup(C, feats, scen, peers, guide, summary, basis)


main()
