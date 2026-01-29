import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import math
import requests
import json
from datetime import datetime

# Import live dat (pokud soubor existuje)
try:
    from fotmob import get_live_matches, get_match_details
except ImportError:
    def get_live_matches(): return []
    def get_match_details(id): return {}

# --- KONFIGURACE ---
MODEL_FILE = "ultimate_goals_model.json"
FEATURES_FILE = "model_features.pkl"
METADATA_FILE = "model_metadata.json"
DATA_STATS_CSV = "data_stats.csv"
DATA_ELO_CSV = "data_elo.csv"
DATA_FIFA_CSV = "data_fifa.csv"

st.set_page_config(page_title="AI Goals Predictor PRO", page_icon="⚽", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 36px; font-weight: 800; color: #1E88E5; margin-bottom: 20px; }
    .score-board { background-color: #121212; padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 25px; border: 1px solid #333; }
    .team-name { font-size: 26px; font-weight: bold; color: #FFFFFF; }
    .score-digit { font-size: 50px; font-weight: 900; color: #4CAF50; margin: 0 15px; }
    .metric-card { background: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 6px solid #1E88E5; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .prob-text { font-size: 18px; font-weight: 600; color: #455A64; }
</style>
""", unsafe_allow_html=True)

# --- POMOCNÉ FUNKCE (Mapping & Pravděpodobnost) ---
def normalize_name(name):
    if name is None: return ""
    name = str(name).lower().strip()
    mapping = {
        "man city": "man city", "manchester city": "man city", "man utd": "man united", "manchester united": "man united", 
        "nott'm forest": "forest", "nottingham forest": "forest", "spurs": "tottenham", "tottenham hotspur": "tottenham",
        "wolves": "wolverhampton wanderers", "newcastle": "newcastle united", "brighton": "brighton & hove albion",
        "west ham": "west ham united", "sheffield utd": "sheffield united", "leicester": "leicester city", "leeds": "leeds united", "luton": "luton town",
        "bayern munich": "bayern", "fc bayern münchen": "bayern", "bayer leverkusen": "leverkusen", "bayer 04 leverkusen": "leverkusen",
        "borussia dortmund": "dortmund", "borussia m.gladbach": "gladbach", "borussia mönchengladbach": "gladbach",
        "eintracht frankfurt": "frankfurt", "rasenballsport leipzig": "rb leipzig", "rb leipzig": "rb leipzig",
        "fc cologne": "koeln", "1. fc köln": "koeln", "mainz 05": "mainz", "1. fsv mainz 05": "mainz",
        "st. pauli": "st pauli", "fc st. pauli": "st pauli", "vfb stuttgart": "stuttgart",
        "werder bremen": "werder", "sv werder bremen": "werder", "wolfsburg": "wolfsburg", "vfl wolfsburg": "wolfsburg",
        "augsburg": "augsburg", "fc augsburg": "augsburg", "hoffenheim": "hoffenheim", "tsg 1899 hoffenheim": "hoffenheim",
        "union berlin": "union berlin", "1. fc union berlin": "union berlin", "bochum": "bochum", "vfl bochum 1848": "bochum",
        "fc heidenheim": "heidenheim", "1. fc heidenheim 1846": "heidenheim", "freiburg": "freiburg", "sc freiburg": "freiburg",
        "hamburger sv": "hamburg", "atletico madrid": "atletico", "atlético madrid": "atletico", "athletic club": "athletic", 
        "barcelona": "barcelona", "fc barcelona": "barcelona", "real sociedad": "sociedad", "sevilla": "sevilla", 
        "valencia": "valencia", "villarreal": "villarreal", "getafe": "getafe", "ac milan": "milan", "inter": "inter", 
        "juventus": "juventus", "roma": "roma", "napoli": "napoli", "atalanta": "atalanta"
    }
    return mapping.get(name, name)

def calculate_probs(predicted_total, current_goals):
    def poisson(k, lamb): return (lamb**k * math.exp(-lamb)) / math.factorial(k)
    lamb = max(0.01, predicted_total - current_goals)
    probs = {i: poisson(i, lamb) for i in range(7)}
    over_probs = {
        f"Over {current_goals + 0.5}": 1.0 - probs[0],
        f"Over {current_goals + 1.5}": 1.0 - (probs[0] + probs[1]),
        f"Over {current_goals + 2.5}": 1.0 - (probs[0] + probs[1] + probs[2])
    }
    return over_probs, lamb

# --- NAČÍTÁNÍ ZDROJŮ ---
@st.cache_resource
def load_model_assets():
    try:
        m = xgb.XGBRegressor()
        m.load_model(MODEL_FILE)
        f = joblib.load(FEATURES_FILE)
        return m, f
    except: return None, None

@st.cache_data
def load_static_data():
    try:
        stats = pd.read_csv(DATA_STATS_CSV)
        elo = pd.read_csv(DATA_ELO_CSV)
        fifa = pd.read_csv(DATA_FIFA_CSV)
        
        stats['norm_h'] = stats['home_team'].apply(normalize_name)
        stats['norm_a'] = stats['away_team'].apply(normalize_name)
        
        # Home/Away Split Profily
        profiles = {}
        all_teams = set(stats['norm_h']) | set(stats['norm_a'])
        for team in all_teams:
            h_games = stats[stats['norm_h'] == team]
            a_games = stats[stats['norm_a'] == team]
            profiles[team] = {
                'h_att': h_games['home_xg'].mean() if not h_games.empty else 1.4,
                'h_def': h_games['away_xg'].mean() if not h_games.empty else 1.2,
                'a_att': a_games['away_xg'].mean() if not a_games.empty else 1.1,
                'a_def': a_games['home_xg'].mean() if not a_games.empty else 1.5
            }
        
        elo_map = elo.sort_values('valid_from').groupby('team').tail(1).set_index('team')['elo'].to_dict()
        fifa_map = fifa.set_index('team')[['attack', 'defence', 'overall']].to_dict('index')
        
        return sorted(list(all_teams)), elo_map, fifa_map, profiles
    except: return [], {}, {}, {}

# --- LOGIKA LIVE DAT ---
model, feat_names = load_model_assets()
teams, db_elo, db_fifa, db_profiles = load_static_data()

st.sidebar.title("📡 Live Feed")
if st.sidebar.button("🔄 Refresh Live Matches"):
    st.session_state['live_list'] = get_live_matches()

if 'live_list' in st.session_state and st.session_state['live_list']:
    match_options = {f"{m['home']} - {m['away']} ({m['time']}')": m for m in st.session_state['live_list']}
    selected = st.sidebar.selectbox("Select match:", list(match_options.keys()))
    if st.sidebar.button("⚡ Load Match Data"):
        m = match_options[selected]
        d = get_match_details(m['id'])
        st.session_state.update({
            'min': int(m['time'].split('+')[0]), 'sh': m['score_h'], 'sa': m['score_a'],
            'xgh': d.get('xg_h', 0.0), 'xga': d.get('xg_a', 0.0),
            'sth': d.get('shots_h', 0), 'sta': d.get('shots_a', 0)
        })

# --- HLAVNÍ FORMULÁŘ ---
st.markdown('<div class="main-header">AI Goals Predictor PRO</div>', unsafe_allow_html=True)

col_t1, col_t2 = st.columns(2)
h_team = col_t1.selectbox("🏠 Home Team", teams, index=0)
a_team = col_t2.selectbox("✈️ Away Team", teams, index=1)

with st.container():
    st.markdown("### Live Match Statistics")
    c1, c2, c3 = st.columns(3)
    minute = c1.number_input("⏱ Minute", 0, 90, st.session_state.get('min', 45))
    score_h = c2.number_input(f"Score {h_team}", 0, 15, st.session_state.get('sh', 0))
    score_a = c3.number_input(f"Score {a_team}", 0, 15, st.session_state.get('sa', 0))
    
    c4, c5, c6, c7 = st.columns(4)
    xg_h = c4.number_input(f"xG {h_team}", 0.0, 10.0, st.session_state.get('xgh', 0.0), step=0.1)
    shots_h = c5.number_input(f"Shots {h_team}", 0, 50, st.session_state.get('sth', 0))
    xg_a = c6.number_input(f"xG {a_team}", 0.0, 10.0, st.session_state.get('xga', 0.0), step=0.1)
    shots_a = c7.number_input(f"Shots {a_team}", 0, 50, st.session_state.get('sta', 0))

st.markdown(f"""
<div class="score-board">
    <span class="team-name">{h_team}</span>
    <span class="score-digit">{score_h}</span>
    <span class="score-digit">:</span>
    <span class="score-digit">{score_a}</span>
    <span class="team-name">{a_team}</span>
    <div style="margin-top:10px; opacity:0.7;">Minute: {minute}' | xG: {xg_h:.2f} - {xg_a:.2f}</div>
</div>
""", unsafe_allow_html=True)

# --- PREDIKCE ---
if st.button("🔮 RUN AI PREDICTION", type="primary", use_container_width=True):
    if model:
        h_n, a_n = normalize_name(h_team), normalize_name(a_team)
        ph, pa = db_profiles.get(h_n, {}), db_profiles.get(a_n, {})
        eh, ea = db_elo.get(h_n, 1500), db_elo.get(a_n, 1500)
        fh, fa = db_fifa.get(h_n, {'attack':75, 'defence':75, 'overall':75}), db_fifa.get(a_n, {'attack':75, 'defence':75, 'overall':75})
        
        in_data = pd.DataFrame([{
            'minute': minute, 'time_remaining': 90-minute, 'score_home': score_h, 'score_away': score_a,
            'goal_diff': score_h-score_a, 'current_total_goals': score_h+score_a,
            'xg_home': xg_h, 'xg_away': xg_a, 'xg_total': xg_h+xg_a, 'xg_diff': xg_h-xg_a,
            'shots_home': shots_h, 'shots_away': shots_a, 'efficiency_h': score_h-xg_h, 'efficiency_a': score_a-xg_a,
            'elo_home': eh, 'elo_diff': eh-ea, 'fifa_att_diff': fh['attack']-fa['attack'], 'squad_qual_diff': fh['overall']-fa['overall'],
            'home_team_home_att': ph.get('h_att', 1.3), 'home_team_home_def': ph.get('h_def', 1.2),
            'away_team_away_att': pa.get('a_att', 1.1), 'away_team_away_def': pa.get('a_def', 1.4),
            'avg_shot_qual_h': (xg_h/shots_h) if shots_h>0 else 0
        }])
        
        pred_total = model.predict(in_data[feat_names])[0]
        over_probs, expected_more = calculate_probs(pred_total, score_h+score_a)
        
        # --- VÝSLEDKY ---
        st.divider()
        res_c1, res_c2 = st.columns([1, 1.5])
        
        with res_c1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Predicted Total")
            st.title(f"{pred_total:.2f}")
            st.write(f"Remaining Expected Goals: **{expected_more:.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.write("")
            for line, prob in over_probs.items():
                col_p1, col_p2 = st.columns([1, 3])
                col_p1.markdown(f"**{line}**")
                col_p2.progress(int(prob*100))
                st.caption(f"Probability: {prob*100:.1f}%")

        with res_c2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred_total, title = {'text': "Total Goals Expectancy"},
                gauge = {'axis': {'range': [0, 6]}, 'bar': {'color': "#1E88E5"},
                         'steps': [{'range': [0, 2.5], 'color': "#eeeeee"}, {'range': [2.5, 4.5], 'color': "#dddddd"}]}
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

# --- METADATA ---
st.write("")
with st.expander("🧠 Model Information & Statistics"):
    try:
        with open(METADATA_FILE, "r") as f:
            meta = json.load(f)
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Model MAE", meta['mae'])
        m_c2.metric("Train Matches", f"~{meta['train_rows']//90}")
        m_c3.metric("Test Matches", f"~{meta['test_rows']//90}")
        m_c4.metric("Features", meta['features'])
        st.caption(f"Last training: {meta['date']} | Data interval: 1 minute snapshot")
    except:
        st.info("No metadata available. Run training script first.")
