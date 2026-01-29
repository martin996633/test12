import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import math
import json
import os

# Pokus o import live dat (pokud existuje fotmob.py)
try:
    from fotmob import get_live_matches, get_match_details
except ImportError:
    def get_live_matches(): return []
    def get_match_details(id): return {}

# --- KONFIGURACE ---
MODEL_FILE = "ultimate_goals_model.json"
FEATURES_FILE = "model_features.pkl"
METADATA_FILE = "model_metadata.json"

# Předpokládáme, že máš exportovaná data z DB do CSV pro potřeby aplikace
DATA_STATS_CSV = "data_stats.csv" 
DATA_ELO_CSV = "data_elo.csv"
DATA_FIFA_CSV = "data_fifa.csv"

st.set_page_config(page_title="AI Goals Predictor PRO", page_icon="⚽", layout="wide")

# --- CSS STYLY ---
st.markdown("""
<style>
    .main-header { font-size: 36px; font-weight: 800; color: #1E88E5; margin-bottom: 20px; }
    .score-board { background-color: #121212; padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 25px; border: 1px solid #333; }
    .team-name { font-size: 26px; font-weight: bold; color: #FFFFFF; }
    .score-digit { font-size: 50px; font-weight: 900; color: #4CAF50; margin: 0 15px; }
    .metric-card { background: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 6px solid #1E88E5; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

# --- POMOCNÉ FUNKCE ---

def normalize_name(name):
    """Musí být identický jako v model.py"""
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
    except Exception as e:
        st.error(f"Chyba při načítání modelu: {e}")
        return None, None

@st.cache_data
def load_static_data():
    """
    Načte CSV soubory a vypočítá Home/Away profily stejně jako model.py.
    """
    try:
        # Pokud nemáme CSV, použijeme dummy data nebo prázdné DF
        if not os.path.exists(DATA_STATS_CSV):
            return [], {}, {}, {}

        stats = pd.read_csv(DATA_STATS_CSV)
        elo = pd.read_csv(DATA_ELO_CSV)
        fifa = pd.read_csv(DATA_FIFA_CSV)
        
        stats['norm_h'] = stats['home_team'].apply(normalize_name)
        stats['norm_a'] = stats['away_team'].apply(normalize_name)
        
        # --- VÝPOČET SPLIT PROFILŮ (Klíčová část) ---
        profiles = {}
        all_teams = set(stats['norm_h'].unique()) | set(stats['norm_a'].unique())
        
        for team in all_teams:
            h_games = stats[stats['norm_h'] == team]
            a_games = stats[stats['norm_a'] == team]
            
            # Defaultní hodnoty pokud není dost dat
            profiles[team] = {
                'h_att': h_games['home_xg'].mean() if len(h_games) > 1 else 1.4,
                'h_def': h_games['away_xg'].mean() if len(h_games) > 1 else 1.2,
                'a_att': a_games['away_xg'].mean() if len(a_games) > 1 else 1.1,
                'a_def': a_games['home_xg'].mean() if len(a_games) > 1 else 1.5
            }
        
        # Elo mapping (poslední známé Elo)
        elo['norm_team'] = elo['team'].apply(normalize_name)
        elo_map = elo.sort_values('valid_from').groupby('norm_team').tail(1).set_index('norm_team')['elo'].to_dict()
        
        # FIFA mapping
        fifa['norm_team'] = fifa['team'].apply(normalize_name)
        fifa_map = fifa.set_index('norm_team')[['attack', 'defence', 'overall']].to_dict('index')
        
        return sorted(list(all_teams)), elo_map, fifa_map, profiles
    except Exception as e:
        st.warning(f"Nepodařilo se načíst statistická data (CSV): {e}")
        return [], {}, {}, {}

# --- INITIALIZACE ---
model, feat_names = load_model_assets()
teams, db_elo, db_fifa, db_profiles = load_static_data()

# --- SIDEBAR: LIVE FEED ---
st.sidebar.title("📡 Live Feed")

if 'live_list' not in st.session_state:
    st.session_state['live_list'] = []

if st.sidebar.button("🔄 Refresh Live Matches"):
    with st.spinner("Stahuji data..."):
        st.session_state['live_list'] = get_live_matches()

if st.session_state['live_list']:
    match_options = {f"{m['home']} - {m['away']} ({m['time']}')": m for m in st.session_state['live_list']}
    selected = st.sidebar.selectbox("Select match:", list(match_options.keys()))
    
    if st.sidebar.button("⚡ Load Match Data"):
        m = match_options[selected]
        d = get_match_details(m['id'])
        
        # Uložení do session state pro automatické vyplnění
        st.session_state['auto_h_team'] = m['home'] # Názvy z FotMobu se musí namapovat ručně v UI
        st.session_state['auto_a_team'] = m['away']
        
        try:
            min_clean = int(str(m['time']).split('+')[0].replace("'", ""))
        except:
            min_clean = 0
            
        st.session_state.update({
            'min': min_clean, 
            'sh': m['score_h'], 
            'sa': m['score_a'],
            'xgh': d.get('xg_h', 0.0), 
            'xga': d.get('xg_a', 0.0),
            'sth': d.get('shots_h', 0), 
            'sta': d.get('shots_a', 0)
        })

# --- HLAVNÍ ROZHRANÍ ---
st.markdown('<div class="main-header">AI Goals Predictor PRO</div>', unsafe_allow_html=True)

# Výběr týmů
col_t1, col_t2 = st.columns(2)
# Zkusíme předvybrat týmy pokud jsou v session state, jinak default
idx_h = 0
idx_a = 1
if 'auto_h_team' in st.session_state and teams:
    # Jednoduchý pokus o fuzzy match nebo jen selectbox, uživatel doladí
    pass 

h_team = col_t1.selectbox("🏠 Home Team", teams, index=idx_h)
a_team = col_t2.selectbox("✈️ Away Team", teams, index=idx_a)

# Vstupy statistik
with st.container():
    st.markdown("### Live Match Statistics")
    c1, c2, c3 = st.columns(3)
    minute = c1.number_input("⏱ Minute", 0, 95, st.session_state.get('min', 45))
    score_h = c2.number_input(f"Score {h_team}", 0, 15, st.session_state.get('sh', 0))
    score_a = c3.number_input(f"Score {a_team}", 0, 15, st.session_state.get('sa', 0))
    
    c4, c5, c6, c7 = st.columns(4)
    xg_h = c4.number_input(f"xG {h_team}", 0.0, 10.0, st.session_state.get('xgh', 0.0), step=0.01)
    shots_h = c5.number_input(f"Shots {h_team}", 0, 50, st.session_state.get('sth', 0))
    xg_a = c6.number_input(f"xG {a_team}", 0.0, 10.0, st.session_state.get('xga', 0.0), step=0.01)
    shots_a = c7.number_input(f"Shots {a_team}", 0, 50, st.session_state.get('sta', 0))

# Scoreboard
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
    if model and feat_names:
        # 1. Příprava dat
        h_n, a_n = normalize_name(h_team), normalize_name(a_team)
        
        # Načtení kontextu (Elo, FIFA, Profily)
        eh = db_elo.get(h_n, 1500)
        ea = db_elo.get(a_n, 1500)
        
        fh = db_fifa.get(h_n, {'attack':75, 'defence':75, 'overall':75})
        fa = db_fifa.get(a_n, {'attack':75, 'defence':75, 'overall':75})
        
        ph = db_profiles.get(h_n, {'h_att': 1.4, 'h_def': 1.2, 'a_att': 1.1, 'a_def': 1.5})
        pa = db_profiles.get(a_n, {'h_att': 1.4, 'h_def': 1.2, 'a_att': 1.1, 'a_def': 1.5})
        
        # 2. Vytvoření DataFrame řádku (Musí přesně sedět s model.py!)
        input_data = {
            'minute': minute,
            'time_remaining': 90 - minute,
            'score_home': score_h,
            'score_away': score_a,
            'goal_diff': score_h - score_a,
            'current_total_goals': score_h + score_a,
            'is_draw': 1 if score_h == score_a else 0,
            
            'xg_home': xg_h,
            'xg_away': xg_a,
            'xg_total': xg_h + xg_a,
            'xg_diff': xg_h - xg_a,
            
            'shots_home': shots_h,
            'shots_away': shots_a,
            
            'efficiency_h': score_h - xg_h,
            'efficiency_a': score_a - xg_a,
            'avg_shot_qual_h': (xg_h / shots_h) if shots_h > 0 else 0,
            
            'elo_home': eh,
            'elo_diff': eh - ea,
            
            'fifa_att_diff': int(fh['attack']) - int(fa['attack']),
            'fifa_def_diff': int(fh['defence']) - int(fa['defence']),
            'squad_qual_diff': int(fh['overall']) - int(fa['overall']),
            
            # NOVÉ FEATURES (HOME/AWAY SPLIT)
            'home_team_home_att': ph['h_att'],
            'home_team_home_def': ph['h_def'],
            'away_team_away_att': pa['a_att'],
            'away_team_away_def': pa['a_def']
        }
        
        df_in = pd.DataFrame([input_data])
        
        # 3. Seřazení sloupců podle tréninku
        try:
            df_in = df_in[feat_names]
            
            # 4. Predikce
            pred_total = model.predict(df_in)[0]
            over_probs, expected_more = calculate_probs(pred_total, score_h + score_a)
            
            # --- VIZUALIZACE VÝSLEDKŮ ---
            st.divider()
            res_c1, res_c2 = st.columns([1, 1.5])
            
            with res_c1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Predicted Final Total")
                st.title(f"{pred_total:.2f}")
                st.write(f"Expected More Goals: **{expected_more:.2f}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.write("")
                st.caption("Betting Probabilities (Poisson)")
                for line, prob in over_probs.items():
                    col_p1, col_p2 = st.columns([1, 3])
                    col_p1.markdown(f"**{line}**")
                    col_p2.progress(int(prob*100))
                    st.caption(f"{prob*100:.1f}%")

            with res_c2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", 
                    value = pred_total, 
                    title = {'text': "Total Goals Expectancy"},
                    gauge = {
                        'axis': {'range': [0, 6]}, 
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 2.5], 'color': "#eeeeee"}, 
                            {'range': [2.5, 4.5], 'color': "#dddddd"}
                        ]
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

        except KeyError as e:
            st.error(f"Nesoulad features mezi modelem a aplikací. Chybí: {e}")
            st.write("Aplikace posílá:", list(df_in.columns))
            st.write("Model očekává:", feat_names)
            
    else:
        st.error("Model nebyl načten. Ujisti se, že existují soubory .json a .pkl.")

# --- METADATA ---
st.write("")
with st.expander("🧠 Model Information & Metadata"):
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                meta = json.load(f)
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Model MAE", meta.get('mae_score', 'N/A'))
            m_c2.metric("Features", meta.get('features_count', 'N/A'))
            m_c3.metric("Snapshot Interval", "1 min")
            m_c4.metric("Last Training", meta.get('training_date', 'N/A'))
        except:
            st.warning("Chyba při čtení metadat.")
    else:
        st.info("Metadata zatím neexistují (spusť trénink).")
