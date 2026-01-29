import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import math
import json
import os

# --- KONFIGURACE SOUBORŮ ---
MODEL_FILENAME = "ultimate_goals_model.ubj"
FEATURES_FILENAME = "model_features.pkl"
METADATA_FILENAME = "model_metadata.json"

# CSV data
STATS_CSV = "data_stats.csv"
ELO_CSV = "data_elo.csv"
FIFA_CSV = "data_fifa.csv"

st.set_page_config(page_title="AI Goals Predictor PRO", page_icon="⚽", layout="wide")

# --- CSS STYLY ---
st.markdown("""
<style>
    .main-header { font-size: 36px; font-weight: 800; color: #1E88E5; margin-bottom: 20px; text-align: center;}
    .score-board { background-color: #121212; padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 25px; border: 1px solid #333; }
    .team-name { font-size: 26px; font-weight: bold; color: #FFFFFF; }
    .score-digit { font-size: 50px; font-weight: 900; color: #4CAF50; margin: 0 15px; }
    .metric-card { background: #f8f9fa; padding: 20px; border-radius: 12px; border-left: 6px solid #1E88E5; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
    .bet-row { padding: 10px; border-bottom: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# --- POMOCNÉ FUNKCE ---

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
    """Vypočítá pravděpodobnosti pro Over i Under."""
    def poisson(k, lamb): return (lamb**k * math.exp(-lamb)) / math.factorial(k)
    
    # Lambda je očekávaný počet ZBÝVAJÍCÍCH gólů
    lamb = max(0.01, predicted_total - current_goals)
    
    # Pravděpodobnost pro přesně 0, 1, 2... dalších gólů
    probs = {i: poisson(i, lamb) for i in range(7)}
    
    # Over: Šance, že padne více než X gólů
    over_probs = {
        f"Over {current_goals + 0.5}": 1.0 - probs[0],
        f"Over {current_goals + 1.5}": 1.0 - (probs[0] + probs[1]),
        f"Over {current_goals + 2.5}": 1.0 - (probs[0] + probs[1] + probs[2])
    }
    
    # Under: Šance, že padne méně než X gólů
    under_probs = {
        f"Under {current_goals + 0.5}": probs[0],                # Padne 0 dalších
        f"Under {current_goals + 1.5}": probs[0] + probs[1],     # Padne 0 nebo 1 další
        f"Under {current_goals + 2.5}": probs[0] + probs[1] + probs[2] # Padne 0, 1 nebo 2 další
    }
    
    return over_probs, under_probs, lamb

# --- HLAVNÍ FIX: NAČÍTÁNÍ MODELU ---
@st.cache_resource
def load_model_assets():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_model = os.path.join(current_dir, MODEL_FILENAME)
        path_features = os.path.join(current_dir, FEATURES_FILENAME)

        if not os.path.exists(path_model):
            st.error(f"❌ CHYBA: Soubor modelu nebyl nalezen: {path_model}")
            return None, None

        m = xgb.XGBRegressor()
        m.load_model(path_model)
        f = joblib.load(path_features)
        return m, f
    except Exception as e:
        st.error(f"❌ Chyba při načítání modelu: {e}")
        return None, None

@st.cache_data
def load_static_data():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_stats = os.path.join(current_dir, STATS_CSV)
        path_elo = os.path.join(current_dir, ELO_CSV)
        path_fifa = os.path.join(current_dir, FIFA_CSV)

        if not os.path.exists(path_stats):
            return [], {}, {}, {}

        stats = pd.read_csv(path_stats)
        elo = pd.read_csv(path_elo)
        fifa = pd.read_csv(path_fifa)
        
        stats['norm_h'] = stats['home_team'].apply(normalize_name)
        stats['norm_a'] = stats['away_team'].apply(normalize_name)
        
        profiles = {}
        all_teams = set(stats['norm_h'].unique()) | set(stats['norm_a'].unique())
        
        for team in all_teams:
            h_games = stats[stats['norm_h'] == team]
            a_games = stats[stats['norm_a'] == team]
            profiles[team] = {
                'h_att': h_games['home_xg'].mean() if len(h_games) > 1 else 1.4,
                'h_def': h_games['away_xg'].mean() if len(h_games) > 1 else 1.2,
                'a_att': a_games['away_xg'].mean() if len(a_games) > 1 else 1.1,
                'a_def': a_games['home_xg'].mean() if len(a_games) > 1 else 1.5
            }
        
        elo['norm_team'] = elo['team'].apply(normalize_name)
        elo_map = elo.sort_values('valid_from').groupby('norm_team').tail(1).set_index('norm_team')['elo'].to_dict()
        fifa['norm_team'] = fifa['team'].apply(normalize_name)
        fifa_map = fifa.set_index('norm_team')[['attack', 'defence', 'overall']].to_dict('index')
        
        return sorted(list(all_teams)), elo_map, fifa_map, profiles
    except Exception as e:
        st.warning(f"⚠️ Chyba při načítání dat: {e}")
        return [], {}, {}, {}

# --- INITIALIZACE ---
model, feat_names = load_model_assets()
teams, db_elo, db_fifa, db_profiles = load_static_data()

# --- HLAVNÍ UI ---
st.markdown('<div class="main-header">🤖 AI Goals Calculator</div>', unsafe_allow_html=True)

# 1. Výběr týmů
col_t1, col_t2 = st.columns(2)
h_team = col_t1.selectbox("🏠 Domácí Tým", teams, index=0)
a_team = col_t2.selectbox("✈️ Hostující Tým", teams, index=1)

# 2. Manuální zadání
st.markdown("### 📝 Zadej aktuální stav")
with st.container():
    c1, c2, c3 = st.columns(3)
    minute = c1.number_input("⏱ Minuta", 0, 95, 0)
    score_h = c2.number_input(f"Góly {h_team}", 0, 15, 0)
    score_a = c3.number_input(f"Góly {a_team}", 0, 15, 0)
    
    c4, c5, c6, c7 = st.columns(4)
    xg_h = c4.number_input(f"xG {h_team}", 0.0, 10.0, 0.0, step=0.01)
    shots_h = c5.number_input(f"Střely {h_team}", 0, 50, 0)
    xg_a = c6.number_input(f"xG {a_team}", 0.0, 10.0, 0.0, step=0.01)
    shots_a = c7.number_input(f"Střely {a_team}", 0, 50, 0)

# 3. Scoreboard
st.markdown(f"""
<div class="score-board">
    <span class="team-name">{h_team}</span>
    <span class="score-digit">{score_h}</span>
    <span class="score-digit">:</span>
    <span class="score-digit">{score_a}</span>
    <span class="team-name">{a_team}</span>
    <div style="margin-top:10px; opacity:0.7;">{minute}. minuta | xG: {xg_h:.2f} - {xg_a:.2f}</div>
</div>
""", unsafe_allow_html=True)

# --- VÝPOČET ---
if st.button("🚀 VYPOČÍTAT PREDIKCI", type="primary", use_container_width=True):
    if model and feat_names:
        h_n, a_n = normalize_name(h_team), normalize_name(a_team)
        
        # Načtení dat
        eh = db_elo.get(h_n, 1500)
        ea = db_elo.get(a_n, 1500)
        fh = db_fifa.get(h_n, {'attack':75, 'defence':75, 'overall':75})
        fa = db_fifa.get(a_n, {'attack':75, 'defence':75, 'overall':75})
        ph = db_profiles.get(h_n, {'h_att': 1.4, 'h_def': 1.2, 'a_att': 1.1, 'a_def': 1.5})
        pa = db_profiles.get(a_n, {'h_att': 1.4, 'h_def': 1.2, 'a_att': 1.1, 'a_def': 1.5})
        
        input_data = {
            'minute': minute, 'time_remaining': 90 - minute,
            'score_home': score_h, 'score_away': score_a,
            'goal_diff': score_h - score_a, 'current_total_goals': score_h + score_a,
            'is_draw': 1 if score_h == score_a else 0,
            'xg_home': xg_h, 'xg_away': xg_a, 'xg_total': xg_h + xg_a, 'xg_diff': xg_h - xg_a,
            'shots_home': shots_h, 'shots_away': shots_a,
            'efficiency_h': score_h - xg_h, 'efficiency_a': score_a - xg_a,
            'avg_shot_qual_h': (xg_h / shots_h) if shots_h > 0 else 0,
            'elo_home': eh, 'elo_diff': eh - ea,
            'fifa_att_diff': int(fh['attack']) - int(fa['attack']),
            'fifa_def_diff': int(fh['defence']) - int(fa['defence']),
            'squad_qual_diff': int(fh['overall']) - int(fa['overall']),
            'home_team_home_att': ph['h_att'], 'home_team_home_def': ph['h_def'],
            'away_team_away_att': pa['a_att'], 'away_team_away_def': pa['a_def']
        }
        
        df_in = pd.DataFrame([input_data])
        
        try:
            df_in = df_in[feat_names]
            pred_total = model.predict(df_in)[0]
            
            # Získání Over i Under
            over_probs, under_probs, expected_more = calculate_probs(pred_total, score_h + score_a)
            
            # --- VIZUALIZACE ---
            st.divider()
            c_res1, c_res2 = st.columns([1, 1.3])
            
            with c_res1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Očekávaný Total")
                st.title(f"{pred_total:.2f}")
                st.write(f"Zbývá gólů: **{expected_more:.2f}**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with c_res2:
                st.markdown("#### 🎲 Sázkové Příležitosti (O/U)")
                
                # Tabulka Over / Under
                current_g = score_h + score_a
                lines = [current_g + 0.5, current_g + 1.5, current_g + 2.5]
                
                for line in lines:
                    o_key = f"Over {line}"
                    u_key = f"Under {line}"
                    
                    o_val = over_probs.get(o_key, 0)
                    u_val = under_probs.get(u_key, 0)
                    
                    # Layout pro řádek
                    row_c1, row_c2, row_c3 = st.columns([1, 1, 1.2])
                    
                    # Zvýraznění vyšší pravděpodobnosti
                    color_o = "green" if o_val > 0.5 else "grey"
                    color_u = "green" if u_val > 0.5 else "grey"
                    
                    row_c1.markdown(f"**⬆️ Over {line}**")
                    row_c1.write(f":{color_o}[{o_val*100:.1f}%]")
                    
                    row_c2.markdown(f"**⬇️ Under {line}**")
                    row_c2.write(f":{color_u}[{u_val*100:.1f}%]")
                    
                    # Vizuální progress bar (poměr sil)
                    row_c3.write("") # Spacer
                    row_c3.progress(int(o_val*100))

        except KeyError as e:
            st.error(f"⚠️ Chyba ve struktuře dat: {e}")
    else:
        st.error("Model není načten.")

# --- FOOTER ---
st.write("")
with st.expander("ℹ️ Informace o modelu"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_meta = os.path.join(current_dir, METADATA_FILENAME)
    if os.path.exists(path_meta):
        try:
            with open(path_meta, "r") as f: meta = json.load(f)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Chyba (MAE)", meta.get('mae_score', 'N/A'))
            m2.metric("Trénováno na", f"{meta.get('training_rows_snapshots', 0)//90} zápasech")
            m3.metric("Snapshot interval", "1 min")
            m4.metric("Poslední update", meta.get('training_date', 'N/A'))
        except: st.text("Metadata nelze přečíst.")
    else: st.info("Metadata nejsou k dispozici.")
