import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import math

# --- KONFIGURACE ---
MODEL_FILE = "ultimate_goals_model.json"
FEATURES_FILE = "model_features.pkl"
DATA_STATS_CSV = "data_stats.csv"
DATA_ELO_CSV = "data_elo.csv"
DATA_FIFA_CSV = "data_fifa.csv"

st.set_page_config(page_title="ProBet AI Predictor", page_icon="⚽", layout="wide")

# --- CUSTOM CSS (Pro hezčí vzhled) ---
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .score-board { background-color: #1E1E1E; padding: 20px; border-radius: 10px; text-align: center; color: white; margin-bottom: 20px;}
    .team-name { font-size: 28px; font-weight: bold; color: #E0E0E0; }
    .score { font-size: 48px; font-weight: 800; color: #4CAF50; margin: 0 20px; }
    .meta-info { color: #B0BEC5; font-size: 14px; }
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- MAPPING FUNKCE ---
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
        "hamburger sv": "hamburg",
        "atletico madrid": "atletico", "atlético madrid": "atletico", "athletic club": "athletic", "athletic club de bilbao": "athletic",
        "barcelona": "barcelona", "fc barcelona": "barcelona", "real betis": "betis", "real betis balompié": "betis",
        "celta vigo": "celta", "rc celta": "celta", "real sociedad": "sociedad", "real oviedo": "oviedo",
        "alaves": "alaves", "deportivo alavés": "alaves", "girona": "girona", "girona fc": "girona",
        "mallorca": "mallorca", "rcd mallorca": "mallorca", "osasuna": "osasuna", "ca osasuna": "osasuna",
        "sevilla": "sevilla", "sevilla fc": "sevilla", "valencia": "valencia", "valencia cf": "valencia",
        "villarreal": "villarreal", "villarreal cf": "villarreal", "getafe": "getafe", "getafe cf": "getafe",
        "espanyol": "espanyol", "rcd espanyol": "espanyol", "cadiz": "cadiz", "cádiz cf": "cadiz",
        "almeria": "almeria", "ud almería": "almeria", "elche": "elche", "elche cf": "elche",
        "valladolid": "valladolid", "real valladolid cf": "valladolid",
        "paris saint germain": "paris sg", "paris saint-germain": "paris sg", "marseille": "marseille", "olympique de marseille": "marseille",
        "lyon": "lyon", "olympique lyonnais": "lyon", "lille": "lille", "lille osc": "lille",
        "monaco": "monaco", "as monaco": "monaco", "nice": "nice", "ogc nice": "nice",
        "rennes": "rennes", "stade rennais fc": "rennes", "lens": "lens", "rc lens": "lens",
        "strasbourg": "strasbourg", "rc strasbourg alsace": "strasbourg", "toulouse": "toulouse", "toulouse fc": "toulouse",
        "nantes": "nantes", "fc nantes": "nantes", "reims": "reims", "stade de reims": "reims",
        "montpellier": "montpellier", "montpellier hsc": "montpellier", "lorient": "lorient", "fc lorient": "lorient",
        "metz": "metz", "fc metz": "metz", "brest": "brest", "stade brestois 29": "brest",
        "le havre": "le havre", "le havre ac": "le havre", "auxerre": "auxerre", "aj auxerre": "auxerre",
        "angers": "angers", "angers sco": "angers",
        "ac milan": "milan", "inter": "inter", "juventus": "juventus", "roma": "roma", "lazio": "lazio",
        "napoli": "napoli", "atalanta": "atalanta", "fiorentina": "fiorentina", "bologna": "bologna",
        "torino": "torino", "udinese": "udinese", "empoli": "empoli", "verona": "verona", "hellas verona": "verona",
        "hellas verona fc": "verona", "lecce": "lecce", "salernitana": "salernitana", "monza": "monza",
        "sassuolo": "sassuolo", "frosinone": "frosinone", "genoa": "genoa", "cagliari": "cagliari",
        "parma calcio 1913": "parma", "parma": "parma", "como": "como", "venezia": "venezia"
    }
    if name in mapping: return mapping[name]
    for key, value in mapping.items():
        if key in name: return value
    return name

# --- NAČÍTÁNÍ DAT ---
@st.cache_resource
def load_stuff():
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
        return model, features
    except: return None, None

@st.cache_data
def load_csv():
    try:
        stats = pd.read_csv(DATA_STATS_CSV)
        elo = pd.read_csv(DATA_ELO_CSV)
        fifa = pd.read_csv(DATA_FIFA_CSV)
        stats['norm_home'] = stats['home_team'].apply(normalize_name)
        stats['norm_away'] = stats['away_team'].apply(normalize_name)
        elo['norm_team'] = elo['team'].apply(normalize_name)
        fifa['norm_team'] = fifa['team'].apply(normalize_name)
        
        profiles = {}
        all_teams = set(stats['norm_home'].unique()) | set(stats['norm_away'].unique())
        for team in all_teams:
            h = stats[stats['norm_home'] == team]
            a = stats[stats['norm_away'] == team]
            tot = len(h)+len(a)
            if tot<3: continue
            ppda = (h['home_ppda'].mean() + a['away_ppda'].mean())/2 if 'home_ppda' in stats.columns else 10
            deep = (h['home_deep_completions'].mean() + a['away_deep_completions'].mean())/2 if 'home_deep_completions' in stats.columns else 5
            xg = (h['home_xg'].sum() + a['away_xg'].sum())/tot
            profiles[team] = {'avg_xg': xg, 'ppda': ppda, 'deep': deep}
            
        latest_elo = elo.sort_values('valid_from').groupby('norm_team').tail(1).set_index('norm_team')['elo'].to_dict()
        fifa_map = fifa.set_index('norm_team')[['attack','defence','overall']].to_dict('index')
        return sorted(list(all_teams)), latest_elo, fifa_map, profiles
    except: return [], {}, {}, {}

# --- POISSON CALCULATOR (Pro sázkové pravděpodobnosti) ---
def poisson_probability(k, lamb):
    return (lamb**k * math.exp(-lamb)) / math.factorial(k)

def calculate_probs(predicted_total, current_goals):
    remaining_lambda = max(0.01, predicted_total - current_goals)
    
    # Šance na přesně 0, 1, 2... dalších gólů
    probs = {}
    for i in range(6):
        probs[i] = poisson_probability(i, remaining_lambda)
    
    # Cumulative probabilities (Over lines)
    over_probs = {
        f"Over {current_goals + 0.5}": 1.0 - probs[0], # Padne aspoň 1
        f"Over {current_goals + 1.5}": 1.0 - (probs[0] + probs[1]), # Padnou aspoň 2
        f"Over {current_goals + 2.5}": 1.0 - (probs[0] + probs[1] + probs[2]) # Padnou aspoň 3
    }
    return over_probs, remaining_lambda

# --- UI START ---
model, feat_names = load_stuff()
teams, db_elo, db_fifa, db_profiles = load_csv()

# 1. SETUP ZÁPASU (Sidebar pro čistší vzhled)
with st.sidebar:
    st.header("⚙️ Nastavení Zápasu")
    h_team = st.selectbox("Domácí", teams, index=0)
    a_team = st.selectbox("Hosté", teams, index=1)
    
    h_norm, a_norm = normalize_name(h_team), normalize_name(a_team)
    h_e, a_e = db_elo.get(h_norm, 1500), db_elo.get(a_norm, 1500)
    f_h = db_fifa.get(h_norm, {'attack': 75, 'defence': 75, 'overall': 75})
    f_a = db_fifa.get(a_norm, {'attack': 75, 'defence': 75, 'overall': 75})

    st.divider()
    st.info(f"**Elo Strength:**\n{h_team}: {int(h_e)}\n{a_team}: {int(a_e)}")

# 2. SCOREBOARD HEADER
col_min, col_sc, col_stats = st.columns([1, 4, 1])

# Vstupy (Umístíme je trochu elegantněji)
st.markdown("### 📝 Live Statistiky")
r1_c1, r1_c2, r1_c3 = st.columns([1, 1, 1])
with r1_c1:
    minute = st.number_input("⏱ Minuta", 0, 90, 45)
with r1_c2:
    s_h = st.number_input(f"Góly {h_team}", 0, 10, 0)
with r1_c3:
    s_a = st.number_input(f"Góly {a_team}", 0, 10, 0)

r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
with r2_c1: xg_h = st.number_input(f"xG {h_team}", 0.0, 10.0, 0.0, step=0.01)
with r2_c2: sh_h = st.number_input(f"Střely {h_team}", 0, 40, 0)
with r2_c3: xg_a = st.number_input(f"xG {a_team}", 0.0, 10.0, 0.0, step=0.01)
with r2_c4: sh_a = st.number_input(f"Střely {a_team}", 0, 40, 0)

# Vizuální Scoreboard
st.markdown(f"""
<div class="score-board">
    <div class="meta-info">LIVE PREDICTION ENGINE • {minute}' MIN</div>
    <div>
        <span class="team-name">{h_team}</span>
        <span class="score">{s_h} - {s_a}</span>
        <span class="team-name">{a_team}</span>
    </div>
    <div class="meta-info">xG: {xg_h} - {xg_a}</div>
</div>
""", unsafe_allow_html=True)


# --- VÝPOČET A PŘEDPOVĚĎ ---
if st.button("🚀 ANALYZOVAT A PŘEDPOVĚDĚT", type="primary", use_container_width=True):
    
    # 1. Příprava dat
    prof = db_profiles.get(h_norm, {'avg_xg': 1.3, 'ppda': 10, 'deep': 6})
    input_data = {
        'minute': minute, 'time_remaining': 90-minute,
        'score_home': s_h, 'score_away': s_a, 'goal_diff': s_h-s_a,
        'current_total_goals': s_h+s_a, 'is_draw': 1 if s_h==s_a else 0,
        'xg_home': xg_h, 'xg_away': xg_a, 'xg_total': xg_h+xg_a, 'xg_diff': xg_h-xg_a,
        'shots_home': sh_h, 'shots_away': sh_a,
        'efficiency_h': s_h-xg_h, 'efficiency_a': s_a-xg_a,
        'avg_shot_qual_h': (xg_h/sh_h) if sh_h>0 else 0,
        'momentum_xg_h': 0.0, 'momentum_pressure_h': 0.0, # HACK
        'elo_home': h_e, 'elo_diff': h_e-a_e,
        'fifa_att_diff': int(f_h['attack'])-int(f_a['attack']),
        'fifa_def_diff': int(f_h['defence'])-int(f_a['defence']),
        'squad_qual_diff': int(f_h['overall'])-int(f_a['overall']),
        'profile_avg_xg_h': prof['avg_xg'], 'profile_ppda_h': prof['ppda'], 'profile_deep_h': prof['deep']
    }
    
    df_in = pd.DataFrame([input_data])
    try:
        pred_total = model.predict(df_in[feat_names])[0]
    except:
        st.error("Chyba modelu.")
        st.stop()

    # 2. Probability Engine (Poisson)
    current_goals = s_h + s_a
    over_probs, expected_more = calculate_probs(pred_total, current_goals)

    # --- DASHBOARD VÝSLEDKŮ ---
    
    # A. HLAVNÍ KARTY
    c_res1, c_res2, c_res3 = st.columns([1.2, 1, 1])
    
    with c_res1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("🏁 PŘEDPOKLÁDANÝ TOTAL")
        st.markdown(f"<h1 style='color: #2196F3; margin:0;'>{pred_total:.2f}</h1>", unsafe_allow_html=True)
        st.write(f"Model očekává ještě **{expected_more:.2f}** gólů.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c_res2:
        # SÁZKOVÉ ŠANCE
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.caption("🎲 SÁZKOVÉ ŠANCE")
        
        # Klíčová sázka (Nejbližší Over)
        next_line = int(current_goals)
        key_prob = over_probs.get(f"Over {next_line + 0.5}", 0) * 100
        
        st.metric(f"OVER {next_line}.5", f"{key_prob:.1f} %")
        st.progress(int(key_prob))
        
        # Vyšší line
        higher_line_prob = over_probs.get(f"Over {next_line + 1.5}", 0) * 100
        if higher_line_prob > 30:
            st.caption(f"Šance na Over {next_line + 1}.5: **{higher_line_prob:.1f} %**")
            
        st.markdown('</div>', unsafe_allow_html=True)

    with c_res3:
         # ATTACK PERFORMANCE
        eff_h = s_h - xg_h
        eff_a = s_a - xg_a
        
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[eff_h, eff_a], y=[h_team, a_team], orientation='h',
            marker=dict(color=['#66BB6A' if eff_h>=0 else '#EF5350', '#66BB6A' if eff_a>=0 else '#EF5350'])
        ))
        fig_bar.update_layout(title="Efektivita (Goals vs xG)", height=150, margin=dict(l=10, r=10, t=30, b=20), xaxis=dict(range=[-2, 2]))
        st.plotly_chart(fig_bar, use_container_width=True)

    # B. VISUAL GOAL PROGRESS
    st.write("---")
    st.subheader("📊 Goal Timeline Prediction")
    
    fig = go.Figure()
    # Šedá zóna (Co už padlo)
    fig.add_trace(go.Bar(
        y=['Zápas'], x=[current_goals], name='Aktuální stav', orientation='h',
        marker=dict(color='#CFD8DC', line=dict(width=0))
    ))
    # Barevná zóna (Co se čeká)
    color_pred = '#4CAF50' if expected_more > 0.5 else '#FF9800'
    fig.add_trace(go.Bar(
        y=['Zápas'], x=[expected_more], name='Očekávaný přídavek', orientation='h',
        marker=dict(color=color_pred, line=dict(width=0)),
        text=[f"+{expected_more:.2f}"], textposition='auto'
    ))
    
    fig.update_layout(
        barmode='stack', height=100, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[0, max(4, int(pred_total)+1)], showgrid=True),
        yaxis=dict(showticklabels=False), showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
