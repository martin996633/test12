import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go

# --- KONFIGURACE SOUBORŮ ---
# Tyto soubory musí ležet ve stejné složce
MODEL_FILE = "ultimate_goals_model.json"
FEATURES_FILE = "model_features.pkl"
DATA_STATS_CSV = "data_stats.csv"
DATA_ELO_CSV = "data_elo.csv"
DATA_FIFA_CSV = "data_fifa.csv"

# Nastavení stránky
st.set_page_config(
    page_title="⚽ Live Football Predictor",
    page_icon="🔮",
    layout="wide"
)

# --- MAPOVACÍ FUNKCE (MUSÍ BÝT SHODNÁ S MODEL.PY) ---
def normalize_name(name):
    """ULTIMÁTNÍ MAPPING: Sjednocuje názvy mezi Understat, ClubElo a FIFA."""
    if name is None: return ""
    name = str(name).lower().strip()
    
    mapping = {
        # --- ANGLIE ---
        "man city": "man city", "manchester city": "man city",
        "man utd": "man united", "manchester united": "man united", 
        "nott'm forest": "forest", "nottingham forest": "forest",
        "spurs": "tottenham", "tottenham hotspur": "tottenham",
        "wolves": "wolverhampton wanderers",
        "newcastle": "newcastle united", "brighton": "brighton & hove albion",
        "west ham": "west ham united", "sheffield utd": "sheffield united",
        "leicester": "leicester city", "leeds": "leeds united", "luton": "luton town",
        
        # --- NĚMECKO ---
        "bayern munich": "bayern", "fc bayern münchen": "bayern",
        "bayer leverkusen": "leverkusen", "bayer 04 leverkusen": "leverkusen",
        "borussia dortmund": "dortmund", 
        "borussia m.gladbach": "gladbach", "borussia mönchengladbach": "gladbach",
        "eintracht frankfurt": "frankfurt", 
        "rasenballsport leipzig": "rb leipzig", "rb leipzig": "rb leipzig",
        "fc cologne": "koeln", "1. fc köln": "koeln",
        "mainz 05": "mainz", "1. fsv mainz 05": "mainz",
        "st. pauli": "st pauli", "fc st. pauli": "st pauli",
        "vfb stuttgart": "stuttgart", 
        "werder bremen": "werder", "sv werder bremen": "werder",
        "wolfsburg": "wolfsburg", "vfl wolfsburg": "wolfsburg",
        "augsburg": "augsburg", "fc augsburg": "augsburg",
        "hoffenheim": "hoffenheim", "tsg 1899 hoffenheim": "hoffenheim",
        "union berlin": "union berlin", "1. fc union berlin": "union berlin",
        "bochum": "bochum", "vfl bochum 1848": "bochum",
        "fc heidenheim": "heidenheim", "1. fc heidenheim 1846": "heidenheim",
        "freiburg": "freiburg", "sc freiburg": "freiburg",
        "hamburger sv": "hamburg",

        # --- ŠPANĚLSKO ---
        "atletico madrid": "atletico", "atlético madrid": "atletico",
        "athletic club": "athletic", "athletic club de bilbao": "athletic",
        "barcelona": "barcelona", "fc barcelona": "barcelona",
        "real betis": "betis", "real betis balompié": "betis",
        "celta vigo": "celta", "rc celta": "celta",
        "real sociedad": "sociedad", "real oviedo": "oviedo",
        "alaves": "alaves", "deportivo alavés": "alaves",
        "girona": "girona", "girona fc": "girona",
        "mallorca": "mallorca", "rcd mallorca": "mallorca",
        "osasuna": "osasuna", "ca osasuna": "osasuna",
        "sevilla": "sevilla", "sevilla fc": "sevilla",
        "valencia": "valencia", "valencia cf": "valencia",
        "villarreal": "villarreal", "villarreal cf": "villarreal",
        "getafe": "getafe", "getafe cf": "getafe",
        "espanyol": "espanyol", "rcd espanyol": "espanyol",
        "cadiz": "cadiz", "cádiz cf": "cadiz",
        "almeria": "almeria", "ud almería": "almeria",
        "elche": "elche", "elche cf": "elche",
        "valladolid": "valladolid", "real valladolid cf": "valladolid",

        # --- FRANCIE ---
        "paris saint germain": "paris sg", "paris saint-germain": "paris sg",
        "marseille": "marseille", "olympique de marseille": "marseille",
        "lyon": "lyon", "olympique lyonnais": "lyon",
        "lille": "lille", "lille osc": "lille",
        "monaco": "monaco", "as monaco": "monaco",
        "nice": "nice", "ogc nice": "nice",
        "rennes": "rennes", "stade rennais fc": "rennes",
        "lens": "lens", "rc lens": "lens",
        "strasbourg": "strasbourg", "rc strasbourg alsace": "strasbourg",
        "toulouse": "toulouse", "toulouse fc": "toulouse",
        "nantes": "nantes", "fc nantes": "nantes",
        "reims": "reims", "stade de reims": "reims",
        "montpellier": "montpellier", "montpellier hsc": "montpellier",
        "lorient": "lorient", "fc lorient": "lorient",
        "metz": "metz", "fc metz": "metz",
        "brest": "brest", "stade brestois 29": "brest",
        "le havre": "le havre", "le havre ac": "le havre",
        "auxerre": "auxerre", "aj auxerre": "auxerre",
        "angers": "angers", "angers sco": "angers",

        # --- ITÁLIE ---
        "ac milan": "milan", 
        "inter": "inter", "juventus": "juventus", "roma": "roma",
        "lazio": "lazio", "napoli": "napoli", "atalanta": "atalanta",
        "fiorentina": "fiorentina", "bologna": "bologna", "torino": "torino",
        "udinese": "udinese", "empoli": "empoli", 
        "verona": "verona", "hellas verona": "verona", "hellas verona fc": "verona",
        "lecce": "lecce", "salernitana": "salernitana", "monza": "monza",
        "sassuolo": "sassuolo", "frosinone": "frosinone", "genoa": "genoa",
        "cagliari": "cagliari", "parma calcio 1913": "parma", "parma": "parma",
        "como": "como", "venezia": "venezia"
    }
    
    if name in mapping:
        return mapping[name]
    
    # Fuzzy match fallback
    for key, value in mapping.items():
        if key in name: 
            return value
            
    return name

# --- NAČÍTÁNÍ DAT ---

@st.cache_resource
def load_model_and_features():
    """Načte model a features."""
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
        return model, features
    except Exception as e:
        st.error(f"Chyba při načítání modelu. Zkontroluj, zda existuje soubor {MODEL_FILE}. Chyba: {e}")
        st.stop()

@st.cache_data
def load_csv_data():
    """Načte a připraví data z CSV souborů."""
    try:
        df_stats = pd.read_csv(DATA_STATS_CSV)
        df_elo = pd.read_csv(DATA_ELO_CSV)
        df_fifa = pd.read_csv(DATA_FIFA_CSV)
    except FileNotFoundError as e:
        st.error(f"Chybí CSV soubory! Spustil jsi export? Chyba: {e}")
        st.stop()
    
    # 1. Normalizace jmen v datech (aby seděla s výběrem v aplikaci)
    df_stats['norm_home'] = df_stats['home_team'].apply(normalize_name)
    df_stats['norm_away'] = df_stats['away_team'].apply(normalize_name)
    df_elo['norm_team'] = df_elo['team'].apply(normalize_name)
    df_fifa['norm_team'] = df_fifa['team'].apply(normalize_name)

    # 2. Výpočet profilů týmů
    profiles = {}
    all_teams = set(df_stats['norm_home'].unique()) | set(df_stats['norm_away'].unique())
    
    for team in all_teams:
        h_games = df_stats[df_stats['norm_home'] == team]
        a_games = df_stats[df_stats['norm_away'] == team]
        total = len(h_games) + len(a_games)
        if total < 3: continue
        
        ppda = 10.0
        if 'home_ppda' in df_stats.columns:
            ppda = (h_games['home_ppda'].mean() + a_games['away_ppda'].mean()) / 2
            
        deep = 5.0
        if 'home_deep_completions' in df_stats.columns:
            deep = (h_games['home_deep_completions'].mean() + a_games['away_deep_completions'].mean()) / 2
            
        xg_avg = (h_games['home_xg'].sum() + a_games['away_xg'].sum()) / total
        
        profiles[team] = {
            'avg_xg': xg_avg,
            'ppda': ppda if not pd.isna(ppda) else 10.0,
            'deep': deep if not pd.isna(deep) else 5.0
        }

    # 3. Lookup tabulky
    if 'valid_from' in df_elo.columns:
        df_elo = df_elo.sort_values('valid_from')
    latest_elo = df_elo.groupby('norm_team').tail(1).set_index('norm_team')['elo'].to_dict()
    fifa_map = df_fifa.set_index('norm_team')[['attack', 'defence', 'overall']].to_dict('index')

    return sorted(list(all_teams)), latest_elo, fifa_map, profiles

# --- HLAVNÍ APLIKACE ---

model, feature_names = load_model_and_features()
team_list, db_elo, db_fifa, db_profiles = load_csv_data()

st.title("⚽ AI Football Live Predictor")
st.markdown("XGBoost model trénovaný na datech Understat, ClubElo a FIFA 25.")

# 1. VÝBĚR TÝMŮ
col1, col2 = st.columns(2)
with col1:
    home_team_sel = st.selectbox("Domácí tým", team_list, index=0)
with col2:
    away_team_sel = st.selectbox("Hostující tým", team_list, index=1)

home_norm = normalize_name(home_team_sel)
away_norm = normalize_name(away_team_sel)

# Získání dat
h_elo = db_elo.get(home_norm, 1500)
a_elo = db_elo.get(away_norm, 1500)
h_fifa = db_fifa.get(home_norm, {'attack': 75, 'defence': 75, 'overall': 75})
a_fifa = db_fifa.get(away_norm, {'attack': 75, 'defence': 75, 'overall': 75})

# Debug Info - Pro klid duše, že mapping funguje
with st.expander("🔍 Kontrola Dat (Mapping)", expanded=True):
    c1, c2 = st.columns(2)
    c1.write(f"**{home_team_sel}** -> `{home_norm}`")
    c1.write(f"Elo: **{int(h_elo)}** | FIFA Att: **{h_fifa['attack']}**")
    
    c2.write(f"**{away_team_sel}** -> `{away_norm}`")
    c2.write(f"Elo: **{int(a_elo)}** | FIFA Att: **{a_fifa['attack']}**")

    if h_elo == 1500 and home_norm != "1500":
        st.warning(f"⚠️ Pozor: {home_team_sel} má defaultní Elo 1500. Mapping možná nesedí.")

st.divider()

# 2. LIVE VSTUPY
st.subheader("🔴 Aktuální stav zápasu")

col_time, col_score = st.columns([1, 1])
with col_time:
    minute = st.slider("Minuta", 0, 90, 0)
with col_score:
    sc1, sc2 = st.columns(2)
    score_h = sc1.number_input(f"Góly {home_team_sel}", 0, 15, 0)
    score_a = sc2.number_input(f"Góly {away_team_sel}", 0, 15, 0)

st.write("**Statistiky (Understat / Livesport)**")
col_s1, col_s2 = st.columns(2)
with col_s1:
    xg_h = st.number_input(f"xG {home_team_sel}", 0.0, 10.0, 0.0, step=0.01)
    shots_h = st.number_input(f"Střely {home_team_sel}", 0, 50, 0)
with col_s2:
    xg_a = st.number_input(f"xG {away_team_sel}", 0.0, 10.0, 0.0, step=0.01)
    shots_a = st.number_input(f"Střely {away_team_sel}", 0, 50, 0)

with st.expander("Momentum (Volitelné)"):
    mc1, mc2 = st.columns(2)
    mom_xg_h = mc1.number_input("xG Domácí (posl. 10 min)", 0.0, 5.0, 0.0)
    mom_shots_h = mc2.number_input("Střely Domácí (posl. 10 min)", 0, 15, 0)

# 3. PREDIKCE
st.divider()
if st.button("🔮 PŘEDPOVĚDĚT VÝSLEDEK", type="primary", use_container_width=True):
    
    # Načtení profilu
    prof_h = db_profiles.get(home_norm, {'avg_xg': 1.3, 'ppda': 10, 'deep': 6})
    
    # Sestavení vstupu (Musí sedět s modelem!)
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
        
        'momentum_xg_h': mom_xg_h,
        'momentum_pressure_h': mom_shots_h,
        
        'elo_home': h_elo,
        'elo_diff': h_elo - a_elo,
        'fifa_att_diff': int(h_fifa['attack']) - int(a_fifa['attack']),
        'fifa_def_diff': int(h_fifa['defence']) - int(a_fifa['defence']),
        'squad_qual_diff': int(h_fifa['overall']) - int(a_fifa['overall']),
        
        'profile_avg_xg_h': prof_h['avg_xg'],
        'profile_ppda_h': prof_h['ppda'],
        'profile_deep_h': prof_h['deep'],
    }
    
    # Predikce
    df_in = pd.DataFrame([input_data])
    try:
        df_in = df_in[feature_names] # Seřazení sloupců
    except KeyError as e:
        st.error(f"Chyba struktury: {e}")
        st.stop()
        
    pred_total = model.predict(df_in)[0]
    
    # Zobrazení
    already_scored = score_h + score_a
    to_go = max(0, pred_total - already_scored)
    
    c_res1, c_res2 = st.columns([1, 2])
    with c_res1:
        st.metric("Očekávaný Total", f"{pred_total:.2f}")
        st.metric("Zbývá gólů", f"{to_go:.2f}")
        
    with c_res2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_total,
            title = {'text': "Total Goals"},
            gauge = {
                'axis': {'range': [0, max(5, int(pred_total)+2)]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, already_scored], 'color': "#bdc3c7"},
                    {'range': [already_scored, pred_total], 'color': "#d5f5e3"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': pred_total}
            }
        ))
        fig.update_layout(height=250, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
