import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go

# --- KONFIGURACE ---
MODEL_FILE = "ultimate_goals_model.json"
FEATURES_FILE = "model_features.pkl"
DATA_STATS_CSV = "data_stats.csv"
DATA_ELO_CSV = "data_elo.csv"
DATA_FIFA_CSV = "data_fifa.csv"

st.set_page_config(page_title="⚽ AI Live Predictor", page_icon="⚡", layout="centered")

# --- MAPPING (Musí zůstat, aby fungovaly týmy) ---
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

# --- UI START ---
model, feat_names = load_stuff()
teams, db_elo, db_fifa, db_profiles = load_csv()

st.title("⚡ AI Live Predictor")

# VÝBĚR TÝMŮ
c1, c2 = st.columns(2)
h_team = c1.selectbox("🏠 Domácí", teams, index=0)
a_team = c2.selectbox("✈️ Hosté", teams, index=1)
h_norm, a_norm = normalize_name(h_team), normalize_name(a_team)

# KONTEXT (Elo)
h_e, a_e = db_elo.get(h_norm, 1500), db_elo.get(a_norm, 1500)
st.caption(f"Síla týmů (Elo): {h_team} ({int(h_e)}) vs {a_team} ({int(a_e)})")
st.divider()

# VSTUPY (Momentum odstraněno z očí uživatele)
col_min, col_score, col_stats = st.columns([1, 1.5, 2.5])
with col_min:
    minute = st.number_input("Minuta", 0, 90, 0)
with col_score:
    s_h = st.number_input(f"Góly {h_team}", 0, 10, 0)
    s_a = st.number_input(f"Góly {a_team}", 0, 10, 0)
with col_stats:
    c_x1, c_x2 = st.columns(2)
    xg_h = c_x1.number_input(f"xG {h_team}", 0.0, 10.0, 0.0, step=0.01)
    xg_a = c_x2.number_input(f"xG {a_team}", 0.0, 10.0, 0.0, step=0.01)
    
    c_sh1, c_sh2 = st.columns(2)
    sh_h = c_sh1.number_input(f"Střely {h_team}", 0, 40, 0)
    sh_a = c_sh2.number_input(f"Střely {a_team}", 0, 40, 0)

# VÝPOČET
if st.button("🔮 ANALYZOVAT ZÁPAS", type="primary", use_container_width=True):
    prof = db_profiles.get(h_norm, {'avg_xg': 1.3, 'ppda': 10, 'deep': 6})
    f_h = db_fifa.get(h_norm, {'attack': 75, 'defence': 75, 'overall': 75})
    f_a = db_fifa.get(a_norm, {'attack': 75, 'defence': 75, 'overall': 75})
    
    input_data = {
        'minute': minute, 'time_remaining': 90-minute,
        'score_home': s_h, 'score_away': s_a, 'goal_diff': s_h-s_a,
        'current_total_goals': s_h+s_a, 'is_draw': 1 if s_h==s_a else 0,
        'xg_home': xg_h, 'xg_away': xg_a, 'xg_total': xg_h+xg_a, 'xg_diff': xg_h-xg_a,
        'shots_home': sh_h, 'shots_away': sh_a,
        'efficiency_h': s_h-xg_h, 'efficiency_a': s_a-xg_a,
        'avg_shot_qual_h': (xg_h/sh_h) if sh_h>0 else 0,
        
        # --- HACK PRO STARÝ MODEL ---
        # Posíláme nuly, aby model nespadl, i když to uživatel nevidí
        'momentum_xg_h': 0.0,
        'momentum_pressure_h': 0.0,
        # ----------------------------
        
        'elo_home': h_e, 'elo_diff': h_e-a_e,
        'fifa_att_diff': int(f_h['attack'])-int(f_a['attack']),
        'fifa_def_diff': int(f_h['defence'])-int(f_a['defence']),
        'squad_qual_diff': int(f_h['overall'])-int(f_a['overall']),
        'profile_avg_xg_h': prof['avg_xg'], 'profile_ppda_h': prof['ppda'], 'profile_deep_h': prof['deep']
    }
    
    # Seřazení sloupců a predikce
    df_in = pd.DataFrame([input_data])
    try:
        df_in = df_in[feat_names]
        pred_total = model.predict(df_in)[0]
    except KeyError as e:
        st.error(f"Chyba modelu (chybí feature): {e}")
        st.stop()
    
    # VIZUALIZACE VÝSLEDKŮ (Nový Dashboard Design)
    curr_total = s_h + s_a
    expected_more = max(0, pred_total - curr_total)
    
    st.markdown("---")
    
    col_res1, col_res2 = st.columns([1, 1.5])
    
    with col_res1:
        st.subheader("🏁 Finální Total")
        # Velké zelené číslo
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50; font-size: 60px;'>{pred_total:.2f}</h1>", unsafe_allow_html=True)
        
        # Slovní interpretace
        if expected_more > 1.5:
            st.warning(f"🔥 Očekávám ještě cca 2 góly!")
        elif expected_more > 0.6:
            st.info(f"⚡ Ještě jeden gól by měl padnout.")
        else:
            st.success("❄️ Zápas už se spíše dohraje.")
            
    with col_res2:
        st.subheader("📊 Průběh gólů")
        
        # Sloupcový graf (Stacked Bar Chart)
        fig = go.Figure()
        
        # Co už padlo (Šedá)
        fig.add_trace(go.Bar(
            y=['Góly'], x=[curr_total], name='Aktuální stav',
            orientation='h', marker=dict(color='#CFD8DC', line=dict(width=0))
        ))
        
        # Co se čeká (Zelená)
        fig.add_trace(go.Bar(
            y=['Góly'], x=[expected_more], name='Očekávaný přídavek',
            orientation='h', marker=dict(color='#4CAF50', line=dict(width=0)),
            text=[f"+{expected_more:.2f}"], textposition='auto'
        ))
        
        fig.update_layout(
            barmode='stack', 
            height=150, 
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(range=[0, max(5, int(pred_total)+2)], showgrid=False, title="Počet gólů"),
            yaxis=dict(showticklabels=False),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Efektivita (Bonus)
    with st.expander("🔍 Detail: Efektivita a štěstí"):
        ce1, ce2 = st.columns(2)
        eff_h = s_h - xg_h
        eff_a = s_a - xg_a
        ce1.metric(f"{h_team}", f"{eff_h:+.2f}", help="Kladné = Tým dává víc gólů než by měl (Skill/Štěstí). Záporné = Spaluje šance.")
        ce2.metric(f"{a_team}", f"{eff_a:+.2f}")
