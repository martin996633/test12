import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import zipfile  # <--- TOTO JSME PŘIDALI PRO ROZBALENÍ

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(
    page_title="⚽ AI Betting Advisor (Full Power)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLOVÁNÍ ---
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .metric-box {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
    }
    .recommendation-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .rec-buy { background-color: #1b5e20; color: white; border: 2px solid #4caf50; }
    .rec-wait { background-color: #e65100; color: white; border: 2px solid #ff9800; }
    .rec-skip { background-color: #b71c1c; color: white; border: 2px solid #ef5350; }
    </style>
""", unsafe_allow_html=True)

# --- MAPOVÁNÍ LIG ---
LEAGUE_MAPPING = {
    '🇬🇧 Premier League': 1,
    '🇪🇸 La Liga': 2,
    '🇩🇪 Bundesliga': 3,
    '🇮🇹 Serie A': 4,
    '🇫🇷 Ligue 1': 5,
    '🌍 Jiná': 0
}

# --- NAČTENÍ MODELU (S AUTO-UNZIP) ---
@st.cache_resource
def load_model():
    model_file = "blind_league_model.ubj"
    zip_file = "blind_league_model.zip"
    features_file = "blind_league_features.pkl"

    try:
        # 1. Pokud model není rozbalený, rozbalíme ho ze ZIPu
        if not os.path.exists(model_file):
            if os.path.exists(zip_file):
                with st.spinner("⏳ Rozbaluji velký model... chvilku strpení..."):
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(".")
                st.success("✅ Model úspěšně rozbalen!")
            else:
                st.error(f"❌ Nenalezen ani model ({model_file}), ani ZIP ({zip_file})!")
                return None, None

        # 2. Načtení modelu
        model = xgb.XGBRegressor()
        model.load_model(model_file)
        
        # 3. Načtení features
        features = joblib.load(features_file)
        
        return model, features

    except Exception as e:
        st.error(f"❌ Kritická chyba při načítání: {e}")
        return None, None

model, feature_names = load_model()

# --- SIDEBAR: VSTUPNÍ DATA ---
st.sidebar.header("📝 Zápasové Statistiky")

# 1. Základní info
selected_league_name = st.sidebar.selectbox("Liga", list(LEAGUE_MAPPING.keys()))
league_code = LEAGUE_MAPPING[selected_league_name]

minute = st.sidebar.slider("Minuta zápasu", 0, 90, 35)

col1, col2 = st.sidebar.columns(2)
goals_h = col1.number_input("Góly Domácí", 0, 10, 0)
goals_a = col2.number_input("Góly Hosté", 0, 10, 0)

# 2. Detailní statistiky
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistiky hry")

shots_h = st.sidebar.number_input("Střely Domácí", 0, 50, 3)
shots_a = st.sidebar.number_input("Střely Hosté", 0, 50, 2)

sot_h = st.sidebar.number_input("Střely na bránu (H)", 0, 20, 1)
sot_a = st.sidebar.number_input("Střely na bránu (A)", 0, 20, 0)

xg_h = st.sidebar.number_input("xG Domácí", 0.0, 10.0, 0.45, step=0.01)
xg_a = st.sidebar.number_input("xG Hosté", 0.0, 10.0, 0.25, step=0.01)


# --- HLAVNÍ LOGIKA ---
def calculate_prediction():
    if model is None: return 0, 0

    current_total = goals_h + goals_a
    
    input_data = {
        'minute': minute,
        'time_remaining': 90 - minute,
        'score_home': goals_h, 'score_away': goals_a,
        'goal_diff': goals_h - goals_a,
        'total_goals_current': current_total,
        'is_draw': 1 if goals_h == goals_a else 0,
        'xg_home': xg_h, 'xg_away': xg_a, 'xg_total': xg_h + xg_a, 'xg_diff': xg_h - xg_a,
        'shots_home': shots_h, 'shots_away': shots_a, 'shots_total': shots_h + shots_a,
        'sot_home': sot_h, 'sot_away': sot_a, 'sot_total': sot_h + sot_a, 'sot_diff': sot_h - sot_a,
        'efficiency_h': goals_h - xg_h,
        'efficiency_a': goals_a - xg_a,
        'conversion_rate_h': (goals_h / sot_h) if sot_h > 0 else 0,
        'avg_shot_qual_h': (xg_h / shots_h) if shots_h > 0 else 0,
        'league_code': league_code
    }

    df_input = pd.DataFrame([input_data])
    # Důležité: Oříznout sloupce podle toho, co zná model
    if feature_names:
        df_input = df_input[feature_names]

    pred_remaining = model.predict(df_input)[0]
    pred_remaining = max(0.0, pred_remaining)
    pred_total = current_total + pred_remaining
    
    return pred_remaining, pred_total

# --- UI: DASHBOARD ---
st.title("⚽ AI Live Betting Advisor")
st.markdown(f"**Liga:** {selected_league_name} | **Čas:** {minute}' | **Stav:** {goals_h}:{goals_a}")

if st.sidebar.button("🔮 VYPOČÍTAT PREDIKCI", type="primary"):
    
    pred_rem, pred_total = calculate_prediction()
    
    if model is not None:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Predikce zbytku", f"{pred_rem:.2f} gólů")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_b:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Predikce CELKEM", f"{pred_total:.2f} gólů")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_c:
            fair_line = round(pred_total * 2) / 2
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Očekávaná hranice", f"{fair_line}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🎯 Doporučení pro sázky (Under)")
        
        lines = [1.5, 2.5, 3.5, 4.5]
        safety_margin = 0.35
        cols = st.columns(4)
        
        for i, line in enumerate(lines):
            with cols[i]:
                is_possible = pred_total < (line - safety_margin)
                current_total = goals_h + goals_a
                
                if current_total >= line:
                    st.markdown(f'<div class="recommendation-box rec-skip"><h4>UNDER {line}</h4><p>🚫 Padlo</p></div>', unsafe_allow_html=True)
                elif is_possible:
                    confidence = min(100, int((line - pred_total) * 100))
                    st.markdown(f'<div class="recommendation-box rec-buy"><h4>UNDER {line}</h4><p>✅ BET</p><small>Síla: {confidence}%</small></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="recommendation-box rec-wait"><h4>UNDER {line}</h4><p>⚠️ RISK</p></div>', unsafe_allow_html=True)

    # TAHÁK Z BACKTESTU
    with st.expander("📚 Zobrazit TOP strategie (Rok 2026 data)"):
        st.markdown("""
        **Nejlepší strategie podle backtestu:**
        * **🇮🇹 Serie A (60. min, 0:0):** Under 2.5 (87% úspěšnost)
        * **🇬🇧 Premier League (45. min, 0:0):** Under 3.5 (94% úspěšnost)
        * **🇪🇸 La Liga (45. min, 0:0):** Under 3.5 (95% úspěšnost)
        * **🇫🇷 Ligue 1 (30. min, 0:0):** Under 3.5 (81% úspěšnost)
        """)

else:
    st.info("👈 Zadej data vlevo.")
