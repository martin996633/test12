import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(
    page_title="⚽ AI Betting Advisor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLOVÁNÍ (Pro moderní vzhled) ---
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

# --- NAČTENÍ MODELU ---
@st.cache_resource
def load_model():
    try:
        model = xgb.XGBRegressor()
        model.load_model("blind_league_model.ubj")
        features = joblib.load("blind_league_features.pkl")
        return model, features
    except Exception as e:
        st.error(f"❌ Chyba při načítání modelu: {e}")
        st.warning("Ujisti se, že soubory .ubj a .pkl jsou ve stejné složce jako app.py")
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

# 2. Detailní statistiky (Sázkař zadává z Flashscore)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistiky hry")

shots_h = st.sidebar.number_input("Střely Domácí", 0, 50, 3)
shots_a = st.sidebar.number_input("Střely Hosté", 0, 50, 2)

sot_h = st.sidebar.number_input("Střely na bránu (H)", 0, 20, 1)
sot_a = st.sidebar.number_input("Střely na bránu (A)", 0, 20, 0)

# xG (Pokud není k dispozici, odhadneme nahrubo)
xg_h = st.sidebar.number_input("xG Domácí", 0.0, 10.0, 0.45, step=0.01)
xg_a = st.sidebar.number_input("xG Hosté", 0.0, 10.0, 0.25, step=0.01)


# --- HLAVNÍ LOGIKA ---
def calculate_prediction():
    if model is None: return

    # 1. Feature Engineering (Přepočet vstupů na formát modelu)
    current_total = goals_h + goals_a
    
    # Odvozené statistiky
    input_data = {
        'minute': minute,
        'time_remaining': 90 - minute,
        'score_home': goals_h, 'score_away': goals_a,
        'goal_diff': goals_h - goals_a,
        'total_goals_current': current_total,
        'is_draw': 1 if goals_h == goals_a else 0,
        
        'xg_home': xg_h, 'xg_away': xg_a,
        'xg_total': xg_h + xg_a,
        'xg_diff': xg_h - xg_a,
        
        'shots_home': shots_h, 'shots_away': shots_a,
        'shots_total': shots_h + shots_a,
        
        'sot_home': sot_h, 'sot_away': sot_a,
        'sot_total': sot_h + sot_a,
        'sot_diff': sot_h - sot_a,
        
        'efficiency_h': goals_h - xg_h,
        'efficiency_a': goals_a - xg_a,
        'conversion_rate_h': (goals_h / sot_h) if sot_h > 0 else 0,
        'avg_shot_qual_h': (xg_h / shots_h) if shots_h > 0 else 0,
        
        'league_code': league_code
    }

    # Vytvoření DataFrame se správným pořadím sloupců
    df_input = pd.DataFrame([input_data])
    # Zajištění, že máme jen features, které model zná, a ve správném pořadí
    df_input = df_input[feature_names]

    # 2. Predikce
    pred_remaining = model.predict(df_input)[0]
    # Ošetření záporných predikcí (model může matematicky ulítnout)
    pred_remaining = max(0.0, pred_remaining)
    
    pred_total = current_total + pred_remaining
    
    return pred_remaining, pred_total

# --- UI: HLAVNÍ DASHBOARD ---
st.title("⚽ AI Live Betting Advisor")
st.markdown(f"**Liga:** {selected_league_name} | **Čas:** {minute}' | **Stav:** {goals_h}:{goals_a}")

if st.sidebar.button("🔮 VYPOČÍTAT PREDIKCI", type="primary"):
    
    pred_rem, pred_total = calculate_prediction()
    
    # 1. Sekce: Výsledky modelu
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Predikce zbytku zápasu", f"{pred_rem:.2f} gólů")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Predikce CELKEM", f"{pred_total:.2f} gólů")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        # Zobrazíme "Fair Line" (Kde je střední hodnota)
        fair_line = round(pred_total * 2) / 2  # Zaokrouhlení na nejbližší 0.5
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Očekávaná hranice", f"{fair_line}")
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Sekce: Strategické karty (UNDER)
    st.markdown("### 🎯 Doporučení pro sázky (Under)")
    
    lines = [1.5, 2.5, 3.5, 4.5]
    safety_margin = 0.35  # Jak moc si musí být model jistý (z backtestu)
    
    cols = st.columns(4)
    
    for i, line in enumerate(lines):
        with cols[i]:
            # Logika doporučení
            is_possible = pred_total < (line - safety_margin)
            
            # Jen pokud už není hranice překonaná
            current_total = goals_h + goals_a
            if current_total >= line:
                st.markdown(f"""
                <div class="recommendation-box rec-skip">
                    <h4>UNDER {line}</h4>
                    <p>🚫 Hranice překročena</p>
                </div>
                """, unsafe_allow_html=True)
            elif is_possible:
                # Výpočet "síly" signálu
                diff = line - pred_total
                confidence = min(100, int(diff * 100))
                
                st.markdown(f"""
                <div class="recommendation-box rec-buy">
                    <h4>UNDER {line}</h4>
                    <p style="font-size: 20px">✅ BET</p>
                    <small>Síla signálu: {confidence}%</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recommendation-box rec-wait">
                    <h4>UNDER {line}</h4>
                    <p>⚠️ RISK</p>
                    <small>Model čeká: {pred_total:.2f}</small>
                </div>
                """, unsafe_allow_html=True)

    # 3. Sekce: Tahák z Backtestu (Statické info)
    with st.expander("📚 Zobrazit TOP strategie z historie (Tahák)"):
        st.markdown("""
        **Založeno na datech z roku 2026:**
        * **🇮🇹 Serie A (60. min, 0:0):** Under 2.5 (87% úspěšnost)
        * **🇬🇧 Premier League (45. min, 0:0):** Under 3.5 (94% úspěšnost)
        * **🇪🇸 La Liga (45. min, 0:0):** Under 3.5 (95% úspěšnost)
        * **🇫🇷 Ligue 1 (30. min, 0:0):** Under 3.5 (81% úspěšnost)
        * ⚠️ **Bundesliga:** Vyhýbat se Under sázkám, vysoká volatilita.
        """)

else:
    st.info("👈 Zadej aktuální statistiky v postranním panelu a klikni na 'Vypočítat'.")

# Footer
st.markdown("---")
st.caption("AI Model v1.0 | Data: Understat | Engine: XGBoost | © Martin")
