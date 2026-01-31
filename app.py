import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import math
import json
import os
import zipfile

# --- KONFIGURACE ---
# Hledáme ZIP archiv s modelem
MODEL_ARCHIVE = "blind_remaining_model.zip"
MODEL_FILENAME = "blind_remaining_model.ubj"
FEATURES_FILENAME = "blind_remaining_features.pkl"
METADATA_FILENAME = "blind_remaining_metadata.json"

st.set_page_config(page_title="AI Blind Predictor", page_icon="🔮", layout="wide")

# --- STYLY ---
st.markdown("""
<style>
    .main-header { font-size: 36px; font-weight: 800; color: #D81B60; margin-bottom: 20px; text-align: center;}
    .score-board { background-color: #121212; padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 25px; border: 1px solid #333; }
    .stat-box { background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- VÝPOČTY ---
def calculate_probs(predicted_remaining, current_goals):
    # Lambda (očekávaný počet dalších gólů) musí být kladná
    lamb = max(0.01, predicted_remaining)
    
    def poisson(k, lamb): return (lamb**k * math.exp(-lamb)) / math.factorial(k)
    probs = {i: poisson(i, lamb) for i in range(7)}
    
    # Over/Under se vztahuje k CELKOVÉMU skóre (Current + Remaining)
    # Příklad: Stav 2:0 (Current=2). Over 2.5 znamená, že padne ještě > 0.5 gólu.
    
    over = {
        f"Over {current_goals + 0.5}": 1.0 - probs[0],            # Padne aspoň 1 další
        f"Over {current_goals + 1.5}": 1.0 - (probs[0]+probs[1]), # Padnou aspoň 2 další
        f"Over {current_goals + 2.5}": 1.0 - (probs[0]+probs[1]+probs[2])
    }
    
    under = {
        f"Under {current_goals + 0.5}": probs[0],                # Nepadne už nic
        f"Under {current_goals + 1.5}": probs[0]+probs[1],       # Padne max 1
        f"Under {current_goals + 2.5}": probs[0]+probs[1]+probs[2]
    }
    return over, under, lamb

@st.cache_resource
def load_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_archive = os.path.join(current_dir, MODEL_ARCHIVE)
        path_model = os.path.join(current_dir, MODEL_FILENAME)
        path_features = os.path.join(current_dir, FEATURES_FILENAME)

        # Logika rozbalení ZIPu
        if not os.path.exists(path_model):
            if os.path.exists(path_archive):
                with zipfile.ZipFile(path_archive, 'r') as zip_ref:
                    zip_ref.extractall(current_dir)
            else:
                return None, None

        m = xgb.XGBRegressor()
        m.load_model(path_model)
        f = joblib.load(path_features)
        return m, f
    except: return None, None

model, feat_names = load_model()

# --- UI APLIKACE ---
st.markdown('<div class="main-header">🔮 AI Blind Predictor</div>', unsafe_allow_html=True)
st.caption("Model: Blind Mode (Nezná týmy) | Cíl: Remaining Goals (Zbývající góly)")

# Vstupní data
c1, c2 = st.columns(2)
h_name = c1.text_input("Domácí", "Domácí")
a_name = c2.text_input("Hosté", "Hosté")

st.markdown("### 📊 Statistiky Zápasu")
with st.container():
    # Hlavní řádek
    col1, col2, col3 = st.columns(3)
    minute = col1.number_input("⏱ Minuta", 0, 95, 60)
    g_h = col2.number_input(f"⚽ Góly {h_name}", 0, 10, 0)
    g_a = col3.number_input(f"⚽ Góly {a_name}", 0, 10, 0)
    
    st.divider()
    
    # Detailní statistiky
    c_xg, c_sh, c_sot = st.columns(3)
    
    c_xg.markdown(f"**{h_name}**")
    xg_h = c_xg.number_input(f"xG Home", 0.0, 10.0, 0.0, step=0.01)
    sh_h = c_sh.number_input(f"Střely Home", 0, 50, 0)
    sot_h = c_sot.number_input(f"SoT (Na bránu) Home", 0, 50, 0)
    
    c_xg.markdown(f"**{a_name}**")
    xg_a = c_xg.number_input(f"xG Away", 0.0, 10.0, 0.0, step=0.01)
    sh_a = c_sh.number_input(f"Střely Away", 0, 50, 0)
    sot_a = c_sot.number_input(f"SoT (Na bránu) Away", 0, 50, 0)

# Scoreboard vizualizace
st.markdown(f"""
<div class="score-board">
    <span style="font-size:24px">{h_name}</span> 
    <span style="font-size:40px; font-weight:bold; margin:0 15px;">{g_h} : {g_a}</span> 
    <span style="font-size:24px">{a_name}</span><br>
    <div style="margin-top:10px; color:#aaa;">Min: {minute}' | SoT: {sot_h} - {sot_a}</div>
</div>
""", unsafe_allow_html=True)

# Tlačítko výpočtu
if st.button("🚀 SPOČÍTAT PREDIKCI", type="primary", use_container_width=True):
    if model:
        # Příprava dat pro model
        data = {
            'minute': minute, 'time_remaining': 90-minute,
            'score_home': g_h, 'score_away': g_a, 'goal_diff': g_h-g_a, 
            'total_goals_current': g_h+g_a,
            'is_draw': 1 if g_h==g_a else 0,
            
            'xg_home': xg_h, 'xg_away': xg_a, 'xg_total': xg_h+xg_a, 'xg_diff': xg_h-xg_a,
            'shots_home': sh_h, 'shots_away': sh_a, 'shots_total': sh_h+sh_a,
            'sot_home': sot_h, 'sot_away': sot_a, 'sot_total': sot_h+sot_a, 'sot_diff': sot_h-sot_a,
            
            'efficiency_h': g_h-xg_h, 'efficiency_a': g_a-xg_a,
            'conversion_rate_h': (g_h/sot_h) if sot_h>0 else 0,
            'avg_shot_qual_h': (xg_h/sh_h) if sh_h>0 else 0
        }
        
        # Predikce
        df = pd.DataFrame([data])[feat_names]
        pred_remaining = model.predict(df)[0]
        pred_remaining = max(0.0, pred_remaining) # Ošetření
        
        expected_total = (g_h + g_a) + pred_remaining
        o, u, lamb = calculate_probs(pred_remaining, g_h+g_a)
        
        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        res_c1, res_c2 = st.columns([1, 1.5])
        
        with res_c1:
            st.markdown('<div class="stat-box">', unsafe_allow_html=True)
            st.caption("🤖 AI předpovídá ještě:")
            st.markdown(f"<h1 style='color:#D81B60'>+ {pred_remaining:.2f}</h1>", unsafe_allow_html=True)
            st.metric("Očekávaný TOTAL (FT)", f"{expected_total:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with res_c2:
            st.markdown("#### 🎲 Pravděpodobnosti (Sázky)")
            
            # Tabulka Over / Under
            current_g = g_h + g_a
            lines = [current_g + 0.5, current_g + 1.5, current_g + 2.5]
            
            for line in lines:
                o_key = f"Over {line}"
                u_key = f"Under {line}"
                
                o_val = o.get(o_key, 0)
                u_val = u.get(u_key, 0)
                
                # Barvy: Zelená pro > 50%, Šedá pro zbytek
                c_o = "green" if o_val > 0.5 else "grey"
                c_u = "green" if u_val > 0.5 else "grey"
                
                row = st.columns([1, 1])
                row[0].write(f"⬆️ {o_key}: :{c_o}[**{o_val*100:.1f}%**]")
                row[1].write(f"⬇️ {u_key}: :{c_u}[**{u_val*100:.1f}%**]")
                st.progress(int(o_val*100)) # Progress bar ukazuje sílu Overu

    else:
        st.error("❌ Model nebyl nalezen. Zkontroluj, zda jsi nahrál soubory na GitHub.")

# --- FOOTER ---
with st.expander("ℹ️ Metadata modelu"):
    if os.path.exists(METADATA_FILENAME):
        with open(METADATA_FILENAME, "r") as f: st.json(json.load(f))
    else: st.info("Metadata nedostupná")
