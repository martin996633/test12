import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import plotly.graph_objects as go
import math
import json
import os
import zipfile

# --- KONFIGURACE ---
MODEL_ARCHIVE = "blind_remaining_model.zip"
MODEL_FILENAME = "blind_remaining_model.ubj"
FEATURES_FILENAME = "blind_remaining_features.pkl"
METADATA_FILENAME = "blind_remaining_metadata.json"

st.set_page_config(page_title="AI Blind Predictor (Remaining)", page_icon="🔮", layout="wide")

# --- STYLY ---
st.markdown("""
<style>
    .main-header { font-size: 36px; font-weight: 800; color: #D81B60; margin-bottom: 20px; text-align: center;}
    .score-board { background-color: #121212; padding: 25px; border-radius: 15px; text-align: center; color: white; margin-bottom: 25px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# Funkce pro pravděpodobnosti (Upravená pro 'remaining')
def calculate_probs(predicted_remaining, current_goals):
    # Lambda nemůže být záporná, minimum je 0.01
    lamb = max(0.01, predicted_remaining)
    
    def poisson(k, lamb): return (lamb**k * math.exp(-lamb)) / math.factorial(k)
    probs = {i: poisson(i, lamb) for i in range(7)}
    
    # Over/Under počítáme k CELKOVÉMU skóre
    # Příklad: Stav 1:0 (current=1). Over 1.5 znamená, že padne ještě aspoň 1 gól (k > 0)
    
    over = {
        f"Over {current_goals + 0.5}": 1.0 - probs[0],          # Padne > 0 dalších
        f"Over {current_goals + 1.5}": 1.0 - (probs[0]+probs[1]), # Padne > 1 dalších
        f"Over {current_goals + 2.5}": 1.0 - (probs[0]+probs[1]+probs[2])
    }
    
    under = {
        f"Under {current_goals + 0.5}": probs[0],
        f"Under {current_goals + 1.5}": probs[0]+probs[1],
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

# --- UI ---
st.markdown('<div class="main-header">🔮 AI Blind Predictor</div>', unsafe_allow_html=True)
st.caption("Model trénovaný na predikci ZBÝVAJÍCÍCH gólů (Remaining Goals).")

c1, c2 = st.columns(2)
h_name = c1.text_input("Domácí", "Domácí")
a_name = c2.text_input("Hosté", "Hosté")

st.markdown("### 📊 Zadej statistiky")
with st.container():
    col1, col2, col3 = st.columns(3)
    minute = col1.number_input("Minuta", 0, 95, 45)
    g_h = col2.number_input(f"Góly {h_name}", 0, 10, 0)
    g_a = col3.number_input(f"Góly {a_name}", 0, 10, 0)
    
    st.divider()
    c_xg, c_sh, c_sot = st.columns(3)
    
    # DOMÁCÍ
    c_xg.markdown(f"**{h_name}**")
    xg_h = c_xg.number_input(f"xG Home", 0.0, 10.0, 0.0, step=0.01)
    sh_h = c_sh.number_input(f"Střely Home", 0, 50, 0)
    sot_h = c_sot.number_input(f"SoT Home", 0, 50, 0)
    
    # HOSTÉ
    c_xg.markdown(f"**{a_name}**")
    xg_a = c_xg.number_input(f"xG Away", 0.0, 10.0, 0.0, step=0.01)
    sh_a = c_sh.number_input(f"Střely Away", 0, 50, 0)
    sot_a = c_sot.number_input(f"SoT Away", 0, 50, 0)

st.markdown(f"""
<div class="score-board">
    {h_name} <b>{g_h} : {g_a}</b> {a_name}<br>
    <small>Min: {minute}' | SoT: {sot_h} - {sot_a}</small>
</div>
""", unsafe_allow_html=True)

if st.button("🚀 PREDIKOVAT", type="primary", use_container_width=True):
    if model:
        # PŘÍPRAVA DAT
        data = {
            'minute': minute, 'time_remaining': 90-minute,
            'score_home': g_h, 'score_away': g_a, 'goal_diff': g_h-g_a, 
            'total_goals_current': g_h+g_a, # Teď jen jako kontext
            'is_draw': 1 if g_h==g_a else 0,
            
            'xg_home': xg_h, 'xg_away': xg_a, 'xg_total': xg_h+xg_a, 'xg_diff': xg_h-xg_a,
            'shots_home': sh_h, 'shots_away': sh_a, 'shots_total': sh_h+sh_a,
            'sot_home': sot_h, 'sot_away': sot_a, 'sot_total': sot_h+sot_a, 'sot_diff': sot_h-sot_a,
            
            'efficiency_h': g_h-xg_h, 'efficiency_a': g_a-xg_a,
            'conversion_rate_h': (g_h/sot_h) if sot_h>0 else 0,
            'avg_shot_qual_h': (xg_h/sh_h) if sh_h>0 else 0
        }
        
        df = pd.DataFrame([data])[feat_names]
        
        # PŘÍMÝ VÝSTUP MODELU = REMAINING GOALS
        pred_remaining = model.predict(df)[0]
        # Ošetření záporných čísel (nemůže padnout -0.1 gólu)
        pred_remaining = max(0.0, pred_remaining)
        
        # Celkový očekávaný výsledek
        expected_total = (g_h + g_a) + pred_remaining
        
        o, u, lamb = calculate_probs(pred_remaining, g_h+g_a)
        
        # --- VÝSLEDKY ---
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown('<div style="background:#eee;padding:15px;border-radius:10px;">', unsafe_allow_html=True)
            st.caption("AI Očekává ještě:")
            st.title(f"+ {pred_remaining:.2f} gólů")
            st.metric("Očekávaný TOTAL", f"{expected_total:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c_res2:
            st.write("Sázky (O/U):")
            for k,v in o.items(): 
                val = v*100
                color = "green" if val > 50 else "black"
                st.write(f"{k}: :{color}[**{val:.1f}%**]")

    else:
        st.error("Model nenalezen.")
