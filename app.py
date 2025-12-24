import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
import torch
import torch.nn.functional as F

# --- FIX IMPORT TENSORFLOW BARU ---
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(layout="wide", page_title="Dashboard UAP Analisis Emosi")

# ==========================================
# 2. KONFIGURASI PATH
# ==========================================
BASE_PATH = 'Models' 
EMOTION_LABELS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# ==========================================
# 3. FUNGSI LOAD MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_lstm_resources():
    try:
        lstm_folder = os.path.join(BASE_PATH, 'LSTM_Emotion')
        tokenizer_path = os.path.join(lstm_folder, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        model_path = os.path.join(lstm_folder, 'model_lstm.h5')
        model = load_model(model_path, compile=False) 
        return tokenizer, model
    except Exception as e:
        return None, None

@st.cache_resource
def load_transformer_resources(model_folder_name):
    try:
        path = os.path.join(BASE_PATH, model_folder_name)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        return tokenizer, model
    except Exception as e:
        return None, None

# Load Resources di Awal (Lazy Loading)
# Kita load semua di awal agar switching menu cepat
with st.spinner('Memuat semua model ke memori...'):
    lstm_tokenizer, lstm_model = load_lstm_resources()
    bert_tokenizer, bert_model = load_transformer_resources('BERT_Emotion')
    rob_tokenizer, rob_model = load_transformer_resources('RoBERTa_Emotion')

# ==========================================
# 4. FUNGSI PREDIKSI
# ==========================================
def predict_lstm(text):
    if not lstm_model: return "Error", np.zeros(6)
    seq = lstm_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=150, padding='post', truncating='post')
    prob = lstm_model.predict(padded, verbose=0)[0]
    return EMOTION_LABELS[np.argmax(prob)], prob

def predict_transformer(text, tokenizer, model):
    if not model: return "Error", np.zeros(6)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1).detach().numpy()[0]
    return EMOTION_LABELS[np.argmax(probs)], probs

# ==========================================
# 5. UI DASHBOARD (SIDEBAR & MAIN)
# ==========================================

# --- SIDEBAR MENU ---
with st.sidebar:
    st.header("⚙️ Panel Kontrol")
    st.write("Pilih model yang ingin diuji:")
    
    # Widget Pilihan Model
    selected_mode = st.radio(
        "Mode Pengujian:",
        ("Semua Model (Komparasi)", "Hanya LSTM", "Hanya BERT", "Hanya RoBERTa")
    )
    
    st.info("💡 **Tips:** Gunakan mode 'Semua Model' untuk membandingkan performa, atau pilih satu model untuk analisis mendalam.")

# --- MAIN CONTENT ---
st.title("🧠 Dashboard Analisis Emosi")
st.markdown(f"### Mode Aktif: **{selected_mode}**")
st.divider()

col_input, col_result = st.columns([1, 1.5])

with col_input:
    st.subheader("📝 Input Teks")
    user_input = st.text_area("Masukkan teks (Bahasa Inggris):", height=150, placeholder="I am so surprised to see this result!")
    btn_predict = st.button("🔍 Analisis Emosi", type="primary", use_container_width=True)

if btn_predict and user_input:
    with col_result:
        st.subheader("📊 Hasil Prediksi")

        # --- LOGIKA TAMPILAN BERDASARKAN PILIHAN ---
        
        # 1. MODE KOMPARASI (SEMUA MODEL)
        if selected_mode == "Semua Model (Komparasi)":
            # Run All
            l_lbl, l_prob = predict_lstm(user_input)
            b_lbl, b_prob = predict_transformer(user_input, bert_tokenizer, bert_model)
            r_lbl, r_prob = predict_transformer(user_input, rob_tokenizer, rob_model)
            
            # Kartu Score
            c1, c2, c3 = st.columns(3)
            c1.metric("LSTM", l_lbl.upper(), f"{max(l_prob):.1%}")
            c2.metric("BERT", b_lbl.upper(), f"{max(b_prob):.1%}")
            c3.metric("RoBERTa", r_lbl.upper(), f"{max(r_prob):.1%}")
            
            # Grafik Perbandingan
            df_viz = pd.DataFrame({
                'Emosi': EMOTION_LABELS * 3,
                'Probability': np.concatenate([l_prob, b_prob, r_prob]),
                'Model': ['LSTM']*6 + ['BERT']*6 + ['RoBERTa']*6
            })
            fig = px.bar(df_viz, x="Emosi", y="Probability", color="Model", barmode="group",
                         color_discrete_map={'LSTM': '#FFA07A', 'BERT': '#87CEFA', 'RoBERTa': '#90EE90'})
            st.plotly_chart(fig, use_container_width=True)

        # 2. MODE SINGLE MODEL (SALAH SATU)
        else:
            # Tentukan model mana yang jalan
            if selected_mode == "Hanya LSTM":
                lbl, prob = predict_lstm(user_input)
                color_theme = '#FFA07A' # Orange
            elif selected_mode == "Hanya BERT":
                lbl, prob = predict_transformer(user_input, bert_tokenizer, bert_model)
                color_theme = '#87CEFA' # Blue
            else: # RoBERTa
                lbl, prob = predict_transformer(user_input, rob_tokenizer, rob_model)
                color_theme = '#90EE90' # Green

            # Tampilan Single
            st.metric(f"Prediksi {selected_mode}", lbl.upper(), f"{max(prob):.1%}")
            
            st.markdown("#### Detail Probabilitas Emosi:")
            # Grafik Detail untuk 1 Model
            df_single = pd.DataFrame({
                'Emosi': EMOTION_LABELS,
                'Probability': prob
            })
            
            # Urutkan dari probabilitas tertinggi
            df_single = df_single.sort_values(by='Probability', ascending=True)
            
            fig = px.bar(df_single, x="Probability", y="Emosi", orientation='h', text_auto='.1%',
                         title=f"Distribusi Keyakinan Model ({selected_mode})",
                         color_discrete_sequence=[color_theme])
            
            st.plotly_chart(fig, use_container_width=True)