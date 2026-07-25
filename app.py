import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# ==========================================
# KONFIGURASI TAMPILAN & CSS (Profesional Sekolah, Hijau Gelap, Bold)
# ==========================================
st.set_page_config(
    page_title="MLOps Dashboard Olist - FTI",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
    <style>
    /* Memaksa semua teks menjadi tebal (bold) dan berwarna hijau gelap */
    html, body, [class*="css"], [class*="st-"] {
        font-weight: bold !important;
        color: #1B5E20 !important; 
    }
    
    /* Mengubah warna latar belakang aplikasi menjadi hijau abu-abu sangat pucat (elegan, tidak cerah) */
    .stApp {
        background-color: #DDE5B6 !important; 
    }
    
    /* Modifikasi tombol */
    .stButton>button {
        background-color: #2E7D32 !important;
        color: white !important;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1B5E20 !important;
    }
    
    /* Modifikasi kotak unggah file */
    .stFileUploader {
        background-color: #ADC178 !important;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- JUDUL APLIKASI ---
st.title("⚙️ MLOps Dashboard & Data Decision Olist")
st.write("Sistem Portal Prediksi & Segmentasi Terpadu berbasis Cloud-native App.")
st.subheader("Opsi B: Deteksi Dini Pembatalan Transaksi (Supervised - Klasifikasi)")

# --- PEMBAGIAN 7 SIKLUS MENGGUNAKAN TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Ingestion", "2. Preprocessing", "3. Feature Eng.", 
    "4. Data Model", "5. Evaluation", "6. Visualization", "7. Decision"
])

# Variabel global untuk menyimpan state sesi
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_terlatih' not in st.session_state:
    st.session_state.model_terlatih = None
if 'scaler_aktif' not in st.session_state:
    st.session_state.scaler_aktif = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# --- SIKLUS 1: DATA INGESTION ---
with tab1:
    st.header("Siklus 1: Data Ingestion")
    st.write("Unggah data historis transaksi untuk melatih algoritma.")
    uploaded_file = st.file_uploader("Pilih file CSV Dataset", type=['csv'], key="data_latih")
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Berkas berhasil diunggah dan dibaca sistem!")
        st.dataframe(st.session_state.df.head())

# --- SIKLUS 2: DATA PREPROCESSING ---
with tab2:
    st.header("Siklus 2: Data Preprocessing")
    st.write("Automasi pembersihan data awal sebelum diolah.")
    if st.session_state.df is not None:
        if st.button("Jalankan Preprocessing Data"):
            df_bersih = st.session_state.df.dropna()
            st.session_state.df = df_bersih
            st.success("Pemrosesan selesai! Baris data yang tidak lengkap telah dieliminasi.")
            st.write(f"Sisa baris data valid: {len(df_bersih)}")
    else:
        st.warning("Silakan unggah data di tab Ingestion terlebih dahulu.")

# --- SIKLUS 3: FEATURE ENGINEERING ---
with tab3:
    st.header("Siklus 3: Feature Engineering")
    st.write("Standardisasi rentang nilai kolom numerik (Z-score Scaling).")
    
    if st.session_state.df is not None:
        if st.button("Terapkan Feature Engineering"):
            try:
                fitur_prediktor = ['nilai_belanja', 'jumlah_cicilan', 'jumlah_item', 'frekuensi_belanja']
                X_data = st.session_state.df[fitur_prediktor]
                y_target = st.session_state.df['status_batal']
                
                scaler = StandardScaler()
                X_skala = scaler.fit_transform(X_data)
                st.session_state.scaler_aktif = scaler 
                
                X_train, X_test, y_train, y_test = train_test_split(X_skala, y_target, test_size=0.2, random_state=42)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success("Standardisasi matriks selesai diterapkan ke dalam memori.")
            except KeyError:
                st.error("Pastikan dataset yang diunggah memiliki nama kolom yang tepat.")
    else:
        st.warning("Silakan unggah data di tab Ingestion terlebih dahulu.")

# --- SIKLUS 4: DATA MODEL (IMPLEMENTASI KODE DOSEN - FASE 1) ---
with tab4:
    st.header("Siklus 4: Dynamic Training & Export (Model Registry)")
    st.write("Pelatihan algoritma dan penyimpanan sistem secara terstruktur.")
    
    model_name_input = st.text_input("Beri Nama Versi Model Baru Anda:", "model_klasifikasi_v1")
    file_bin_name = f"{model_name_input}.joblib"
    
    if st.session_state.get('X_train') is not None:
        if st.button("Latih & Ekspor Model Ke Server Lokal"):
            
            # Melatih algoritma dengan data historis riil (bukan simulasi)
            model_rf = RandomForestClassifier(random_state=42)
            model_rf.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.model_terlatih = model_rf
            
            # Logika Model Registry dari Dosen (Pengecekan Duplikasi)
            if not os.path.exists(file_bin_name):
                joblib.dump(model_rf, file_bin_name)
                joblib.dump(st.session_state.scaler_aktif, f"scaler_{file_bin_name}")
                st.success(f"🎉 Sukses! Model '{file_bin_name}' & komponen scaler berhasil diekspor ke direktori.")
            else:
                st.warning(f"⚠️ Model dengan nama '{file_bin_name}' sudah terdaftar. Silakan tentukan nama versi baru untuk menghindari bentrok data!")
    else:
        st.warning("Jalankan tahapan Feature Engineering terlebih dahulu untuk menyiapkan matriks.")

# --- SIKLUS 5: EVALUATION ---
with tab5:
    st.header("Siklus 5: Evaluation & Validasi")
    st.write("Pengujian kualitas akurasi model sebelum diimplementasikan.")
    
    if st.session_state.model_terlatih is not None:
        y_prediksi = st.session_state.model_terlatih.predict(st.session_state.X_test)
        akurasi = accuracy_score(st.session_state.y_test, y_prediksi)
        skor_f1 = f1_score(st.session_state.y_test, y_prediksi)
        
        kolom1, kolom2 = st.columns(2)
        kolom1.metric("Tingkat Akurasi (Accuracy)", f"{akurasi:.2f}")
        kolom2.metric("Nilai F1-Score", f"{skor_f1:.2f}")
    else:
        st.warning("Latih model algoritma di tab Data Model terlebih dahulu.")

# --- SIKLUS 6: DATA VISUALIZATION ---
with tab6:
    st.header("Siklus 6: Data Visualization")
    st.write("Diagram visual penyebaran klasifikasi historis transaksi.")
    
    if st.session_state.df is not None:
        try:
            fig, ax = plt.subplots(facecolor='#DDE5B6')
            ax.set_facecolor('#DDE5B6')
            st.session_state.df['status_batal'].value_counts().plot(kind='bar', color=['#2E7D32', '#606C38'], ax=ax)
            ax.set_title("Distribusi Status Transaksi Historis", fontweight='bold', color='#1B5E20')
            ax.set_xticklabels(['Selesai Berhasil (0)', 'Dibatalkan (1)'], rotation=0, fontweight='bold', color='#1B5E20')
            
            ax.tick_params(colors='#1B5E20')
            ax.spines['bottom'].set_color('#1B5E20')
            ax.spines['left'].set_color('#1B5E20')
            st.pyplot(fig)
        except Exception as e:
            st.write("Visualisasi diagram belum tersedia.")
    else:
        st.warning("Silakan unggah data historis terlebih dahulu.")

# --- SIKLUS 7: DATA DECISION & INFERENCE (IMPLEMENTASI KODE DOSEN - FASE 2) ---
with tab7:
    st.header("Siklus 7: Registry Model Historis & Keputusan Bisnis")
    
    # Membaca direktori untuk mencari model yang tersimpan
    list_files = glob.glob("*.joblib")
    clean_model_list = [f for f in list_files if not f.startswith("scaler_")]
    
    if len(clean_model_list) > 0:
        model_terpilih = st.selectbox("Pilih Versi Model Acuan untuk Inferensi Data Baru:", clean_model_list)
        
        # Eksekusi pemanggilan model terpilih
        active_model = joblib.load(model_terpilih)
        active_scaler = joblib.load(f"scaler_{model_terpilih}")
        st.info(f"🚀 Konfigurasi '{model_terpilih}' aktif. Sistem siap menerima data transaksi baru.")
        
        file_baru = st.file_uploader("Unggah CSV Transaksi Baru (Massal)", type=['csv'], key="data_baru")
        
        if file_baru is not None:
            if st.button("Jalankan Mesin Prediksi Massal"):
                df_baru = pd.read_csv(file_baru)
                try:
                    X_input_baru = df_baru[['nilai_belanja', 'jumlah_cicilan', 'jumlah_item', 'frekuensi_belanja']]
                    X_input_skala = active_scaler.transform(X_input_baru)
                    
                    hasil_prediksi = active_model.predict(X_input_skala)
                    df_baru['Prediksi_Sistem_Batal'] = hasil_prediksi
                    
                    st.success("Proses Prediksi Massal Selesai!")
                    st.dataframe(df_baru)
                except KeyError:
                    st.error("Kolom pada berkas data baru tidak kompatibel dengan spesifikasi arsitektur.")
    else:
        st.error("📭 Belum ada model biner (*.joblib) yang terdaftar di direktori server ini. Selesaikan Siklus 4 terlebih dahulu.")

    st.markdown("---")
    
    # Rekomendasi Taktis (Tetap dipertahankan untuk Bab 4.2)
    st.subheader("Rekomendasi Kebijakan Bisnis Strategis (Actionable Insights)")
    st.markdown("""
    * **Operasional Taktis:** Sistem mengidentifikasi pola bahwa nilai belanja masif yang dibarengi dengan durasi cicilan yang panjang memiliki rasio pembatalan yang signifikan.
    * **Tindakan Logistik Gudang:** Terapkan penundaan proses administrasi pengemasan selama 15 menit pasca-checkout khusus untuk transaksi yang dilabeli rentan 'Batal (1)' oleh sistem prediksi.
    * **Strategi Pemasaran:** Tawarkan penyesuaian promo pengiriman secara terotomatisasi kepada kelompok pengguna dengan riwayat interaksi tinggi yang terdeteksi ragu dalam menyelesaikan pembayaran.
    """)
    