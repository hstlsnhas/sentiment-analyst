import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
from collections import Counter
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Ulasan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid;
        text-align: center;
    }
    .card-positif { border-left-color: #28a745; }
    .card-negatif { border-left-color: #dc3545; }
    .card-netral  { border-left-color: #ffc107; }
    .card-total   { border-left-color: #2d6a9f; }
    .badge-positif { background: #d4edda; color: #155724; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-negatif { background: #f8d7da; color: #721c24; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .badge-netral  { background: #fff3cd; color: #856404; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .rekomendasi-box {
        background: #f0f7ff;
        border: 1px solid #b3d4f5;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .step-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border: 1px solid #dee2e6;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Analisis Sentimen Ulasan</h1>
    <p style="margin:0;opacity:0.9;">Pipeline lengkap: IndoBERT · KeyBERT · Agregasi Sektor · Rekomendasi AI</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — KONFIGURASI
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")

    st.markdown("### 🔑 Anthropic API Key")
    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Digunakan untuk rekomendasi AI. Dapatkan di console.anthropic.com"
    )

    st.markdown("---")
    st.markdown("### 🧠 Model IndoBERT")
    indobert_model = st.selectbox(
        "Model Klasifikasi Sentimen",
        ["crypter70/IndoBERT-Sentiment-Analysis", "mdhugol/indonesia-bert-sentiment-classifier"],
        help="Model HuggingFace untuk klasifikasi sentimen bahasa Indonesia"
    )

    st.markdown("### 📌 Ekstraksi Topik")
    top_n_keywords = st.slider("Jumlah topik per ulasan", 1, 5, 3)
    top_n_aggregate = st.slider("Jumlah topik teratas (agregasi)", 3, 10, 5)

    st.markdown("### 🔧 Filter Data")
    filter_mismatch = st.checkbox(
        "Hapus rating-sentimen tidak konsisten",
        value=True,
        help="Hapus baris: rating ≥3.5 & Negatif, atau rating <3.5 & Positif"
    )

    st.markdown("---")
    st.markdown("### 📖 Alur Pipeline")
    steps = ["📂 Upload & Merge Data", "🧹 Cleaning", "🤖 Klasifikasi IndoBERT",
             "🏷️ Ekstraksi Topik (KeyBERT)", "📊 Agregasi Sektor", "💡 Rekomendasi AI"]
    for i, s in enumerate(steps, 1):
        st.markdown(f"<div class='step-box'><b>{i}.</b> {s}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS UTAMA
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Data & Pipeline",
    "🤖 Klasifikasi Sentimen",
    "🏷️ Ekstraksi Topik",
    "📊 Agregasi & Visualisasi",
    "💡 Rekomendasi AI"
])

# ═══════════════════════════════════════════
# TAB 1 — DATA & PIPELINE
# ═══════════════════════════════════════════
with tab1:
    st.markdown("## 📂 Upload & Persiapan Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### File Lama (JSON — berisi koordinat)")
        file_lama = st.file_uploader(
            "Upload file JSON lama (dengan place_lat, place_lng)",
            type=["json"],
            key="file_lama"
        )
        if file_lama:
            st.success(f"✅ {file_lama.name} ter-upload")
            df_lama_preview = pd.read_json(file_lama)
            st.caption(f"Shape: {df_lama_preview.shape} | Kolom: {list(df_lama_preview.columns)}")
            with st.expander("Preview 5 baris"):
                st.dataframe(df_lama_preview.head())

    with col2:
        st.markdown("#### File Baru (CSV — berisi ulasan & Zonasi Kawasan)")
        file_baru = st.file_uploader(
            "Upload file CSV baru (dengan kolom text, rating, sentiment, Zonasi Kawasan)",
            type=["csv"],
            key="file_baru"
        )
        if file_baru:
            st.success(f"✅ {file_baru.name} ter-upload")
            df_baru_preview = pd.read_csv(file_baru)
            st.caption(f"Shape: {df_baru_preview.shape} | Kolom: {list(df_baru_preview.columns)}")
            with st.expander("Preview 5 baris"):
                st.dataframe(df_baru_preview.head())

    st.markdown("---")

    # Tombol proses merge
    if file_lama and file_baru:
        if st.button("🔀 Merge & Cleaning Data", type="primary", use_container_width=True):
            with st.spinner("Memproses merge dan cleaning..."):
                try:
                    file_lama.seek(0)
                    file_baru.seek(0)
                    df_lama = pd.read_json(file_lama)
                    df_baru = pd.read_csv(file_baru)

                    # Merge
                    df_coords = df_lama[['text', 'place_lat', 'place_lng']] if 'place_lat' in df_lama.columns else df_lama[['text']]
                    df_merged = pd.merge(df_baru, df_coords, on='text', how='left')

                    # Rename sector
                    if 'Zonasi Kawasan' in df_merged.columns:
                        df_merged['sector'] = df_merged['Zonasi Kawasan']
                        df_merged = df_merged.drop(columns=['Zonasi Kawasan'])

                    # Cleaning: hapus di luar area
                    if 'sector' in df_merged.columns:
                        df_clean = df_merged[df_merged['sector'] != 'di luar area'].copy()
                        df_clean = df_clean.dropna(subset=['sector'])
                    else:
                        df_clean = df_merged.copy()

                    # Filter rating-sentimen mismatch
                    if filter_mismatch and 'rating' in df_clean.columns and 'sentiment' in df_clean.columns:
                        before = len(df_clean)
                        df_clean = df_clean[
                            ~((df_clean['rating'] >= 3.5) & (df_clean['sentiment'] == 'Negatif')) &
                            ~((df_clean['rating'] < 3.5) & (df_clean['sentiment'] == 'Positif'))
                        ]
                        removed = before - len(df_clean)
                        st.info(f"🔧 Filter mismatch: {removed} baris dihapus")

                    # Cleaning text_clean
                    if 'text_clean' in df_clean.columns:
                        df_clean['text_clean'] = df_clean['text_clean'].replace(r'^\s*$', np.nan, regex=True)
                        df_clean = df_clean.dropna(subset=['text_clean'])
                    df_clean = df_clean.reset_index(drop=True)

                    st.session_state['df'] = df_clean
                    st.success(f"✅ Data berhasil diproses! Total: **{len(df_clean)}** baris siap dianalisis.")

                    # Metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Baris", len(df_clean))
                    c2.metric("Sektor Unik", df_clean['sector'].nunique() if 'sector' in df_clean.columns else "—")
                    c3.metric("Kolom", len(df_clean.columns))
                    missing_pct = (df_clean.isnull().sum().sum() / df_clean.size * 100)
                    c4.metric("Missing %", f"{missing_pct:.1f}%")

                    with st.expander("📋 Preview Data Bersih"):
                        st.dataframe(df_clean.head(20), use_container_width=True)

                    # Download
                    csv_out = df_clean.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Data Bersih (CSV)",
                                       csv_out, "sentiment_clean.csv", "text/csv")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    elif not file_lama and not file_baru:
        # Demo mode
        st.info("💡 Belum punya file? Gunakan **Demo Mode** di bawah untuk mencoba pipeline.")
        if st.button("🎮 Gunakan Data Demo", use_container_width=True):
            demo_data = {
                'text': [
                    'Pelabuhan sangat bersih dan tertib',
                    'Antrian loket sangat panjang dan tidak ada peneduh',
                    'Toilet umum kotor dan berbau tidak sedap',
                    'Petugas ramah dan membantu wisatawan',
                    'Area parkir sangat sempit dan tidak teratur',
                    'Fasilitas food court lengkap dan harganya terjangkau',
                    'Kapal sering telat berangkat tanpa pemberitahuan',
                    'View pemandangan laut sangat indah',
                    'Kurangnya tempat duduk di ruang tunggu',
                    'Keamanan terjaga dengan baik oleh petugas',
                    'Harga tiket naik tapi pelayanan tidak meningkat',
                    'Akses jalan menuju pelabuhan cukup bagus',
                    'Sistem informasi keberangkatan sudah digital',
                    'Banyak pedagang liar yang mengganggu pengunjung',
                    'Dermaga baru sangat luas dan nyaman',
                ],
                'text_clean': [
                    'pelabuhan sangat bersih dan tertib',
                    'antrian loket sangat panjang tidak ada peneduh',
                    'toilet umum kotor berbau tidak sedap',
                    'petugas ramah membantu wisatawan',
                    'area parkir sempit tidak teratur',
                    'fasilitas food court lengkap harga terjangkau',
                    'kapal sering telat berangkat tanpa pemberitahuan',
                    'pemandangan laut sangat indah',
                    'kurang tempat duduk ruang tunggu',
                    'keamanan terjaga petugas',
                    'harga tiket naik pelayanan tidak meningkat',
                    'akses jalan menuju pelabuhan bagus',
                    'sistem informasi keberangkatan digital',
                    'pedagang liar mengganggu pengunjung',
                    'dermaga baru luas nyaman',
                ],
                'rating': [4.5, 2.0, 1.5, 5.0, 2.0, 4.0, 1.5, 4.5, 3.0, 4.5, 2.5, 3.5, 4.0, 2.0, 5.0],
                'sentiment': ['Positif','Negatif','Negatif','Positif','Negatif','Positif',
                               'Negatif','Positif','Netral','Positif','Negatif','Netral','Positif','Negatif','Positif'],
                'place_name': ['Terminal A','Terminal A','Terminal B','Terminal B','Parkir','Food Court',
                                'Dermaga 1','Dermaga 1','Ruang Tunggu','Pintu Masuk','Loket','Akses Jalan',
                                'Terminal B','Food Court','Dermaga 2'],
                'sector': ['Terminal','Terminal','Terminal','Terminal','Parkir','Fasilitas Komersil',
                            'Dermaga','Dermaga','Terminal','Keamanan','Loket & Tiket','Aksesibilitas',
                            'Terminal','Fasilitas Komersil','Dermaga'],
                'place_lat': [-5.85] * 15,
                'place_lng': [105.87] * 15,
            }
            df_demo = pd.DataFrame(demo_data)
            st.session_state['df'] = df_demo
            st.success(f"✅ Data demo dimuat! {len(df_demo)} baris siap dianalisis.")
            st.dataframe(df_demo, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — KLASIFIKASI SENTIMEN (IndoBERT)
# ═══════════════════════════════════════════
with tab2:
    st.markdown("## 🤖 Klasifikasi Sentimen dengan IndoBERT")

    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload dan proses data di Tab 1 terlebih dahulu.")
    else:
        df = st.session_state['df']
        st.success(f"✅ Data tersedia: **{len(df)}** baris")

        # Cek apakah sudah ada kolom sentiment dari IndoBERT
        has_indobert = 'indobert_sentiment' in df.columns

        col1, col2 = st.columns([2, 1])
        with col1:
            if has_indobert:
                st.info("✅ Klasifikasi IndoBERT sudah dijalankan sebelumnya.")
            else:
                st.markdown("""
                **Model yang digunakan:** `crypter70/IndoBERT-Sentiment-Analysis`

                IndoBERT adalah model BERT yang dilatih khusus untuk bahasa Indonesia.
                Model ini akan mengklasifikasikan setiap ulasan ke dalam 3 kategori:
                **Positif**, **Negatif**, atau **Netral**.
                """)

        with col2:
            batch_size = st.number_input("Batch Size", 8, 64, 32,
                                          help="Sesuaikan dengan RAM/GPU yang tersedia")

        # Info tentang runtime
        st.warning("""
        ⚠️ **Catatan:** IndoBERT memerlukan download model (~400MB) dan komputasi yang cukup besar.
        Proses ini berjalan optimal di mesin dengan GPU (seperti Google Colab).
        Di Streamlit Cloud, gunakan **mode manual upload** jika sudah punya hasil JSON.
        """)

        col_run, col_upload = st.columns(2)

        with col_run:
            run_indobert = st.button("🚀 Jalankan IndoBERT", type="primary", use_container_width=True)

        with col_upload:
            st.markdown("**Atau upload hasil JSON yang sudah ada:**")
            json_result = st.file_uploader("Upload JSON hasil IndoBERT",
                                            type=["json"], key="json_indobert")

        if json_result:
            try:
                df_result = pd.read_json(json_result)
                # Mapping kolom
                if 'sentiment' in df_result.columns:
                    st.session_state['df']['indobert_sentiment'] = df_result['sentiment'].values[:len(df)]
                    if 'score' in df_result.columns:
                        st.session_state['df']['indobert_score'] = df_result['score'].values[:len(df)]
                    st.success("✅ Hasil IndoBERT berhasil dimuat dari file JSON!")
                    df = st.session_state['df']
            except Exception as e:
                st.error(f"❌ Error: {e}")

        if run_indobert:
            if 'text_clean' not in df.columns:
                st.error("❌ Kolom 'text_clean' tidak ditemukan di data!")
            else:
                try:
                    import torch
                    from transformers import pipeline as hf_pipeline

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("📥 Memuat model IndoBERT...")

                    device = 0 if torch.cuda.is_available() else -1
                    device_label = "GPU 🚀" if device == 0 else "CPU 🖥️"
                    st.info(f"Menggunakan: **{device_label}**")

                    classifier = hf_pipeline(
                        "text-classification",
                        model=indobert_model,
                        device=device
                    )

                    status_text.text("🤖 Menjalankan klasifikasi...")
                    texts = df['text_clean'].fillna("").tolist()

                    mapping_label = {"NEUTRAL": "Netral", "POSITIVE": "Positif",
                                     "NEGATIVE": "Negatif", "LABEL_0": "Negatif",
                                     "LABEL_1": "Netral", "LABEL_2": "Positif"}

                    results = []
                    chunk = batch_size
                    total_chunks = (len(texts) + chunk - 1) // chunk

                    for i in range(0, len(texts), chunk):
                        batch = texts[i:i+chunk]
                        out = classifier(batch, batch_size=chunk, truncation=True, max_length=512)
                        results.extend(out)
                        pct = min((i + chunk) / len(texts), 1.0)
                        progress_bar.progress(pct)
                        status_text.text(f"🔄 Diproses: {min(i+chunk, len(texts))}/{len(texts)}")

                    progress_bar.progress(1.0)
                    status_text.text("✅ Selesai!")

                    df['indobert_sentiment'] = [
                        mapping_label.get(x['label'].upper(), x['label']) for x in results
                    ]
                    df['indobert_score'] = [x['score'] for x in results]
                    st.session_state['df'] = df

                    st.success("✅ Klasifikasi selesai!")

                    # Export
                    json_out = df.to_json(orient='records', indent=4, force_ascii=False)
                    st.download_button("⬇️ Download Hasil JSON",
                                       json_out.encode('utf-8'),
                                       "1final_sentiment.json",
                                       "application/json")

                except ImportError:
                    st.error("❌ Library `transformers` atau `torch` belum terinstall. Jalankan: `pip install transformers torch`")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Tampilkan hasil jika sudah ada
        if 'df' in st.session_state:
            df = st.session_state['df']
            sentiment_col = 'indobert_sentiment' if 'indobert_sentiment' in df.columns else \
                            'sentiment' if 'sentiment' in df.columns else None

            if sentiment_col:
                st.markdown("---")
                st.markdown("### 📊 Distribusi Sentimen")

                vc = df[sentiment_col].value_counts()
                total = len(df)

                c1, c2, c3, c4 = st.columns(4)
                pos = vc.get('Positif', 0)
                neg = vc.get('Negatif', 0)
                net = vc.get('Netral', 0)

                with c1:
                    st.markdown(f"<div class='metric-card card-total'><h2>{total}</h2><p>Total Ulasan</p></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='metric-card card-positif'><h2>😊 {pos}</h2><p>Positif ({pos/total*100:.1f}%)</p></div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='metric-card card-negatif'><h2>😞 {neg}</h2><p>Negatif ({neg/total*100:.1f}%)</p></div>", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"<div class='metric-card card-netral'><h2>😐 {net}</h2><p>Netral ({net/total*100:.1f}%)</p></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Tabel sample
                with st.expander("📋 Sample Hasil Klasifikasi"):
                    cols_show = ['text_clean', sentiment_col] + \
                                (['indobert_score'] if 'indobert_score' in df.columns else []) + \
                                (['sector'] if 'sector' in df.columns else [])
                    cols_show = [c for c in cols_show if c in df.columns]
                    st.dataframe(df[cols_show].head(30), use_container_width=True)

# ═══════════════════════════════════════════
# TAB 3 — EKSTRAKSI TOPIK (KeyBERT)
# ═══════════════════════════════════════════
with tab3:
    st.markdown("## 🏷️ Ekstraksi Topik dengan KeyBERT + IndoBERT")

    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan proses data di Tab 1 terlebih dahulu.")
    else:
        df = st.session_state['df']

        st.markdown("""
        KeyBERT menggunakan embedding IndoBERT (`indobenchmark/indobert-base-p2`) untuk mengekstrak
        kata kunci yang paling representatif dari setiap ulasan.
        """)

        # Stop words
        stop_words_id = [
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dalam', 'dengan', 'untuk', 'adalah',
            'itu', 'ini', 'saya', 'kami', 'kita', 'anda', 'dia', 'mereka', 'apa', 'mana',
            'siapa', 'kapan', 'mengapa', 'bagaimana', 'bisa', 'boleh', 'ada', 'tidak', 'tak',
            'jangan', 'sudah', 'telah', 'sedang', 'akan', 'juga', 'saja', 'hanya', 'cuma',
            'tetapi', 'tapi', 'namun', 'karena', 'kalau', 'jika', 'seperti', 'bagi', 'serta',
            'sangat', 'sekali', 'banget', 'bgt', 'amat', 'terlalu', 'begitu', 'cukup',
            'lagi', 'tadi', 'nanti', 'besok', 'kemarin', 'sekarang', 'skrg', 'dulu',
            'baru', 'lama', 'bentar', 'sebentar', 'jam', 'menit', 'hari', 'bulan', 'tahun',
            'yg', 'dg', 'dgn', 'klo', 'kl', 'kalo', 'udh', 'sdh', 'udah', 'gak', 'ga', 'gk',
            'aja', 'aj', 'jd', 'jadi', 'krn', 'karna', 'utk', 'untuk', 'pas', 'nih', 'tuh',
            'loh', 'kok', 'deh', 'lah', 'sih', 'kan', 'dong', 'yah', 'ya', 'd', 'pd', 'sama',
            'ama', 'biar', 'buat', 'biarpun', 'dr', 'drpd', 'bagus', 'mantap', 'mantab', 'oke',
            'ok', 'mantul', 'keren', 'enak', 'favorit', 'nyaman', 'bersih', 'ramah', 'cepat',
            'lancar', 'murah', 'mahal', 'lumayan', 'best', 'semoga', 'sukses', 'sip',
            'bakauheni', 'merak', 'lampung', 'pelabuhan', 'dermaga', 'kapal', 'ferry',
            'terminal', 'eksekutif', 'executive', 'mas', 'mun', 'toko', 'ayu', 'rm',
            'daerah', 'tempat', 'lokasi', 'area', 'sektor', 'district'
        ]

        with st.expander("📝 Edit Stop Words"):
            sw_text = st.text_area("Stop Words (pisahkan dengan koma)",
                                   value=", ".join(stop_words_id),
                                   height=150)
            stop_words_id = [w.strip() for w in sw_text.split(',') if w.strip()]
            st.caption(f"Total stop words: {len(stop_words_id)}")

        col1, col2 = st.columns(2)
        with col1:
            ngram_min = st.number_input("N-gram min", 1, 3, 1)
        with col2:
            ngram_max = st.number_input("N-gram max", 1, 3, 3)

        st.warning("""
        ⚠️ KeyBERT juga membutuhkan download model (~1GB). Atau upload JSON yang sudah ada hasil ekstraksi topik.
        """)

        col_run2, col_upload2 = st.columns(2)
        with col_run2:
            run_keybert = st.button("🚀 Jalankan Ekstraksi Topik", type="primary", use_container_width=True)
        with col_upload2:
            json_topic = st.file_uploader("Upload JSON hasil ekstraksi topik",
                                           type=["json"], key="json_topic")

        if json_topic:
            try:
                df_topic = pd.read_json(json_topic)
                if 'topic' in df_topic.columns:
                    st.session_state['df']['topic'] = df_topic['topic'].values[:len(df)]
                    st.success("✅ Hasil topik berhasil dimuat!")
                    df = st.session_state['df']
            except Exception as e:
                st.error(f"❌ Error: {e}")

        if run_keybert:
            if 'text_clean' not in df.columns:
                st.error("❌ Kolom 'text_clean' tidak ditemukan!")
            else:
                try:
                    from keybert import KeyBERT

                    progress = st.progress(0)
                    status = st.empty()
                    status.text("📥 Memuat model KeyBERT + IndoBERT...")

                    kw_model = KeyBERT(model='indobenchmark/indobert-base-p2')

                    def extract_topics(text):
                        if not text or len(str(text)) < 5:
                            return ""
                        keywords = kw_model.extract_keywords(
                            str(text),
                            keyphrase_ngram_range=(ngram_min, ngram_max),
                            stop_words=stop_words_id,
                            top_n=top_n_keywords
                        )
                        return ", ".join([k[0] for k in keywords])

                    texts = df['text_clean'].fillna("").tolist()
                    topics = []
                    for i, t in enumerate(texts):
                        topics.append(extract_topics(t))
                        progress.progress((i + 1) / len(texts))
                        if i % 50 == 0:
                            status.text(f"🔄 {i+1}/{len(texts)} ulasan diproses...")

                    df['topic'] = topics
                    st.session_state['df'] = df
                    status.text("✅ Selesai!")
                    st.success("✅ Ekstraksi topik selesai!")

                    json_out = df.to_json(orient='records', indent=4, force_ascii=False)
                    st.download_button("⬇️ Download Hasil (JSON)",
                                       json_out.encode('utf-8'),
                                       "sentiment_with_topics.json",
                                       "application/json")

                except ImportError:
                    st.error("❌ Library `keybert` belum terinstall. Jalankan: `pip install keybert`")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        # Tampilkan hasil topik
        if 'topic' in st.session_state.get('df', pd.DataFrame()).columns:
            df = st.session_state['df']
            st.markdown("---")
            st.markdown("### 📋 Hasil Ekstraksi Topik")

            with st.expander("Lihat Sample"):
                cols = ['text_clean', 'topic'] + (['sentiment'] if 'sentiment' in df.columns else [])
                st.dataframe(df[[c for c in cols if c in df.columns]].head(20), use_container_width=True)

            # Word frequency chart
            st.markdown("### ☁️ Frekuensi Topik Terpopuler")
            all_topics = ", ".join(df['topic'].dropna().astype(str)).split(', ')
            freq = Counter([t.strip() for t in all_topics if len(t.strip()) > 2])
            top_topics = pd.DataFrame(freq.most_common(20), columns=['Topik', 'Frekuensi'])

            if not top_topics.empty:
                st.bar_chart(top_topics.set_index('Topik')['Frekuensi'])

# ═══════════════════════════════════════════
# TAB 4 — AGREGASI & VISUALISASI
# ═══════════════════════════════════════════
with tab4:
    st.markdown("## 📊 Agregasi Per Sektor & Lokasi")

    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan proses data di Tab 1 terlebih dahulu.")
    else:
        df = st.session_state['df']

        # Deteksi kolom sentimen
        sentiment_col = 'indobert_sentiment' if 'indobert_sentiment' in df.columns else \
                        'sentiment' if 'sentiment' in df.columns else None

        if not sentiment_col:
            st.warning("⚠️ Kolom sentimen belum tersedia. Jalankan klasifikasi di Tab 2.")
        else:
            if st.button("📊 Hitung Agregasi", type="primary", use_container_width=True):
                try:
                    # Helper
                    def get_top_topics_with_counts(series, n=5):
                        all_text = ", ".join(series.astype(str))
                        phrases = [p.strip() for p in all_text.split(',') if len(p.strip()) > 2]
                        counts = Counter(phrases).most_common(n)
                        return ", ".join([f"{t} ({c})" for t, c in counts])

                    def split_topic_data(val):
                        if pd.isna(val) or not val:
                            return None, None
                        parts = [p.strip() for p in val.split(',') if '(' in p]
                        if not parts:
                            return val, None
                        topics = [p.split('(')[0].strip() for p in parts]
                        counts = [re.search(r'\((\d+)\)', p).group(1) if re.search(r'\((\d+)\)', p) else '0' for p in parts]
                        return ", ".join(topics), ", ".join(counts)

                    # Agregasi per sektor & sentimen
                    agg_params = {
                        'jumlah_ulasan': (df['text_clean'] if 'text_clean' in df.columns else df.index.to_series(), 'count')
                    }
                    if 'topic' in df.columns:
                        agg_params['topik_dan_jumlah'] = ('topic', lambda x: get_top_topics_with_counts(x, top_n_aggregate))

                    group_cols = ['sector', sentiment_col] if 'sector' in df.columns else [sentiment_col]

                    df_agg = df.groupby(group_cols).agg(
                        jumlah_ulasan=('text_clean' if 'text_clean' in df.columns else df.columns[0], 'count'),
                        **({'topik_dan_jumlah': ('topic', lambda x: get_top_topics_with_counts(x, top_n_aggregate))} if 'topic' in df.columns else {})
                    ).reset_index()
                    df_agg = df_agg.rename(columns={sentiment_col: 'sentiment'})

                    # ALL SECTOR
                    df_all = df.groupby(sentiment_col).agg(
                        jumlah_ulasan=('text_clean' if 'text_clean' in df.columns else df.columns[0], 'count'),
                        **({'topik_dan_jumlah': ('topic', lambda x: get_top_topics_with_counts(x, top_n_aggregate))} if 'topic' in df.columns else {})
                    ).reset_index()
                    df_all = df_all.rename(columns={sentiment_col: 'sentiment'})
                    df_all['sector'] = 'ALL SECTOR'

                    df_final_agg = pd.concat([df_agg, df_all], ignore_index=True)

                    # Split topik
                    if 'topik_dan_jumlah' in df_final_agg.columns:
                        df_final_agg[['top_topics', 'topic_counts']] = df_final_agg['topik_dan_jumlah'].apply(
                            lambda x: pd.Series(split_topic_data(x))
                        )

                    df_final_agg = df_final_agg.sort_values(['sector', 'sentiment'], ascending=[True, False])
                    st.session_state['df_agregasi'] = df_final_agg

                    # ── AGREGASI LOKASI ──
                    if 'place_name' in df.columns:
                        df_lokasi = df.pivot_table(
                            index=['place_name', 'sector'] if 'sector' in df.columns else ['place_name'],
                            columns=sentiment_col,
                            aggfunc='size',
                            fill_value=0
                        ).reset_index()
                        df_lokasi['total_ulasan'] = df_lokasi.get('Positif', 0) + \
                                                     df_lokasi.get('Negatif', 0) + \
                                                     df_lokasi.get('Netral', 0)
                        if 'sector' in df_lokasi.columns:
                            df_lokasi = df_lokasi.sort_values(['sector', 'total_ulasan'], ascending=[True, False])
                        st.session_state['df_lokasi'] = df_lokasi

                    st.success("✅ Agregasi selesai!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Tampilkan jika sudah ada
            if 'df_agregasi' in st.session_state:
                df_final_agg = st.session_state['df_agregasi']

                st.markdown("### 📋 Tabel Agregasi Sektor × Sentimen")

                # Filter sektor
                sectors_available = df_final_agg['sector'].unique().tolist()
                selected_sectors = st.multiselect(
                    "Filter Sektor:",
                    sectors_available,
                    default=sectors_available,
                    key="filter_sector_agg"
                )

                df_display = df_final_agg[df_final_agg['sector'].isin(selected_sectors)]

                # Color rows
                def color_sentiment(row):
                    colors = {'Positif': '#d4edda', 'Negatif': '#f8d7da', 'Netral': '#fff3cd'}
                    return [f'background-color: {colors.get(row["sentiment"], "")}'] * len(row)

                st.dataframe(
                    df_display.style.apply(color_sentiment, axis=1),
                    use_container_width=True,
                    height=400
                )

                # Pivot chart
                if 'sector' in df_final_agg.columns and 'sector' in df_display.columns:
                    st.markdown("### 📈 Distribusi Sentimen per Sektor")
                    df_chart = df_display[df_display['sector'] != 'ALL SECTOR'].pivot_table(
                        index='sector', columns='sentiment', values='jumlah_ulasan', fill_value=0
                    )
                    if not df_chart.empty:
                        st.bar_chart(df_chart)

                # Agregasi lokasi
                if 'df_lokasi' in st.session_state:
                    st.markdown("### 🗺️ Agregasi Per Lokasi")
                    st.dataframe(st.session_state['df_lokasi'], use_container_width=True)

                # Downloads
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    json_agg = df_final_agg.to_json(orient='records', indent=4, force_ascii=False)
                    st.download_button("⬇️ Download Agregasi JSON",
                                       json_agg.encode('utf-8'),
                                       "1final_aggregate.json", "application/json")
                with col_dl2:
                    csv_agg = df_final_agg.to_csv(index=False).encode('utf-8')
                    st.download_button("⬇️ Download Agregasi CSV",
                                       csv_agg, "1final_aggregate.csv", "text/csv")

                # Build structured JSON
                st.markdown("### 🗂️ Structured JSON (per sektor)")
                if st.button("🔧 Build Structured JSON"):
                    structured_data = {}
                    for _, row in df_final_agg.iterrows():
                        sec = row['sector']
                        if sec not in structured_data:
                            structured_data[sec] = {
                                "sector": sec,
                                "total_ulasan": 0,
                                "all_topik_dan_jumlah_list": [],
                                "sentiment_breakdown": []
                            }
                        jml = int(row['jumlah_ulasan']) if row['jumlah_ulasan'] else 0
                        structured_data[sec]["total_ulasan"] += jml
                        if 'topik_dan_jumlah' in row and row['topik_dan_jumlah']:
                            structured_data[sec]["all_topik_dan_jumlah_list"].append(str(row['topik_dan_jumlah']))
                        breakdown = {
                            "sentiment": row['sentiment'],
                            "jumlah_ulasan": jml,
                        }
                        if 'top_topics' in row:
                            breakdown["top_topics"] = row.get('top_topics')
                        if 'topic_counts' in row:
                            breakdown["topic_counts"] = row.get('topic_counts')
                        structured_data[sec]["sentiment_breakdown"].append(breakdown)

                    final_output = [{
                        "sector": v["sector"],
                        "total_ulasan": v["total_ulasan"],
                        "all_topik_dan_jumlah": ", ".join(v["all_topik_dan_jumlah_list"]),
                        "sentiment_breakdown": v["sentiment_breakdown"]
                    } for v in structured_data.values()]

                    st.session_state['final_output'] = final_output
                    st.success(f"✅ Structured JSON siap: {len(final_output)} sektor")

                    json_struct = json.dumps(final_output, indent=4, ensure_ascii=False)
                    st.download_button("⬇️ Download Structured JSON",
                                       json_struct.encode('utf-8'),
                                       "1final_summary.json", "application/json")

# ═══════════════════════════════════════════
# TAB 5 — REKOMENDASI AI
# ═══════════════════════════════════════════
with tab5:
    st.markdown("## 💡 Rekomendasi AI (Powered by Claude)")

    if 'df_agregasi' not in st.session_state and 'final_output' not in st.session_state:
        st.warning("⚠️ Silakan selesaikan agregasi di Tab 4 terlebih dahulu.")
    else:
        # Pilih mode
        mode_rec = st.radio(
            "Mode Rekomendasi:",
            ["Per Sektor (Komprehensif)", "Per Sentimen (Detail)"],
            horizontal=True
        )

        if not api_key_input:
            st.error("❌ Masukkan **Anthropic API Key** di sidebar kiri untuk menggunakan fitur ini.")
        else:
            st.success("✅ API Key terdeteksi")

            # Build data untuk dikirim ke AI
            if 'final_output' in st.session_state:
                data_for_ai = st.session_state['final_output']
            elif 'df_agregasi' in st.session_state:
                df_agg = st.session_state['df_agregasi']
                data_for_ai = []
                for sec in df_agg['sector'].unique():
                    df_sec = df_agg[df_agg['sector'] == sec]
                    data_for_ai.append({
                        "sector": sec,
                        "total_ulasan": int(df_sec['jumlah_ulasan'].sum()),
                        "all_topik_dan_jumlah": ", ".join(df_sec['topik_dan_jumlah'].dropna().astype(str).tolist()) if 'topik_dan_jumlah' in df_sec.columns else "",
                        "sentiment_breakdown": df_sec[['sentiment', 'jumlah_ulasan']].to_dict(orient='records')
                    })
            else:
                data_for_ai = []

            # Preview data
            with st.expander("📋 Preview Data yang akan dianalisis AI"):
                st.json(data_for_ai[:2] if len(data_for_ai) > 2 else data_for_ai)

            # Pilih sektor
            sectors = [item['sector'] for item in data_for_ai]
            selected_sectors_ai = st.multiselect(
                "Pilih Sektor untuk dianalisis:",
                sectors,
                default=sectors[:min(3, len(sectors))],
                key="sectors_ai"
            )

            if st.button("🤖 Generate Rekomendasi AI", type="primary", use_container_width=True):
                if not selected_sectors_ai:
                    st.warning("Pilih minimal 1 sektor.")
                else:
                    data_selected = [item for item in data_for_ai if item['sector'] in selected_sectors_ai]

                    progress = st.progress(0)
                    results_rec = {}

                    for idx, item in enumerate(data_selected):
                        sec = item['sector']
                        st.markdown(f"#### 🔄 Memproses: **{sec}**...")

                        if mode_rec == "Per Sektor (Komprehensif)":
                            prompt = f"""Anda adalah konsultan bisnis profesional bidang pariwisata dan pelayanan publik.

Nama Sektor: {item['sector']}
Total Ulasan: {item['total_ulasan']}
Topik Utama: {item.get('all_topik_dan_jumlah', 'N/A')}
Detail Sentimen: {json.dumps(item['sentiment_breakdown'], indent=2, ensure_ascii=False)}

Berdasarkan data di atas, buatlah satu paragraf kesimpulan (maksimal 150 kata) yang mencakup:
1. Analisis masalah utama (dari sisi negatif/netral)
2. Peluang strategis yang bisa dikembangkan (dari sisi positif)  
3. Rekomendasi tindakan konkret untuk meningkatkan kepuasan pengunjung

Gunakan Bahasa Indonesia profesional dan lugas. Hasilnya harus SATU paragraf saja."""

                        else:  # Per sentimen
                            paragraphs = []
                            for bd in item['sentiment_breakdown']:
                                s = bd.get('sentiment', '')
                                topics_bd = bd.get('top_topics', item.get('all_topik_dan_jumlah', 'N/A'))
                                if s == 'Positif':
                                    instr = "Berikan saran apresiasi dan strategi promosi untuk mempertahankan keunggulan ini."
                                elif s == 'Negatif':
                                    instr = "Berikan solusi perbaikan konkret dan langkah-langkah perbaikan prioritas."
                                else:
                                    instr = "Berikan rekomendasi untuk mengkonversi sentimen netral menjadi positif."
                                paragraphs.append(f"Sentimen: {s}\nTopik: {topics_bd}\nInstruksi: {instr}")

                            prompt = f"""Anda adalah konsultan strategis pariwisata.

Sektor: {item['sector']}
Total Ulasan: {item['total_ulasan']}

{chr(10).join(paragraphs)}

Untuk setiap sentimen di atas, berikan 1 paragraf rekomendasi singkat dan spesifik.
Gunakan Bahasa Indonesia profesional."""

                        try:
                            import urllib.request
                            import urllib.error

                            payload = json.dumps({
                                "model": "claude-sonnet-4-6",
                                "max_tokens": 1000,
                                "system": "Anda adalah konsultan strategis bisnis pariwisata Indonesia yang berpengalaman.",
                                "messages": [{"role": "user", "content": prompt}]
                            }).encode('utf-8')

                            req = urllib.request.Request(
                                "https://api.anthropic.com/v1/messages",
                                data=payload,
                                headers={
                                    "Content-Type": "application/json",
                                    "x-api-key": api_key_input,
                                    "anthropic-version": "2023-06-01"
                                },
                                method="POST"
                            )

                            with urllib.request.urlopen(req) as resp:
                                result = json.loads(resp.read().decode('utf-8'))
                                rekomendasi = result['content'][0]['text']

                        except Exception as e:
                            rekomendasi = f"⚠️ Error: {str(e)}"

                        results_rec[sec] = rekomendasi
                        item['rekomendasi_ai'] = rekomendasi

                        # Tampilkan hasil langsung
                        st.markdown(f"**{sec}**")
                        st.markdown(f"<div class='rekomendasi-box'>{rekomendasi}</div>",
                                    unsafe_allow_html=True)

                        progress.progress((idx + 1) / len(data_selected))
                        time.sleep(0.3)

                    st.session_state['results_rec'] = results_rec
                    st.session_state['final_with_rec'] = data_for_ai
                    st.success("✅ Semua rekomendasi selesai dibuat!")

                    # Export
                    json_final = json.dumps(data_for_ai, indent=4, ensure_ascii=False)
                    st.download_button(
                        "⬇️ Download Hasil Lengkap + Rekomendasi (JSON)",
                        json_final.encode('utf-8'),
                        "1final_summary_with_recommend.json",
                        "application/json",
                        use_container_width=True
                    )

            # Tampilkan hasil sebelumnya
            elif 'results_rec' in st.session_state:
                st.markdown("### 📋 Hasil Rekomendasi Sebelumnya")
                for sec, rec in st.session_state['results_rec'].items():
                    st.markdown(f"**{sec}**")
                    st.markdown(f"<div class='rekomendasi-box'>{rec}</div>",
                                unsafe_allow_html=True)

                if 'final_with_rec' in st.session_state:
                    json_final = json.dumps(st.session_state['final_with_rec'], indent=4, ensure_ascii=False)
                    st.download_button("⬇️ Download Hasil Lengkap (JSON)",
                                       json_final.encode('utf-8'),
                                       "1final_summary_with_recommend.json",
                                       "application/json")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:1rem 0;">
    Analisis Sentimen Ulasan · IndoBERT + KeyBERT + Claude AI · Built with Streamlit
</div>
""", unsafe_allow_html=True)
