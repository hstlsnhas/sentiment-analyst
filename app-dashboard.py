import streamlit as st
import pandas as pd
import json
import re
from collections import Counter
import math
from pathlib import Path
import os
import pydeck as pdk

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Sentimen",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0F1923; color: #E8EDF2; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0B1520 !important;
    border-right: 1px solid #1E2D3D;
}
[data-testid="stSidebar"] * { color: #E8EDF2 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem 2rem; max-width: 100%; }

/* ── Cards ── */
.card {
    background: #1E2D3D;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #243447;
    height: 100%;
}
.card-sm {
    background: #1E2D3D;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #243447;
}

/* ── KPI Metric ── */
.kpi-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
.kpi-label { font-size: 0.78rem; font-weight: 500; color: #8899AA; letter-spacing: 0.08em; text-transform: uppercase; }
.kpi-sub   { font-size: 0.82rem; color: #8899AA; margin-top: 4px; }

/* ── Sentiment colors ── */
.pos  { color: #00D4AA; }
.neg  { color: #FF5C5C; }
.neu  { color: #FFB347; }

/* ── Sentiment bar ── */
.sent-bar-wrap { display: flex; border-radius: 4px; overflow: hidden; height: 6px; width: 100%; background: #243447; margin: 4px 0 0 0; }
.sent-bar-pos  { background: #00D4AA; height: 6px; }
.sent-bar-neg  { background: #FF5C5C; height: 6px; }
.sent-bar-neu  { background: #FFB347; height: 6px; }

/* ── Sector nav button ── */
.sector-btn {
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    padding: 0.6rem 0.8rem;
    margin: 2px 0;
    cursor: pointer;
    width: 100%;
    text-align: left;
    border-radius: 0 8px 8px 0;
    transition: all 0.15s;
}
.sector-btn:hover  { background: #1E2D3D; }
.sector-btn.active { background: #1E2D3D; border-left-color: #00D4AA; }

/* ── Topic chip ── */
.topic-chip {
    display: inline-block;
    background: #243447;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.78rem;
    margin: 2px 3px;
    color: #B0C4D8;
}
.topic-chip-pos { background: #0d3d30; color: #00D4AA; border: 1px solid #00D4AA33; }
.topic-chip-neg { background: #3d1515; color: #FF5C5C; border: 1px solid #FF5C5C33; }
.topic-chip-neu { background: #3d2e0d; color: #FFB347; border: 1px solid #FFB34733; }

/* ── Recommendation box ── */
.rec-box {
    background: linear-gradient(135deg, #13243a 0%, #1a2d40 100%);
    border: 1px solid #00D4AA33;
    border-left: 3px solid #00D4AA;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #C8D8E8;
    margin: 0.4rem 0;
}

/* ── Section header ── */
.sec-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #556677;
    margin: 1.4rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1E2D3D;
}

/* ── Table styling ── */
.dataframe { background: #1E2D3D !important; }

/* ── Map container ── */
.map-wrap { border-radius: 12px; overflow: hidden; border: 1px solid #243447; }

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-pos { background: #0d3d30; color: #00D4AA; }
.badge-neg { background: #3d1515; color: #FF5C5C; }
.badge-neu { background: #3d2e0d; color: #FFB347; }

/* ── Scrollable ulasan list ── */
.ulasan-item {
    border-bottom: 1px solid #1E2D3D;
    padding: 0.75rem 0;
    font-size: 0.875rem;
    line-height: 1.6;
}
.ulasan-meta { font-size: 0.75rem; color: #556677; margin-bottom: 4px; }

/* ── Upload zone ── */
.upload-zone {
    background: #1E2D3D;
    border: 2px dashed #243447;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #556677;
}

/* ── Streamlit overrides ── */
.stSelectbox > div > div { background: #1E2D3D !important; border-color: #243447 !important; }
.stFileUploader > div { background: #1E2D3D !important; }
div[data-testid="stMetric"] { background: #1E2D3D; border-radius: 10px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def parse_topics(topic_str):
    """Parse 'topik (n), topik2 (n2)' → list of (topic, count)"""
    if not topic_str or pd.isna(topic_str):
        return []
    items = []
    for part in str(topic_str).split(','):
        part = part.strip()
        m = re.match(r'^(.+?)\s*\((\d+)\)\s*$', part)
        if m:
            items.append((m.group(1).strip(), int(m.group(2))))
        elif part:
            items.append((part, 1))
    return items

def sentiment_bar_html(pos, neg, neu):
    total = pos + neg + neu
    if total == 0: return ""
    pp = pos/total*100; np_ = neg/total*100; nup = neu/total*100
    return f"""<div class="sent-bar-wrap">
        <div class="sent-bar-pos" style="width:{pp:.1f}%"></div>
        <div class="sent-bar-neg" style="width:{np_:.1f}%"></div>
        <div class="sent-bar-neu" style="width:{nup:.1f}%"></div>
    </div>"""

def badge_html(sentiment):
    cls = {'Positif':'pos','Negatif':'neg','Netral':'neu'}.get(sentiment,'neu')
    icon = {'Positif':'↑','Negatif':'↓','Netral':'→'}.get(sentiment,'')
    return f'<span class="badge badge-{cls}">{icon} {sentiment}</span>'

def topic_chips_html(topics_list, sentiment=None):
    cls = ''
    if sentiment == 'Positif': cls = ' topic-chip-pos'
    elif sentiment == 'Negatif': cls = ' topic-chip-neg'
    elif sentiment == 'Netral': cls = ' topic-chip-neu'
    chips = ''.join(f'<span class="topic-chip{cls}">{t} <b>{c}</b></span>'
                    for t, c in topics_list[:8])
    return chips

def get_breakdown(item):
    """Return dict keyed by sentiment from sentiment_breakdown"""
    result = {'Positif': None, 'Negatif': None, 'Netral': None}
    for bd in item.get('sentiment_breakdown', []):
        s = bd.get('sentiment', '')
        result[s] = bd
    return result


# ─────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_json(uploaded_file):
    content = uploaded_file.read()
    return json.loads(content)

@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)


@st.cache_data
def load_json_path(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_csv_path(path):
    return pd.read_csv(path)


# ─────────────────────────────────────────────
# SIDEBAR — UPLOAD + NAVIGASI
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Sumber Data")

    # Auto-load from workspace files in the app folder
    base = Path(__file__).parent
    p_summary_rec = base / "1final_summary_with_recommend.json"
    p_summary     = base / "1final_summary.json"
    p_agg_loc     = base / "1final_aggregate_loc.json"
    p_sentiment   = base / "1final_sentiment.json"

    summary_rec_data = load_json_path(p_summary_rec) if p_summary_rec.exists() else None
    summary_data     = load_json_path(p_summary) if p_summary.exists() else None
    agg_loc_data     = load_json_path(p_agg_loc) if p_agg_loc.exists() else None
    sentiment_data   = load_json_path(p_sentiment) if p_sentiment.exists() else None

    # Pilih sumber summary terbaik
    main_summary = summary_rec_data or summary_data

    # Navigasi sektor
    selected_sector = None
    if main_summary:
        sectors = [item['sector'] for item in main_summary if item['sector'] != 'ALL SECTOR']
        all_sectors = ['ALL SECTOR'] + sectors

        st.markdown("### 🗂️ Sektor")
        if 'selected_sector' not in st.session_state:
            st.session_state.selected_sector = 'ALL SECTOR'

        for sec in all_sectors:
            # Cari data untuk bar
            item_data = next((i for i in main_summary if i['sector'] == sec), None)
            pos = neg = neu = 0
            if item_data:
                bd = get_breakdown(item_data)
                pos = bd['Positif']['jumlah_ulasan'] if bd['Positif'] else 0
                neg = bd['Negatif']['jumlah_ulasan'] if bd['Negatif'] else 0
                neu = bd['Netral']['jumlah_ulasan'] if bd['Netral'] else 0

            active_cls = "active" if st.session_state.selected_sector == sec else ""
            bar = sentiment_bar_html(pos, neg, neu) if (pos+neg+neu) > 0 else ""
            label = "🌐 " + sec if sec == "ALL SECTOR" else sec
            st.markdown(f"""
            <div class="sector-btn {active_cls}" id="btn_{sec}">
                <div style="font-size:0.83rem;font-weight:500">{label}</div>
                {bar}
                <div style="font-size:0.7rem;color:#556677;margin-top:2px">{pos+neg+neu} ulasan</div>
            </div>""", unsafe_allow_html=True)

            if st.button(f"Pilih", key=f"nav_{sec}", help=sec,
                         use_container_width=True, type="secondary"):
                st.session_state.selected_sector = sec
                st.rerun()

        selected_sector = st.session_state.selected_sector


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if not main_summary and not agg_loc_data and not sentiment_data:
    # ── EMPTY STATE ──
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                min-height:60vh;text-align:center;gap:1rem;">
        <div style="font-size:3.5rem">📊</div>
        <h2 style="color:#E8EDF2;margin:0">Dashboard Analisis Sentimen</h2>
        <p style="color:#556677;max-width:420px;line-height:1.7">
            Tempatkan file hasil notebook di folder aplikasi (mis. 1final_*.json) atau
            upload lewat sidebar kiri untuk mulai melihat distribusi sentimen, topik,
            peta lokasi, dan rekomendasi AI per sektor.
        </p>
        <div style="background:#1E2D3D;border-radius:10px;padding:1rem 2rem;text-align:left;margin-top:0.5rem">
            <div style="color:#8899AA;font-size:0.8rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem">File yang dibutuhkan</div>
            <div style="font-size:0.85rem;line-height:2;color:#B0C4D8">
                📄 <code>1final_summary_with_recommend.json</code><br>
                📄 <code>1final_summary.json</code><br>
                📄 <code>1final_aggregate_loc.json</code><br>
                📄 <code>1final_sentiment.json</code>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Ambil data sektor yang dipilih ───
if main_summary:
    current_item = next((i for i in main_summary if i['sector'] == selected_sector), None)
else:
    current_item = None


# ════════════════════════════════════════
# SECTION 1 — HEADER SEKTOR
# ════════════════════════════════════════
if current_item:
    bd = get_breakdown(current_item)
    total = current_item.get('total_ulasan', 0)
    pos   = bd['Positif']['jumlah_ulasan'] if bd['Positif'] else 0
    neg   = bd['Negatif']['jumlah_ulasan'] if bd['Negatif'] else 0
    neu   = bd['Netral']['jumlah_ulasan']  if bd['Netral']  else 0
    pct_pos = pos/total*100 if total else 0
    pct_neg = neg/total*100 if total else 0
    pct_neu = neu/total*100 if total else 0
    health = "🟢 Baik" if pct_pos > 60 else "🟡 Perlu Perhatian" if pct_pos > 40 else "🔴 Kritis"

    # ── Page title
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.3rem">
        <h1 style="margin:0;font-size:1.7rem;font-weight:700;color:#E8EDF2">{selected_sector}</h1>
        <span style="background:#1E2D3D;border-radius:6px;padding:3px 10px;font-size:0.78rem;color:#8899AA">{health}</span>
    </div>
    <div style="color:#556677;font-size:0.83rem;margin-bottom:1.5rem">{total} total ulasan dianalisis</div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="card">
            <div class="kpi-label">Total Ulasan</div>
            <div class="kpi-value" style="color:#E8EDF2">{total:,}</div>
            {sentiment_bar_html(pos, neg, neu)}
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card">
            <div class="kpi-label">Positif</div>
            <div class="kpi-value pos">{pos:,}</div>
            <div class="kpi-sub">{pct_pos:.1f}% dari total</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card">
            <div class="kpi-label">Negatif</div>
            <div class="kpi-value neg">{neg:,}</div>
            <div class="kpi-sub">{pct_neg:.1f}% dari total</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="card">
            <div class="kpi-label">Netral</div>
            <div class="kpi-value neu">{neu:,}</div>
            <div class="kpi-sub">{pct_neu:.1f}% dari total</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # SECTION 2 — TOPIK + REKOMENDASI
    # ════════════════════════════════════════
    col_left, col_right = st.columns([1.1, 1], gap="medium")

    with col_left:
        st.markdown('<div class="sec-header">Topik Utama per Sentimen</div>', unsafe_allow_html=True)

        for sentiment in ['Positif', 'Negatif', 'Netral']:
            bd_item = bd[sentiment]
            if not bd_item:
                continue

            jml = bd_item.get('jumlah_ulasan', 0)
            top_topics_str = bd_item.get('top_topics') or bd_item.get('topik_dan_jumlah') or ''
            topic_counts_str = bd_item.get('topic_counts', '')

            # Parse topics
            topics_list = []
            if top_topics_str and topic_counts_str:
                ts = [t.strip() for t in str(top_topics_str).split(',') if t.strip()]
                cs = [c.strip() for c in str(topic_counts_str).split(',') if c.strip()]
                topics_list = list(zip(ts, cs))[:8]
            elif top_topics_str:
                topics_list = parse_topics(top_topics_str)

            chips = topic_chips_html(topics_list, sentiment)
            color = {'Positif':'#00D4AA','Negatif':'#FF5C5C','Netral':'#FFB347'}[sentiment]

            st.markdown(f"""
            <div class="card-sm" style="margin-bottom:0.6rem;border-left:3px solid {color}33">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">
                    {badge_html(sentiment)}
                    <span style="font-size:0.75rem;color:#556677">{jml:,} ulasan</span>
                </div>
                <div>{chips if chips else '<span style="color:#556677;font-size:0.8rem">— tidak ada topik tersedia —</span>'}</div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sec-header">Rekomendasi AI</div>', unsafe_allow_html=True)

        # Coba ambil rekomendasi: bisa per-sektor atau per-sentimen
        rec_per_sector = current_item.get('rekomendasi_ai') or current_item.get('rekomendasi_komprehensif')

        if rec_per_sector:
            # Satu rekomendasi komprehensif
            for chunk in rec_per_sector.split('\n\n---\n\n'):
                if chunk.strip():
                    st.markdown(f'<div class="rec-box">{chunk.strip()}</div>', unsafe_allow_html=True)
        else:
            # Cek per-sentimen di breakdown
            has_rec = any(bd[s] and bd[s].get('rekomendasi_ai') for s in ['Positif','Negatif','Netral'])
            if has_rec:
                for sentiment in ['Positif','Negatif','Netral']:
                    bd_item = bd[sentiment]
                    if bd_item and bd_item.get('rekomendasi_ai'):
                        color = {'Positif':'#00D4AA','Negatif':'#FF5C5C','Netral':'#FFB347'}[sentiment]
                        st.markdown(f"""
                        <div style="margin-bottom:0.5rem">
                            {badge_html(sentiment)}
                        </div>
                        <div class="rec-box" style="border-left-color:{color};margin-bottom:0.8rem">
                            {bd_item['rekomendasi_ai']}
                        </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="rec-box" style="border-left-color:#243447;color:#556677">
                    Rekomendasi AI belum tersedia untuk sektor ini.<br>
                    Upload file <code>1final_summary_with_recommend.json</code> untuk melihat rekomendasi.
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════
# SECTION 3 — PETA LOKASI
# ════════════════════════════════════════
if agg_loc_data:
    st.markdown('<div class="sec-header">Peta Sebaran Lokasi</div>', unsafe_allow_html=True)

    df_loc = pd.DataFrame(agg_loc_data)

    # Normalize column names (pivot_table bisa buat MultiIndex)
    df_loc.columns = [str(c) for c in df_loc.columns]

    # Filter sektor jika bukan ALL
    if selected_sector != 'ALL SECTOR' and 'sector' in df_loc.columns:
        df_loc_filtered = df_loc[df_loc['sector'] == selected_sector]
    else:
        df_loc_filtered = df_loc

    # Cek koordinat
    has_coords = False
    if sentiment_data:
        df_sent = pd.DataFrame(sentiment_data)
        if 'place_lat' in df_sent.columns and 'place_lng' in df_sent.columns and 'place_name' in df_sent.columns:
            coords = df_sent.groupby('place_name')[['place_lat','place_lng']].first().reset_index()
            df_loc_filtered = df_loc_filtered.merge(coords, on='place_name', how='left')
            has_coords = True

    if has_coords or ('place_lat' in df_loc_filtered.columns):
        df_map = df_loc_filtered.dropna(subset=['place_lat','place_lng']).copy()
        df_map = df_map.rename(columns={'place_lat':'lat','place_lng':'lon'})

        if not df_map.empty:
            col_map, col_loc_table = st.columns([1.6, 1], gap="medium")
            with col_map:
                # Prepare tooltip html showing counts per sentiment
                for c in ['Positif','Negatif','Netral','total_ulasan']:
                    if c not in df_map.columns:
                        df_map[c] = 0

                # radius based on total ulasan (clamped)
                df_map['radius'] = (df_map['total_ulasan'].fillna(1).astype(float).clip(lower=1))

                # assign color per sector
                if 'sector' not in df_map.columns:
                    df_map['sector'] = 'ALL'

                sectors = list(df_map['sector'].fillna('ALL').astype(str).unique())
                palette = [
                    [0,212,170],   # teal
                    [255,92,92],   # red
                    [255,179,71],  # orange
                    [99,102,241],  # purple
                    [120,200,255], # light blue
                    [180,120,240], # violet
                    [255,200,100], # yellow
                    [100,200,150], # green
                ]
                mapping = {s: palette[i % len(palette)] for i, s in enumerate(sectors)}
                df_map['color'] = df_map['sector'].fillna('ALL').map(mapping).apply(lambda c: c + [200])

                initial_view = {
                    'latitude': float(df_map['lat'].mean()),
                    'longitude': float(df_map['lon'].mean()),
                    'zoom': 11,
                    'pitch': 0,
                }

                tooltip = {
                    'html': '<b>{place_name}</b><br/>Sector: {sector}<br/>Positif: {Positif} · Negatif: {Negatif} · Netral: {Netral}<br/>Total: {total_ulasan}',
                    'style': {'backgroundColor': '#0F1923', 'color': '#E8EDF2'}
                }

                layer = pdk.Layer(
                    'ScatterplotLayer',
                    data=df_map,
                    get_position='[lon, lat]',
                    get_fill_color='color',
                    get_radius=100,
                    pickable=True,
                    radius_scale=1,
                )

                deck = pdk.Deck(layers=[layer], initial_view_state=initial_view, tooltip=tooltip)
                st.pydeck_chart(deck, use_container_width=True)
            with col_loc_table:
                st.markdown('<div class="sec-header">Top Lokasi</div>', unsafe_allow_html=True)
                # Tampilkan top lokasi berdasarkan total ulasan
                sort_col = 'total_ulasan' if 'total_ulasan' in df_loc_filtered.columns else df_loc_filtered.columns[-1]
                df_top = df_loc_filtered.nlargest(10, sort_col) if sort_col in df_loc_filtered.columns else df_loc_filtered.head(10)
                show_cols = ['place_name'] + [c for c in ['Positif','Negatif','Netral','total_ulasan'] if c in df_top.columns]
                st.dataframe(df_top[show_cols].reset_index(drop=True),
                             use_container_width=True, height=280,
                             column_config={
                                 'place_name': 'Lokasi',
                                 'Positif': st.column_config.NumberColumn('👍', format="%d"),
                                 'Negatif': st.column_config.NumberColumn('👎', format="%d"),
                                 'Netral':  st.column_config.NumberColumn('➡️', format="%d"),
                                 'total_ulasan': st.column_config.NumberColumn('Total', format="%d"),
                             })
    else:
        # No coords — just show table
        st.dataframe(df_loc_filtered.head(20), use_container_width=True)


# ════════════════════════════════════════
# SECTION 4 — PERBANDINGAN SEMUA SEKTOR
# ════════════════════════════════════════
if main_summary and selected_sector == 'ALL SECTOR':
    st.markdown('<div class="sec-header">Perbandingan Semua Sektor</div>', unsafe_allow_html=True)

    rows = []
    for item in main_summary:
        if item['sector'] == 'ALL SECTOR':
            continue
        bd = get_breakdown(item)
        pos = bd['Positif']['jumlah_ulasan'] if bd['Positif'] else 0
        neg = bd['Negatif']['jumlah_ulasan'] if bd['Negatif'] else 0
        neu = bd['Netral']['jumlah_ulasan']  if bd['Netral']  else 0
        total = item.get('total_ulasan', pos+neg+neu)
        pct_pos = pos/total*100 if total else 0
        rows.append({'Sektor': item['sector'], 'Total': total,
                     'Positif': pos, 'Negatif': neg, 'Netral': neu,
                     '% Positif': round(pct_pos, 1)})

    if rows:
        df_compare = pd.DataFrame(rows).sort_values('% Positif', ascending=False)

        # Visual bar per sektor
        for _, row in df_compare.iterrows():
            bar = sentiment_bar_html(row['Positif'], row['Negatif'], row['Netral'])
            pct = row['% Positif']
            color = '#00D4AA' if pct > 60 else '#FFB347' if pct > 40 else '#FF5C5C'
            st.markdown(f"""
            <div class="card-sm" style="margin-bottom:0.5rem">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:500;font-size:0.88rem">{row['Sektor']}</span>
                    <span style="font-size:0.78rem;color:{color};font-weight:600">{pct}% positif
                        &nbsp;·&nbsp;<span style="color:#556677">{row['Total']} ulasan</span>
                    </span>
                </div>
                {bar}
                <div style="display:flex;gap:1.2rem;margin-top:4px;font-size:0.75rem">
                    <span class="pos">↑ {row['Positif']}</span>
                    <span class="neg">↓ {row['Negatif']}</span>
                    <span class="neu">→ {row['Netral']}</span>
                </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════
# SECTION 5 — TABEL DETAIL ULASAN
# ════════════════════════════════════════
if sentiment_data:
    st.markdown('<div class="sec-header">Detail Ulasan</div>', unsafe_allow_html=True)

    df_sent = pd.DataFrame(sentiment_data)
    sentiment_col = 'indobert_sentiment' if 'indobert_sentiment' in df_sent.columns else \
                    'sentiment' if 'sentiment' in df_sent.columns else None

    # Filter sektor
    if selected_sector != 'ALL SECTOR' and 'sector' in df_sent.columns:
        df_view = df_sent[df_sent['sector'] == selected_sector].copy()
    else:
        df_view = df_sent.copy()

    if sentiment_col and df_view is not None:
        # Filter controls
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        with fc1:
            filter_sent = st.selectbox("Sentimen",
                ['Semua', 'Positif', 'Negatif', 'Netral'], key="filter_sent")
        with fc2:
            sort_by = st.selectbox("Urutkan",
                ['Default'] + (['Rating ↓'] if 'rating' in df_view.columns else []) +
                (['Skor Keyakinan ↓'] if 'score' in df_view.columns or 'indobert_score' in df_view.columns else []),
                key="sort_by")
        with fc3:
            search_kw = st.text_input("🔍 Cari kata kunci dalam ulasan",
                                       placeholder="contoh: parkir, toilet, antrean...",
                                       key="search_kw")

        if filter_sent != 'Semua':
            df_view = df_view[df_view[sentiment_col] == filter_sent]
        if search_kw:
            text_col = 'text' if 'text' in df_view.columns else None
            df_view = df_view[df_view[text_col].str.contains(search_kw, case=False, na=False)]
        if sort_by == 'Rating ↓' and 'rating' in df_view.columns:
            df_view = df_view.sort_values('rating', ascending=False)
        score_col = 'indobert_score' if 'indobert_score' in df_view.columns else \
                    'score' if 'score' in df_view.columns else None
        if sort_by == 'Skor Keyakinan ↓' and score_col:
            df_view = df_view.sort_values(score_col, ascending=False)

        st.caption(f"Menampilkan {len(df_view):,} ulasan")

        # Show columns
        show_cols = []
        for c in ['place_name', 'sector', 'text', sentiment_col,
                   'rating', score_col, 'topic']:
            if c and c in df_view.columns:
                show_cols.append(c)

        col_cfg = {}
        if sentiment_col in show_cols:
            col_cfg[sentiment_col] = st.column_config.TextColumn("Sentimen")
        if 'rating' in show_cols:
            col_cfg['rating'] = st.column_config.NumberColumn("⭐ Rating", format="%.1f")
        if score_col and score_col in show_cols:
            col_cfg[score_col] = st.column_config.ProgressColumn(
                "Keyakinan", min_value=0, max_value=1, format="%.2f")
        if 'topic' in show_cols:
            col_cfg['topic'] = st.column_config.TextColumn("Topik")
        if 'place_name' in show_cols:
            col_cfg['place_name'] = st.column_config.TextColumn("Lokasi")
        if 'text' in show_cols:
            col_cfg['text'] = st.column_config.TextColumn("Ulasan", width="large")

        st.dataframe(
            df_view[show_cols].reset_index(drop=True),
            use_container_width=True,
            height=420,
            column_config=col_cfg
        )

        # Download
        csv_dl = df_view.to_csv(index=False).encode('utf-8')
        st.download_button(
            f"⬇️ Export {len(df_view):,} Ulasan (CSV)",
            csv_dl,
            f"ulasan_{selected_sector.replace(' ','_')}.csv",
            "text/csv"
        )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #1E2D3D;
            text-align:center;color:#334455;font-size:0.75rem">
    Dashboard Analisis Sentimen · IndoBERT + KeyBERT + Claude AI
</div>
""", unsafe_allow_html=True)
