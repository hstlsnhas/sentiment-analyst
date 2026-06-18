import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
import plotly.express as px
import base64

# ─────────────────────────────────────────────
# CONFIG & SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Sentimen Bakauheni",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inisialisasi Global State untuk Sinkronisasi Filter Sektor
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = 'ALL SECTOR'

# ─────────────────────────────────────────────
# LOAD DATA & DYNAMIC BACKGROUND INJECTION
# ─────────────────────────────────────────────
base = Path(__file__).parent
p_bg = base / "pel_bakauheni.jpg"

if p_bg.exists():
    with open(p_bg, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');

    /* Reset & Typography */
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    
    /* Global Background Image dengan overlay gelap */
    .stApp {{ 
        background-image: linear-gradient(rgba(9, 9, 11, 0.90), rgba(9, 9, 11, 0.90)), url("data:image/jpg;base64,{img_base64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        color: #e4e4e7 !important;
    }}
    
    /* Paksa container pembungkus internal Streamlit menjadi transparan */
    .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMain"] {{
        background-color: transparent !important;
    }}

    /* Sembunyikan elemen bawaan Streamlit */
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{ padding: 2rem 3rem; max-width: 100%; }}

    /* Minimalist KPI */
    .kpi-container {{ padding: 1rem 0; display: flex; flex-direction: column; }}
    .kpi-value {{ font-size: 2.5rem; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em; }}
    .kpi-label {{ font-size: 0.85rem; font-weight: 500; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }}
    .kpi-sub   {{ font-size: 0.85rem; color: #71717a; margin-top: 0.3rem; }}

    /* Komponen Warna Sentimen */
    .pos {{ color: #10b981; }} .neg {{ color: #f43f5e; }} .neu {{ color: #f59e0b; }}
    .bg-pos {{ background-color: #10b98122; color: #10b981; }}
    .bg-neg {{ background-color: #f43f5e22; color: #f43f5e; }}
    .bg-neu {{ background-color: #f59e0b22; color: #f59e0b; }}

    /* Bar Sentimen */
    .sent-bar-wrap {{ display: flex; border-radius: 99px; overflow: hidden; height: 4px; width: 100%; background: #27272a; margin-top: 8px; }}
    .sent-bar-pos  {{ background: #10b981; height: 100%; }}
    .sent-bar-neg  {{ background: #f43f5e; height: 100%; }}
    .sent-bar-neu  {{ background: #f59e0b; height: 100%; }}

    /* Tag/Chips Topik */
    .topic-chip {{ display: inline-flex; align-items: center; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.75rem; margin: 0.2rem; font-weight: 500; }}

    /* Kotak Rekomendasi AI */
    .rec-box {{ background: transparent; border-left: 2px solid #27272a; padding: 0.2rem 0 0.2rem 1.2rem; font-size: 0.9rem; line-height: 1.6; color: #d4d4d8; margin-bottom: 1.5rem; }}
    .rec-box strong {{ color: #fafafa; font-weight: 600; }}

    /* Header Bagian */
    .sec-header {{ font-size: 1.2rem; font-weight: 600; color: #fafafa; margin: 2rem 0 1rem 0; letter-spacing: -0.01em; border-left: 3px solid #6c5ce7; padding-left: 8px; }}

    /* Override Desain Input & DataFrame Streamlit */
    .stSelectbox > div > div {{ background: #18181b !important; border: 1px solid #27272a !important; color: #e4e4e7 !important; border-radius: 6px; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #27272a; border-radius: 8px; overflow: hidden; }}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp { background-color: #09090b; color: #e4e4e7; }</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def parse_topics(topic_str):
    if not topic_str or pd.isna(topic_str): return []
    items = []
    for part in str(topic_str).split(','):
        part = part.strip()
        m = re.match(r'^(.+?)\s*\((\d+)\)\s*$', part)
        if m: items.append((m.group(1).strip(), int(m.group(2))))
        elif part: items.append((part, 1))
    return items

def sentiment_bar_html(pos, neg, neu):
    total = pos + neg + neu
    if total == 0: return ""
    pp = pos/total*100; np_ = neg/total*100; nup = neu/total*100
    return f"""<div class="sent-bar-wrap">
        <div class="sent-bar-pos" style="width:{pp:.1f}%"></div>
        <div class="sent-bar-neu" style="width:{nup:.1f}%"></div>
        <div class="sent-bar-neg" style="width:{np_:.1f}%"></div>
    </div>"""

def get_breakdown(item):
    result = {'Positif': None, 'Negatif': None, 'Netral': None}
    for bd in item.get('sentiment_breakdown', []):
        s = bd.get('sentiment', '')
        if s in result: result[s] = bd
    return result

@st.cache_data
def load_json_path(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

# ─────────────────────────────────────────────
# AMBIL DATA JSON
# ─────────────────────────────────────────────
p_summary_rec = base / "1final_summary_with_recommend.json"
p_summary     = base / "1final_summary.json"
p_agg_loc     = base / "1final_aggregate_loc.json"
p_sentiment   = base / "1final_sentiment.json"

summary_rec_data = load_json_path(p_summary_rec) if p_summary_rec.exists() else None
summary_data     = load_json_path(p_summary) if p_summary.exists() else None
agg_loc_data     = load_json_path(p_agg_loc) if p_agg_loc.exists() else None
sentiment_data   = load_json_path(p_sentiment) if p_sentiment.exists() else None

main_summary = summary_rec_data or summary_data

# ════════════════════════════════════════
# DASHBOARD GLOBAL HEADER
# ════════════════════════════════════════
st.markdown("""
<div style="margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #27272a;">
    <h1 style="font-size: 2.2rem; font-weight: 700; color: #fafafa; margin-bottom: 0.6rem; letter-spacing: -0.02em;">
        Analisis Sentimen Kawasan Bakauheni
    </h1>
    <p style="color: #a1a1aa; font-size: 0.95rem; line-height: 1.6; max-width: 1000px; margin: 0;">
        Dashboard interaktif untuk memetakan opini publik (Positif, Negatif, Netral) terhadap layanan dan fasilitas di kawasan <b>Bakauheni</b> berdasarkan Zonasi Fungsional Kawasan dan Prinsip <i>Highest and Best Use (HBU)</i>.
    </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# SEKSI NAVIGASI & PERBANDINGAN SEMUA SEKTOR
# ════════════════════════════════════════
if main_summary:
    st.markdown('<div class="sec-header" style="margin-top:0;">Perbandingan Semua Sektor & Navigasi Utama</div>', unsafe_allow_html=True)
    
    sectors_list = [item['sector'] for item in main_summary if item['sector'] != 'ALL SECTOR']
    all_available_sectors = ['ALL SECTOR'] + sectors_list
    
    for sec_id in all_available_sectors:
        item = next((i for i in main_summary if i['sector'] == sec_id), None)
        if not item: continue
        
        bd = get_breakdown(item)
        pos_ = bd['Positif']['jumlah_ulasan'] if bd['Positif'] else 0
        neg_ = bd['Negatif']['jumlah_ulasan'] if bd['Negatif'] else 0
        neu_ = bd['Netral']['jumlah_ulasan']  if bd['Netral']  else 0
        tot_ = item.get('total_ulasan', pos_ + neg_ + neu_)
        pct_ = (pos_ / tot_ * 100) if tot_ else 0
        
        is_active = (st.session_state.selected_sector == sec_id)
        if is_active:
            box_style = "background: #18181b; border: 1px solid #6c5ce7; box-shadow: 0 0 10px rgba(108, 92, 231, 0.2);"
            active_badge = "<span style='background:#6c5ce7; color:#fafafa; padding:2px 6px; border-radius:4px; font-size:0.7rem; font-weight:600; margin-right:8px;'>TERPILIH (FILTER AKTIF)</span>"
        else:
            box_style = "background: #101014; border: 1px solid #27272a;"
            active_badge = ""
            
        color_bar = '#10b981' if pct_ > 60 else '#f59e0b' if pct_ > 40 else '#f43f5e'
        
        col_card, col_btn = st.columns([5, 1])
        
        with col_card:
            st.markdown(f"""
            <div style="{box_style} border-radius:8px; padding: 0.8rem 1.2rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <span style="font-weight:600; font-size:0.95rem; color:#fafafa">{active_badge}{"Semua Sektor" if sec_id == "ALL SECTOR" else sec_id}</span>
                    <span style="font-size:0.85rem; color:{color_bar}; font-weight:600">{pct_:.1f}% Positif &nbsp;<span style="color:#71717a;font-weight:400">({tot_:,} ulasan)</span></span>
                </div>
                {sentiment_bar_html(pos_, neg_, neu_)}
                <div style="display:flex; gap:1.5rem; margin-top:6px; font-size:0.75rem; color:#a1a1aa;">
                    <span>Positif: <b class="pos">{pos_:,}</b></span>
                    <span>Netral: <b class="neu">{neu_:,}</b></span>
                    <span>Negatif: <b class="neg">{neg_:,}</b></span>
                </div>
            </div>""", unsafe_allow_html=True)
            
        with col_btn:
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            if st.button("Aktifkan Filter", key=f"master_filter_{sec_id}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.selected_sector = sec_id
                st.rerun()

    st.divider()

selected_sector = st.session_state.selected_sector
current_item = next((i for i in main_summary if i['sector'] == selected_sector), None) if main_summary else None

# ════════════════════════════════════════
# SEKSI 1 — INSIGHT SEKTOR AKTIF & KPI
# ════════════════════════════════════════
if current_item:
    bd = get_breakdown(current_item)
    total = current_item.get('total_ulasan', 0)
    pos = bd['Positif']['jumlah_ulasan'] if bd['Positif'] else 0
    neg = bd['Negatif']['jumlah_ulasan'] if bd['Negatif'] else 0
    neu = bd['Netral']['jumlah_ulasan']  if bd['Netral']  else 0
    pct_pos = pos/total*100 if total else 0
    
    dot_color = "#10b981" if pct_pos > 60 else "#f59e0b" if pct_pos > 40 else "#f43f5e"

    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="display:flex;align-items:center;gap:0.8rem;">
            <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};"></div>
            <h3 style="margin:0;font-size:1.3rem;font-weight:600;color:#e4e4e7;letter-spacing:-0.01em;">
                Data Terfilter: {"Semua Sektor" if selected_sector == "ALL SECTOR" else f"Sektor {selected_sector}"}
            </h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="kpi-container"><div class="kpi-label">Positif</div><div class="kpi-value pos">{pos:,}</div>{sentiment_bar_html(pos, 0, 0) if pos else ""}<div class="kpi-sub">{pct_pos:.1f}% dari total</div></div>""", unsafe_allow_html=True)
    with c2:
        pct_neu = neu/total*100 if total else 0
        st.markdown(f"""<div class="kpi-container"><div class="kpi-label">Netral</div><div class="kpi-value neu">{neu:,}</div>{sentiment_bar_html(0, 0, neu) if neu else ""}<div class="kpi-sub">{pct_neu:.1f}% dari total</div></div>""", unsafe_allow_html=True)
    with c3:
        pct_neg = neg/total*100 if total else 0
        st.markdown(f"""<div class="kpi-container"><div class="kpi-label">Negatif</div><div class="kpi-value neg">{neg:,}</div>{sentiment_bar_html(0, neg, 0) if neg else ""}<div class="kpi-sub">{pct_neg:.1f}% dari total</div></div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # SEKSI 2 — TOPIK & REKOMENDASI AI
    # ════════════════════════════════════════
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="sec-header">Topik Dominan Sektor Ini</div>', unsafe_allow_html=True)
        for sentiment, cls_bg in [('Positif', 'bg-pos'), ('Negatif', 'bg-neg'), ('Netral', 'bg-neu')]:
            bd_item = bd[sentiment]
            if not bd_item or bd_item.get('jumlah_ulasan', 0) == 0: continue
            top_topics_str = bd_item.get('top_topics') or bd_item.get('topik_dan_jumlah') or ''
            topics_list = parse_topics(top_topics_str)
            
            if topics_list:
                chips = ''.join(f'<span class="topic-chip {cls_bg}">{t} <span style="opacity:0.6;margin-left:4px">{c}</span></span>' for t, c in topics_list[:6])
                st.markdown(f"""<div style="margin-bottom: 1.2rem;"><div style="font-size:0.8rem;font-weight:600;color:#a1a1aa;margin-bottom:0.4rem;text-transform:uppercase;">{sentiment} ({bd_item.get('jumlah_ulasan'):,})</div><div>{chips}</div></div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sec-header">Rekomendasi Strategis AI</div>', unsafe_allow_html=True)
        rec_per_sector = current_item.get('rekomendasi_ai') or current_item.get('rekomendasi_komprehensif')

        if rec_per_sector:
            for chunk in rec_per_sector.split('\n\n---\n\n'):
                if chunk.strip(): st.markdown(f'<div class="rec-box">{chunk.strip()}</div>', unsafe_allow_html=True)
        else:
            has_rec = False
            for sentiment, color in [('Positif', '#10b981'), ('Negatif', '#f43f5e'), ('Netral', '#f59e0b')]:
                bd_item = bd[sentiment]
                if bd_item and bd_item.get('rekomendasi_ai'):
                    has_rec = True
                    st.markdown(f"""<div style="font-size:0.8rem;font-weight:600;color:{color};margin-bottom:0.2rem;text-transform:uppercase;">Insight {sentiment}</div><div class="rec-box" style="border-left-color:{color}">{bd_item['rekomendasi_ai']}</div>""", unsafe_allow_html=True)
            if not has_rec: st.markdown('<div style="color:#71717a;font-size:0.9rem">Rekomendasi belum tersedia.</div>', unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════════
# SEKSI 3 — SEBARAN LOKASI & PETA INTERAKTIF
# ════════════════════════════════════════
map_filtered_place = None

if agg_loc_data:
    st.markdown('<div class="sec-header">Peta Distribusi Geografis & Ringkasan Lokasi</div>', unsafe_allow_html=True)
    df_loc = pd.DataFrame(agg_loc_data)
    df_loc.columns = [str(c) for c in df_loc.columns]

    df_loc_filtered = df_loc[df_loc['sector'] == selected_sector] if (selected_sector != 'ALL SECTOR' and 'sector' in df_loc.columns) else df_loc

    has_coords = False
    if sentiment_data:
        df_sent = pd.DataFrame(sentiment_data)
        if {'place_lat', 'place_lng', 'place_name'}.issubset(df_sent.columns):
            coords = df_sent.groupby('place_name')[['place_lat','place_lng']].first().reset_index()
            df_loc_filtered = df_loc_filtered.merge(coords, on='place_name', how='left')
            has_coords = True

    c_map, c_table = st.columns([1.5, 1], gap="large")

    with c_map:
        if has_coords or ('place_lat' in df_loc_filtered.columns):
            df_map = df_loc_filtered.dropna(subset=['place_lat','place_lng']).copy()
            df_map = df_map.rename(columns={'place_lat':'lat','place_lng':'lon'})
            
            if not df_map.empty:
                sector_color_map = {
                    'Marina District': '#00d2ff',
                    'Hilltop Resort': '#e056fd',
                    'Harbourfront & Intermoda Area': '#ff9f43',
                    'Harbourfront and Intermoda Area': '#ff9f43'
                }
                df_map['Ukuran Titik'] = 12 

                # PERBAIKAN: Mengubah argument map_style menjadi mapbox_style
                fig = px.scatter_mapbox(
                    df_map, lat="lat", lon="lon",
                    color="sector" if "sector" in df_map.columns else None,
                    size="Ukuran Titik", size_max=8,
                    hover_name="place_name",
                    hover_data={"sector": True, "total_ulasan": True, "Positif": True, "Negatif": True, "Netral": True, "lat": False, "lon": False, "Ukuran Titik": False},
                    color_discrete_map=sector_color_map,
                    zoom=13.2, mapbox_style="carto-darkmatter"
                )
                fig.update_layout(
                    margin={"r":0,"t":0,"l":0,"b":0}, showlegend=True,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(title=dict(text="Sektor", font=dict(color="#fafafa", size=11)), font=dict(color="#a1a1aa", size=10), yanchor="top", y=0.98, xanchor="left", x=0.02, bgcolor="rgba(9, 9, 11, 0.75)", bordercolor="#27272a", borderwidth=1)
                )

                map_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"plotly_map_global_{selected_sector}")

                if map_event and "selection" in map_event and "points" in map_event["selection"]:
                    points_data = map_event["selection"]["points"]
                    if points_data: map_filtered_place = points_data[0].get("hovertext")
        else:
            st.markdown("<div style='color:#71717a;text-align:center;padding:2rem;'>Data koordinat peta tidak tersedia</div>", unsafe_allow_html=True)
    
    with c_table:
        sort_col = 'total_ulasan' if 'total_ulasan' in df_loc_filtered.columns else df_loc_filtered.columns[-1]
        df_table_all = df_loc_filtered.sort_values(by=sort_col, ascending=False) if sort_col in df_loc_filtered.columns else df_loc_filtered
        
        if map_filtered_place:
            st.markdown(f"""<div style="background:#10b98115; border:1px solid #10b98144; border-radius:6px; padding:0.5rem 0.8rem; margin-bottom:10px; font-size:0.8rem; color:#10b981;">📍 Fokus Tempat: <b>{map_filtered_place}</b></div>""", unsafe_allow_html=True)
            df_table_render = df_table_all[df_table_all['place_name'] == map_filtered_place]
        else:
            st.markdown("""<div style="background:#27272a44; border:1px solid #27272a; border-radius:6px; padding:0.5rem 0.8rem; margin-bottom:10px; font-size:0.8rem; color:#a1a1aa;">💡 Klik bulatan tempat di peta untuk mengisolasi baris tabel ringkasan ini.</div>""", unsafe_allow_html=True)
            df_table_render = df_table_all

        show_cols_loc = ['place_name'] + [c for c in ['Positif','Negatif','Netral','total_ulasan'] if c in df_table_render.columns]
        st.dataframe(df_table_render[show_cols_loc].reset_index(drop=True), use_container_width=True, height=335, hide_index=True,
                     column_config={
                         'place_name': st.column_config.TextColumn('Nama Tempat', width="medium"),
                         'Positif': st.column_config.NumberColumn('Pos', format="%d"),
                         'Negatif': st.column_config.NumberColumn('Neg', format="%d"),
                         'Netral':  st.column_config.NumberColumn('Neu', format="%d"),
                         'total_ulasan': st.column_config.NumberColumn('Total', format="%d"),
                     })

st.divider()

# ════════════════════════════════════════
# SEKSI 4 — DETAIL DATA ULASAN
# ════════════════════════════════════════
if sentiment_data:
    st.markdown('<div class="sec-header">Detail Transaksi Ulasan Publik</div>', unsafe_allow_html=True)
    df_view = pd.DataFrame(sentiment_data)
    sentiment_col = next((c for c in ['indobert_sentiment', 'sentiment'] if c in df_view.columns), None)

    if sentiment_col and not df_view.empty:
        # DIUBAH: Menjadi 3 kolom karena filter 'Urutkan' (c3 sebelumnya) sudah dihapus
        c1, c2, c3 = st.columns([1.5, 1.5, 2.0])
        
        if 'sector' in df_view.columns:
            unique_sectors = ['Semua Sektor'] + list(df_view['sector'].dropna().unique())
            ui_mapped_val = 'Semua Sektor' if selected_sector == 'ALL SECTOR' else selected_sector
            default_idx = unique_sectors.index(ui_mapped_val) if ui_mapped_val in unique_sectors else 0
            
            filter_sector = c1.selectbox("Filter Sektor", unique_sectors, index=default_idx, key="table_bottom_sector_select")
            
            internal_selected = 'ALL SECTOR' if filter_sector == 'Semua Sektor' else filter_sector
            if internal_selected != selected_sector:
                st.session_state.selected_sector = internal_selected
                st.rerun()
        else:
            filter_sector = 'Semua Sektor'
            c1.selectbox("Filter Sektor", ['Semua Sektor'], disabled=True)
            
        filter_sent = c2.selectbox("Filter Sentimen", ['Semua Sentimen', 'Positif', 'Negatif', 'Netral'])
        
        search_kw = c3.text_input("Cari Kata Kunci", placeholder="Ketik kata kunci ulasan...")

        if map_filtered_place:
            df_view = df_view[df_view['place_name'] == map_filtered_place]
        elif selected_sector != 'ALL SECTOR':
            df_view = df_view[df_view['sector'] == selected_sector]

        if filter_sent != 'Semua Sentimen': 
            df_view = df_view[df_view[sentiment_col] == filter_sent]
        if search_kw:
            text_col = 'text' if 'text' in df_view.columns else None
            if text_col: df_view = df_view[df_view[text_col].str.contains(search_kw, case=False, na=False)]

        show_cols_detail = [c for c in ['sector', 'place_name', 'text', sentiment_col, 'rating', 'topic'] if c in df_view.columns]
        col_cfg_detail = {
            'sector': st.column_config.TextColumn("Sektor", width="medium"),
            'place_name': st.column_config.TextColumn("Nama Tempat", width="medium"),
            sentiment_col: st.column_config.TextColumn("Sentimen", width="small"),
            'rating': st.column_config.NumberColumn("Rating", format="%.1f"),
            'text': st.column_config.TextColumn("Ulasan", width="large")
        }

        st.markdown(f"<div style='font-size:0.8rem;color:#71717a;margin-bottom:0.5rem'>Menampilkan {len(df_view):,} baris data</div>", unsafe_allow_html=True)
        st.dataframe(df_view[show_cols_detail].reset_index(drop=True), use_container_width=True, height=400, hide_index=True, column_config=col_cfg_detail)

        csv_dl = df_view.to_csv(index=False).encode('utf-8')
        st.download_button(f"Unduh {len(df_view):,} Ulasan Terfilter (.csv)", csv_dl, f"ulasan_bakauheni_terfilter.csv", "text/csv")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1.5rem;border-top:1px solid #27272a;text-align:center;color:#71717a;font-size:0.85rem">
    2026 · @/hstlsnhas
</div>
""", unsafe_allow_html=True)