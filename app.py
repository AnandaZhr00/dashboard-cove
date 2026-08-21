# -*- coding: utf-8 -*-
# Dashboard Visualisasi Analisis Sentimen Ulasan Cove

import os
import base64

import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='Dashboard Sentimen Cove', layout='wide')

# --- Konfigurasi path ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'hasil_akhir_prediksi.csv')
LOGO_DIR = os.path.join(BASE_DIR, 'assets')

URUTAN = ['positif', 'netral', 'negatif']

# Palet warna sentimen
WARNA = {
    'positif': '#2E8B57',   # hijau laut, pertumbuhan dan kondisi yang baik
    'netral':  '#7D8B99',   # abu kebiruan, tidak bermuatan emosi
    'negatif': '#C0392B',   # merah bata, menuntut perhatian
}

# Berkas logo tiap platform, disimpan di folder assets/ di samping app.py
LOGO = {
    'TikTok': 'tiktok.png',
    'Instagram': 'instagram.png',
    'YouTube': 'youtube.png',
    'Google Maps': 'googlemaps.png',
    'Google Play Store': 'googleplay.png',
}


@st.cache_data
def muat_data():
    d = pd.read_csv(DATA_PATH)
    d['tanggal'] = pd.to_datetime(d['tanggal_komentar'], errors='coerce', utc=True)
    try:
        d['tanggal'] = d['tanggal'].dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    d['bulan'] = d['tanggal'].dt.to_period('M').astype(str)
    d['tahun'] = d['tanggal'].dt.year
    return d


@st.cache_data
def logo_b64(platform):
    """Baca logo platform dari folder assets/ lalu ubah menjadi base64."""
    berkas = LOGO.get(platform)
    if not berkas:
        return None
    jalur = os.path.join(LOGO_DIR, berkas)
    if not os.path.exists(jalur):
        return None
    with open(jalur, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def img_logo(platform, ukuran=22):
    """Tag gambar logo platform untuk disisipkan ke dalam HTML."""
    b64 = logo_b64(platform)
    if not b64:
        return ''
    return (f"<img src='data:image/png;base64,{b64}' width='{ukuran}' "
            f"height='{ukuran}' style='object-fit:contain;vertical-align:middle'>")


def css_logo_platform():
    """Sisipkan logo ke dalam tag pilihan platform pada multiselect."""
    aturan = []
    for nama_pf in LOGO:
        b64 = logo_b64(nama_pf)
        if not b64:
            continue
        for atribut in ('aria-label', 'title'):
            aturan.append(
                f'span[data-baseweb="tag"][{atribut}^="{nama_pf}"]::before,'
                f'li[role="option"][{atribut}^="{nama_pf}"]::before {{'
                "content:'';display:inline-block;width:20px;height:20px;"
                "margin-right:8px;vertical-align:middle;background-size:contain;"
                "background-repeat:no-repeat;background-position:center;"
                f'background-image:url("data:image/png;base64,{b64}");}}'
            )
    if aturan:
        st.markdown('<style>' + ''.join(aturan) + '</style>',
                    unsafe_allow_html=True)


def badge_platform(platform):
    """Badge kecil berisi logo + nama platform, mengikuti gaya badge terpilih."""
    logo = img_logo(platform, 18)
    return (
        "<span style='display:inline-flex;align-items:center;gap:8px;"
        "border:1px solid #d9c6f0;background:#f6effc;color:#5b3a91;"
        "border-radius:16px;padding:6px 14px;font-size:13px;font-weight:600;"
        "margin-right:8px;margin-bottom:8px'>"
        f"{logo}{platform}</span>"
    )


def tampilkan_badge_platform(platform_pilih):
    """Tampilkan badge logo platform di bawah judul dashboard.

    Bila hanya sebagian platform dipilih, tampilkan logo platform tsb saja.
    Bila semua platform (atau tidak ada filter aktif) dipilih, tampilkan
    logo seluruh platform.
    """
    if not platform_pilih or len(platform_pilih) == len(LOGO):
        daftar = list(LOGO.keys())
    else:
        # Ikuti urutan LOGO agar tampilan konsisten
        daftar = [p for p in LOGO if p in platform_pilih]

    badge = ''.join(badge_platform(p) for p in daftar)
    st.markdown(f"<div style='margin:4px 0 12px 0'>{badge}</div>",
                unsafe_allow_html=True)


def bintang(nilai):
    """Bintang rating. Kosong bila platform tidak menyediakan rating."""
    try:
        n = int(float(nilai))
    except (TypeError, ValueError):
        return ''
    n = max(0, min(5, n))
    return ("<div style='color:#e8a33d;font-size:14px;letter-spacing:1px'>"
            + '&#9733;' * n + '&#9734;' * (5 - n) + "</div>")


def kartu_ulasan(baris, warna):
    """Kartu ulasan: nama pengguna, bintang, logo platform, dan komentar."""
    nama = str(baris.get('nama_pengguna') or 'Anonim')
    platform = str(baris.get('platform') or '-')
    komentar = str(baris.get('komentar') or '').strip()
    return (
        f"<div style='border-left:4px solid {warna};background:#fafafa;"
        "border-radius:6px;padding:10px 14px;margin-bottom:10px'>"
        "<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='font-weight:600;font-size:14px'>{nama}</span>"
        "<span style='font-size:12px;color:#555;background:#eeeeee;"
        "padding:4px 12px;border-radius:12px;display:inline-flex;"
        f"align-items:center;gap:6px'>{img_logo(platform, 20)}{platform}</span>"
        "</div>"
        f"{bintang(baris.get('rating'))}"
        "<div style='font-style:italic;font-size:13px;color:#333;margin-top:6px'>"
        f"&ldquo;{komentar}&rdquo;</div></div>"
    )


df = muat_data()

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.title('Filter Data')

css_logo_platform()

daftar_pf = sorted(df['platform'].dropna().unique())
platform_pilih = st.sidebar.multiselect(
    'Platform', daftar_pf, default=daftar_pf)

sentimen_pilih = st.sidebar.multiselect(
    'Sentimen', URUTAN, default=URUTAN, format_func=str.capitalize)

tahun_tersedia = sorted([int(t) for t in df['tahun'].dropna().unique()])
if tahun_tersedia:
    rentang = st.sidebar.select_slider(
        'Rentang tahun', options=tahun_tersedia,
        value=(tahun_tersedia[0], tahun_tersedia[-1]))
else:
    rentang = None

mask = (df['platform'].isin(platform_pilih)
        & df['sentimen_prediksi'].isin(sentimen_pilih))
if rentang:
    mask &= df['tahun'].between(rentang[0], rentang[1]) | df['tahun'].isna()

dff = df[mask]

st.sidebar.markdown('---')
st.sidebar.metric('Data ditampilkan', f'{len(dff):,}')
st.sidebar.caption('Sumber: ulasan Google Play Store, Google Maps, Instagram, '
                   'TikTok, dan YouTube.')

# ------------------------------------------------------------------ HEADER
st.title('Dashboard Analisis Sentimen Ulasan Cove')
tampilkan_badge_platform(platform_pilih)
st.caption('Model: IndoBERT (indobenchmark/indobert-base-p1) hasil fine-tuning')

if len(dff) == 0:
    st.warning('Tidak ada data yang cocok dengan filter. Ubah filter di panel kiri.')
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric('Total Ulasan', f'{len(dff):,}')
for kol, s in zip([k2, k3, k4], URUTAN):
    n = int((dff['sentimen_prediksi'] == s).sum())
    kol.metric(s.capitalize(), f'{n:,}', f'{n/len(dff)*100:.1f}%')

st.markdown('---')

tab1, tab2, tab3, tab4 = st.tabs(
    ['Ulasan Pengguna', 'Distribusi', 'Tren Waktu', 'Jelajah Data'])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.markdown('#### Ulasan Pengguna')
    st.caption('Membaca ulasan konsumen satu per satu menurut kelas sentimennya.')

    f1, f2 = st.columns(2)
    kelas_u = f1.selectbox('Kelas Sentimen', URUTAN,
                           format_func=str.capitalize, key='kelas_ulasan')
    opsi_rt = ['Semua'] + [f'{r:.1f}' for r in
                           sorted(dff['rating'].dropna().unique(), reverse=True)]
    rt_u = f2.selectbox('Rating', opsi_rt, key='rt_ulasan')

    ulasan = dff[dff['sentimen_prediksi'] == kelas_u]
    total_kelas = len(ulasan)
    if rt_u != 'Semua':
        ulasan = ulasan[ulasan['rating'] == float(rt_u)]

    st.caption(f'Menampilkan {len(ulasan):,} dari {total_kelas:,} ulasan '
               f'kelas {kelas_u}. Platform mengikuti pilihan pada panel kiri.')

    if len(ulasan) == 0:
        st.info('Tidak ada ulasan yang cocok dengan pilihan di atas.')
    else:
        kartu = [kartu_ulasan(r, WARNA[kelas_u])
                 for _, r in ulasan.head(100).iterrows()]
        with st.container(height=520):
            st.markdown(''.join(kartu), unsafe_allow_html=True)
        if len(ulasan) > 100:
            st.caption('Hanya 100 ulasan pertama yang ditampilkan. '
                       'Gunakan filter untuk mempersempit hasil.')

# ------------------------------------------------------------------ TAB 2
with tab2:
    c1, c2 = st.columns([1, 1.4])

    with c1:
        vc = dff['sentimen_prediksi'].value_counts().reindex(URUTAN).fillna(0)
        fig = px.pie(values=vc.values, names=list(vc.index), hole=0.45,
                     color=list(vc.index), color_discrete_map=WARNA,
                     title='Proporsi Sentimen')
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tab_pf = pd.crosstab(dff['platform'], dff['sentimen_prediksi'])
        tab_pf = tab_pf.reindex(columns=URUTAN).fillna(0)
        fig = px.bar(tab_pf, barmode='stack', color_discrete_map=WARNA,
                     title='Sentimen per Platform',
                     labels={'value': 'Jumlah Ulasan', 'platform': ''})
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        norm = pd.crosstab(dff['platform'], dff['sentimen_prediksi'],
                           normalize='index') * 100
        norm = norm.reindex(columns=URUTAN).fillna(0).round(1)
        fig = px.bar(norm, barmode='stack', color_discrete_map=WARNA,
                     title='Proporsi Sentimen per Platform (%)',
                     labels={'value': 'Persentase', 'platform': ''})
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        if 'rating' in dff.columns and dff['rating'].notna().any():
            rt = dff.dropna(subset=['rating'])
            tab_rt = pd.crosstab(rt['rating'].astype(int), rt['sentimen_prediksi'])
            tab_rt = tab_rt.reindex(columns=URUTAN).fillna(0)
            fig = px.bar(tab_rt, barmode='group', color_discrete_map=WARNA,
                         title='Sentimen Model vs Nilai Rating',
                         labels={'value': 'Jumlah', 'rating': 'Rating'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Data rating tidak tersedia pada filter saat ini.')

# ------------------------------------------------------------------ TAB 3
with tab3:
    tren = dff.dropna(subset=['tanggal'])
    if len(tren) == 0:
        st.info('Tidak ada data dengan tanggal valid.')
    else:
        granularitas = st.radio('Granularitas', ['Bulanan', 'Tahunan'],
                                horizontal=True)
        kunci = 'bulan' if granularitas == 'Bulanan' else 'tahun'

        agg = tren.groupby([kunci, 'sentimen_prediksi']).size().reset_index(
            name='jumlah')
        fig = px.line(agg, x=kunci, y='jumlah', color='sentimen_prediksi',
                      color_discrete_map=WARNA, markers=True,
                      title=f'Tren Jumlah Ulasan per Sentimen ({granularitas})')
        st.plotly_chart(fig, use_container_width=True)

        pivot = tren.pivot_table(index=kunci, columns='sentimen_prediksi',
                                 aggfunc='size', fill_value=0)
        pivot = pivot.reindex(columns=URUTAN).fillna(0)
        proporsi = (pivot.div(pivot.sum(axis=1), axis=0) * 100).round(1)
        fig = px.area(proporsi, color_discrete_map=WARNA,
                      title=f'Proporsi Sentimen dari Waktu ke Waktu ({granularitas})',
                      labels={'value': 'Persentase'})
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ TAB 4
with tab4:
    st.caption('Mencari ulasan berdasarkan kata kunci (termasuk nama platform) '
               'dan mengunduh datanya.')
    cari = st.text_input(
        'Cari kata kunci dalam ulasan',
        '',
        help='Bisa mencari isi komentar maupun nama platform, '
             'mis. "instagram" atau "google play store".')
    tampil = dff
    if cari.strip():
        cocok_komentar = tampil['komentar'].astype(str).str.contains(
            cari, case=False, na=False)
        cocok_platform = tampil['platform'].astype(str).str.contains(
            cari, case=False, na=False)
        tampil = tampil[cocok_komentar | cocok_platform]
        st.caption(f'Ditemukan {len(tampil):,} ulasan mengandung kata "{cari}" '
                   'pada komentar atau nama platform.')

    kolom = [c for c in ['platform', 'nama_pengguna', 'tanggal_komentar',
                         'komentar', 'rating', 'sentimen_prediksi']
             if c in tampil.columns]
    st.dataframe(tampil[kolom].head(500), use_container_width=True,
                 hide_index=True)

    st.download_button('Unduh data terfilter (CSV)',
                       tampil[kolom].to_csv(index=False).encode('utf-8'),
                       'ulasan_terfilter.csv', 'text/csv')

st.markdown('---')
st.caption('Dashboard Analisis Sentimen Ulasan Cove — dibangun dengan Streamlit '
           'dan IndoBERT.')