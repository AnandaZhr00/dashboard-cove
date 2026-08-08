
# -*- coding: utf-8 -*-
# Dashboard Visualisasi Analisis Sentimen Ulasan Cove
 
import os
import json
import re
from collections import Counter
 
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
st.set_page_config(page_title='Dashboard Sentimen Cove', layout='wide')
 
# --- Konfigurasi path (berjalan di server hosting, bukan lagi di Colab) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'hasil_akhir_prediksi.csv')
EVAL_PATH = os.path.join(BASE_DIR, 'data', 'ringkasan_evaluasi.json')
 
# ID model di Hugging Face Hub — hasil push_to_hub() pada notebook (bagian 3.7.4).
# Sesuai notebook: REPO = "Anandazhr0/indobert-sentimen-cove"
# Isi lewat Secret/Environment Variable bernama MODEL_ID di server hosting.
# Kosongkan (atau hapus Secret-nya) bila tidak ingin mengaktifkan tab Uji Model.
MODEL_ID = os.environ.get('MODEL_ID', 'Anandazhr0/indobert-sentimen-cove').strip()
 
# Token HF (hanya perlu diisi jika repo model kamu di-set PRIVATE di Hugging Face Hub).
# Kosongkan bila repo modelnya public — pipeline() bisa load tanpa token.
HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()
 
URUTAN = ['positif', 'netral', 'negatif']
WARNA = {'positif': '#2a9d8f', 'netral': '#e9c46a', 'negatif': '#e76f51'}
 
 
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
def muat_evaluasi():
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH) as f:
            return json.load(f)
    return None
 
 
@st.cache_resource(show_spinner='Memuat model IndoBERT...')
def muat_model():
    if not MODEL_ID:
        return None
    try:
        from transformers import pipeline
        return pipeline('text-classification', model=MODEL_ID, tokenizer=MODEL_ID,
                        truncation=True, max_length=64,
                        token=HF_TOKEN if HF_TOKEN else None)
    except Exception as e:
        st.session_state['galat_model'] = str(e)
        return None
 
 
df = muat_data()
evaluasi = muat_evaluasi()
 
# ------------------------------------------------------------------ SIDEBAR
st.sidebar.title('Filter Data')
 
platform_pilih = st.sidebar.multiselect(
    'Platform', sorted(df['platform'].dropna().unique()),
    default=sorted(df['platform'].dropna().unique()))
 
sentimen_pilih = st.sidebar.multiselect('Sentimen', URUTAN, default=URUTAN)
 
keyakinan_min = st.sidebar.slider('Keyakinan model minimum', 0.0, 1.0, 0.0, 0.05)
 
tahun_tersedia = sorted([int(t) for t in df['tahun'].dropna().unique()])
if tahun_tersedia:
    rentang = st.sidebar.select_slider(
        'Rentang tahun', options=tahun_tersedia,
        value=(tahun_tersedia[0], tahun_tersedia[-1]))
else:
    rentang = None
 
mask = (df['platform'].isin(platform_pilih)
        & df['sentimen_prediksi'].isin(sentimen_pilih)
        & (df['keyakinan_prediksi'] >= keyakinan_min))
if rentang:
    mask &= df['tahun'].between(rentang[0], rentang[1]) | df['tahun'].isna()
 
dff = df[mask]
 
st.sidebar.markdown('---')
st.sidebar.metric('Data ditampilkan', f'{len(dff):,}')
st.sidebar.caption('Sumber: ulasan Google Play Store, Google Maps, Instagram, TikTok, dan YouTube.')
 
# ------------------------------------------------------------------ HEADER
st.title('Dashboard Analisis Sentimen Ulasan Cove')
st.caption('Model: IndoBERT (indobenchmark/indobert-base-p1) hasil fine-tuning')
 
if len(dff) == 0:
    st.warning('Tidak ada data yang cocok dengan filter. Ubah filter di panel kiri.')
    st.stop()
 
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric('Total Ulasan', f'{len(dff):,}')
for kol, s in zip([k2, k3, k4], URUTAN):
    n = int((dff['sentimen_prediksi'] == s).sum())
    kol.metric(s.capitalize(), f'{n:,}', f'{n/len(dff)*100:.1f}%')
if evaluasi:
    m = evaluasi['metrik'][evaluasi['skema_terbaik']]
    k5.metric('Akurasi Model', f"{m['accuracy']*100:.2f}%",
              f"F1-macro {m['f1_macro']*100:.2f}%")
 
st.markdown('---')
 
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Distribusi', 'Tren Waktu', 'Analisis Kata', 'Jelajah Data', 'Uji Model'])
 
# ------------------------------------------------------------------ TAB 1
with tab1:
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
        norm = pd.crosstab(dff['platform'], dff['sentimen_prediksi'], normalize='index') * 100
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
 
# ------------------------------------------------------------------ TAB 2
with tab2:
    tren = dff.dropna(subset=['tanggal'])
    if len(tren) == 0:
        st.info('Tidak ada data dengan tanggal valid.')
    else:
        granularitas = st.radio('Granularitas', ['Bulanan', 'Tahunan'], horizontal=True)
        kunci = 'bulan' if granularitas == 'Bulanan' else 'tahun'
 
        agg = tren.groupby([kunci, 'sentimen_prediksi']).size().reset_index(name='jumlah')
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
 
# ------------------------------------------------------------------ TAB 3
with tab3:
    kolom_teks = 'teks_stemming' if 'teks_stemming' in dff.columns else 'teks_normalisasi'
    pilih_s = st.selectbox('Pilih kelas sentimen', URUTAN)
    subset = dff[dff['sentimen_prediksi'] == pilih_s]
 
    if len(subset) < 3:
        st.info('Data terlalu sedikit untuk divisualisasikan.')
    else:
        teks = ' '.join(subset[kolom_teks].dropna().astype(str)).strip()
        if not teks:
            st.info('Tidak ada teks yang dapat divisualisasikan pada filter ini.')
            st.stop()
 
        c1, c2 = st.columns([1.3, 1])
 
        with c1:
            wc = WordCloud(width=900, height=430, background_color='white',
                           colormap='viridis', max_words=120).generate(teks)
            fig, ax = plt.subplots(figsize=(9, 4.3))
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            ax.set_title(f'WordCloud — Sentimen {pilih_s.capitalize()}')
            st.pyplot(fig)
 
        with c2:
            kata = Counter(teks.split())
            top = pd.DataFrame(kata.most_common(15), columns=['kata', 'frekuensi'])
            fig = px.bar(top.sort_values('frekuensi'), x='frekuensi', y='kata',
                         orientation='h', title='15 Kata Terbanyak',
                         color_discrete_sequence=[WARNA[pilih_s]])
            st.plotly_chart(fig, use_container_width=True)
 
        st.markdown('#### Contoh ulasan')
        st.dataframe(
            subset.nlargest(8, 'keyakinan_prediksi')[
                ['platform', 'komentar', 'keyakinan_prediksi']],
            use_container_width=True, hide_index=True)
 
# ------------------------------------------------------------------ TAB 4
with tab4:
    cari = st.text_input('Cari kata kunci dalam ulasan', '')
    tampil = dff
    if cari.strip():
        tampil = tampil[tampil['komentar'].astype(str).str.contains(cari, case=False, na=False)]
        st.caption(f'Ditemukan {len(tampil):,} ulasan mengandung kata "{cari}".')
 
    kolom = [c for c in ['platform', 'tanggal_komentar', 'komentar', 'rating',
                         'sentimen_prediksi', 'keyakinan_prediksi'] if c in tampil.columns]
    st.dataframe(tampil[kolom].head(500), use_container_width=True, hide_index=True)
 
    st.download_button('Unduh data terfilter (CSV)',
                       tampil[kolom].to_csv(index=False).encode('utf-8'),
                       'ulasan_terfilter.csv', 'text/csv')
 
# ------------------------------------------------------------------ TAB 5
with tab5:
    st.markdown('#### Uji model pada ulasan baru')
    model = muat_model()
 
    if model is None:
        if not MODEL_ID:
            st.info('Tab ini nonaktif karena variabel MODEL_ID belum diisi. '
                    'Isi Secret MODEL_ID dengan alamat model di Hugging Face Hub '
                    'untuk mengaktifkan uji coba ulasan baru.')
        else:
            st.warning('Model gagal dimuat: ' + st.session_state.get('galat_model', 'penyebab tidak diketahui'))
    else:
        teks_uji = st.text_area('Tulis ulasan',
                                'kamarnya bersih dan pengelolanya ramah, tapi wifi sering lambat',
                                height=110)
        if st.button('Analisis Sentimen', type='primary'):
            hasil = model(teks_uji)[0]
            label = hasil['label']
            skor = hasil['score']
            st.markdown(f"### Hasil: :{'green' if label=='positif' else ('orange' if label=='netral' else 'red')}[{label.upper()}]")
            st.progress(float(skor))
            st.caption(f'Tingkat keyakinan model: {skor*100:.2f}%')
 
    if evaluasi:
        st.markdown('---')
        st.markdown('#### Performa model pada data uji')
        tabel = pd.DataFrame(evaluasi['metrik']).T
        tabel = (tabel * 100).round(2)
        tabel.index.name = 'Skema Pembagian Data'
        st.dataframe(tabel, use_container_width=True)
        st.caption(f"Skema terbaik: {evaluasi['skema_terbaik']} | "
                   f"Model dasar: {evaluasi['model_dasar']}")
 
st.markdown('---')
st.caption('Dashboard Analisis Sentimen Ulasan Cove — dibangun dengan Streamlit dan IndoBERT.')
 



