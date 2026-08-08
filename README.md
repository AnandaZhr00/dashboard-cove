---
title: Dashboard Sentimen Cove
emoji: 📊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
---

# Dashboard Analisis Sentimen Ulasan Cove

Dashboard visualisasi hasil analisis sentimen ulasan pengguna Cove dari lima platform
(Google Play Store, Google Maps, Instagram, TikTok, dan YouTube), menggunakan model
IndoBERT hasil fine-tuning.

## Isi dashboard

- **Distribusi** — proporsi sentimen keseluruhan, per platform, dan perbandingan dengan rating
- **Tren Waktu** — pergerakan sentimen bulanan dan tahunan
- **Analisis Kata** — WordCloud dan kata terbanyak per kelas sentimen
- **Jelajah Data** — tabel interaktif dengan pencarian dan unduh CSV
- **Uji Model** — pengujian ulasan baru secara langsung (opsional)

## Berkas yang dibutuhkan

```
app.py
requirements.txt
data/hasil_akhir_prediksi.csv     <- wajib
data/ringkasan_evaluasi.json      <- opsional, untuk menampilkan metrik model
```

## Mengaktifkan tab Uji Model

Isi variabel lingkungan `MODEL_ID` dengan alamat model di Hugging Face Hub,
misalnya `namaakun/indobert-sentimen-cove`. Bila dikosongkan, dashboard tetap
berjalan penuh, hanya tab Uji Model yang nonaktif.
