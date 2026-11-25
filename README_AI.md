# Layanan AI Klasifikasi Informasi Desa Tempel

Layanan FastAPI yang menyarankan kategori (Berita, Kegiatan, Pengumuman, Potensi Desa) untuk informasi baru. Backend Express memanggilnya lewat `POST /api/klasifikasi`; admin tetap memutuskan kategori akhir.

## Isi folder

| File | Fungsi |
|---|---|
| `dataset_desa_tempel.csv` | 72 data latih: 32 asli + 40 tambahan (kolom `sumber`) |
| `tambah_data.py` | Skrip yang menggabungkan `dataset_asli.csv` dengan data tambahan |
| `train_model.py` / `01_klasifikasi_informasi_desa.ipynb` | Pelatihan dan evaluasi (isi sama) |
| `model_klasifikasi_desa.joblib` | Model hasil pelatihan |
| `metrik_evaluasi.json`, `confusion_matrix_terbaik.png`, `distribusi_kategori.png` | Hasil evaluasi untuk laporan |
| `service_klasifikasi.py` | Layanan FastAPI (`/health`, `/classify`) |
| `test_service.py` | 11 tes layanan (pytest) |
| `Dockerfile`, `requirements.txt` | Untuk deploy |

## Menjalankan di komputer sendiri

```bash
cd ai_klasifikasi
python -m venv .venv && .venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
python -m pytest test_service.py -q                     # 11 tes harus lulus
python -m uvicorn service_klasifikasi:app --port 8000
```

Cek: http://localhost:8000/health dan dokumentasi interaktif http://localhost:8000/docs

Untuk melatih ulang setelah data diubah: `python train_model.py` (versi scikit-learn harus sama saat melatih dan menjalankan layanan; `requirements.txt` mengunci `1.9.1`).

## Menyambungkan ke website

1. Ganti file `src/controllers/klasifikasi.controller.js` di proyek website dengan versi pada folder `patch/` (perubahan hanya dua: header `X-API-Key` opsional dan ambang `AI_REVIEW_THRESHOLD`). Frontend React dan route **tidak perlu diubah**.
2. Isi `.env` backend:

```
AI_MOCK=false
AI_SERVICE_URL=http://localhost:8000
AI_TIMEOUT_MS=5000
AI_REVIEW_THRESHOLD=0.4
# opsional, wajib sama di kedua sisi bila dipakai
AI_API_KEY=isi-dengan-string-acak-panjang
```

3. Jalankan layanan AI, lalu backend (`npm run dev`). Buka Admin, Tambah informasi, isi judul dan konten, klik **Minta rekomendasi**.

Bila layanan AI mati, backend mengembalikan 503 dan admin cukup memilih kategori manual.

## Deploy (Render)

Buat **Web Service kedua** dari folder `ai_klasifikasi` (Runtime Docker, instance Free). Salin URL-nya ke `AI_SERVICE_URL` di web service utama dan setel `AI_MOCK=false`. Pada paket gratis layanan tidur setelah sekitar 15 menit sepi, sehingga permintaan pertama bisa lambat sekitar satu menit. Naikkan `AI_TIMEOUT_MS` (misalnya `60000`) atau akses `/health` sebelum demo.

## Hasil evaluasi (untuk laporan)

| Pengujian | Hasil |
|---|---|
| Cross-validation 5-fold (72 data) | akurasi 93,1%, F1-macro 0,931 |
| Hold-out 25% (18 data) | akurasi 94,4% |
| Latih pada data tambahan, uji pada 32 data asli | akurasi 87,5% |

Parameter dipilih lewat grid search dengan cross-validation berulang: `C=10`, unigram + bigram. Dengan ambang `perlu_ditinjau` 0,4, sekitar 14% prediksi ditandai untuk diperiksa admin, dan dalam validasi silang seluruh 5 prediksi yang salah tertangkap oleh ambang tersebut.

**Keterbatasan yang sebaiknya dicantumkan:**
- Data hanya 72 baris, dan 40 di antaranya ditulis sebagai data ilustratif (kolom `sumber = tambahan`). Angka di atas belum tentu berlaku pada berita nyata. Angka 87,5% pada uji ketiga adalah perkiraan yang lebih realistis.
- Kategori "Berita" dan "Kegiatan" paling sering tertukar karena isinya tumpang tindih.
- Confidence adalah probabilitas model, bukan jaminan kebenaran. Itu sebabnya admin tetap memutuskan.
- Cara memperbaiki: tambahkan berita nyata Desa Tempel ke CSV, lalu jalankan `python train_model.py`.
