"""Layanan klasifikasi kategori informasi Desa Tempel (FastAPI).

Kontrak (dipanggil backend Express lewat AI_SERVICE_URL):
  POST /classify  {judul, konten}  ->  {kategori, confidence, skor[], model}
  GET  /health

Jalankan:  uvicorn service_klasifikasi:app --host 0.0.0.0 --port 8000
Variabel lingkungan (opsional):
  AI_API_KEY   bila diisi, setiap permintaan /classify wajib membawa header X-API-Key yang sama
"""
import hmac
import os
import re

import joblib
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_klasifikasi_desa.joblib")
MAX_JUDUL = 150
MAX_KONTEN = 100_000  # sama dengan batas di backend

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"File model tidak ditemukan di '{MODEL_PATH}'. Jalankan 'python train_model.py' terlebih dahulu."
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
nama_model = artifact.get("nama_model", "Model Klasifikasi")


def bersihkan_teks(teks: str) -> str:
    """Harus identik dengan fungsi yang dipakai saat pelatihan (train_model.py)."""
    if not isinstance(teks, str):
        return ""
    teks = teks.lower()
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    return re.sub(r"\s+", " ", teks).strip()


class ClassifyRequest(BaseModel):
    judul: str = Field(..., max_length=MAX_JUDUL)
    konten: str = Field(..., max_length=MAX_KONTEN)
    ringkasan: str = Field("", max_length=1000)  # opsional; backend saat ini belum mengirimnya


app = FastAPI(title="Layanan Klasifikasi Informasi Desa Tempel", version="1.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": nama_model, "classes": [str(c) for c in model.classes_]}


@app.post("/classify")
def classify(req: ClassifyRequest, x_api_key: str | None = Header(default=None)):
    kunci = os.environ.get("AI_API_KEY")
    if kunci and not hmac.compare_digest(x_api_key or "", kunci):
        raise HTTPException(status_code=401, detail="API key tidak valid.")

    judul, konten = req.judul.strip(), req.konten.strip()
    if not judul or not konten:
        raise HTTPException(status_code=400, detail="Judul dan konten wajib diisi.")

    # Format teks sama dengan saat pelatihan: judul dua kali + ringkasan + konten
    teks = bersihkan_teks(f"{judul} {judul} {req.ringkasan.strip()} {konten}")
    if not teks:
        raise HTTPException(status_code=422, detail="Teks tidak mengandung huruf yang dapat dianalisis.")

    try:
        proba = model.predict_proba([teks])[0]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Gagal melakukan klasifikasi: {exc}") from exc

    skor = sorted(
        ({"kategori": str(k), "confidence": round(float(p), 4)} for k, p in zip(model.classes_, proba)),
        key=lambda s: s["confidence"],
        reverse=True,
    )
    return {"kategori": skor[0]["kategori"], "confidence": skor[0]["confidence"], "skor": skor, "model": nama_model}
