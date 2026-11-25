import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_klasifikasi_desa.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"File model tidak ditemukan di '{MODEL_PATH}'. "
        "Jalankan notebook 01_klasifikasi_informasi_desa.ipynb terlebih dahulu."
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
nama_model = artifact.get("nama_model", "Model Klasifikasi")


def bersihkan_teks(teks: str) -> str:
    if not isinstance(teks, str):
        return ""
    teks = teks.lower()
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


class ClassifyRequest(BaseModel):
    judul: str
    konten: str


app = FastAPI(title="Layanan Klasifikasi Informasi Desa Tempel", version="1.0.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": nama_model,
        "classes": list(getattr(model, "classes_", [])),
    }


@app.post("/classify")
def classify(req: ClassifyRequest):
    judul = (req.judul or "").strip()
    konten = (req.konten or "").strip()

    if not judul or not konten:
        raise HTTPException(status_code=400, detail="Judul dan konten wajib diisi.")

    teks_bersih = bersihkan_teks(f"{judul} {konten}")
    if not teks_bersih:
        teks_bersih = f"{judul.lower()} {konten.lower()}".strip()

    try:
        prediksi = str(model.predict([teks_bersih])[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([teks_bersih])[0]
            classes = [str(c) for c in model.classes_]
            paired = sorted(
                zip(classes, proba),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            skor = [
                {"kategori": kat, "confidence": round(float(conf), 4)}
                for kat, conf in paired
            ]
            confidence = round(float(paired[0][1]), 4)
        else:
            skor = [{"kategori": prediksi, "confidence": 0.75}]
            confidence = 0.75

        return {
            "kategori": prediksi,
            "confidence": confidence,
            "skor": skor,
            "model": nama_model,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan klasifikasi: {exc}")