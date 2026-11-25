"""Melatih dan mengevaluasi model klasifikasi informasi Desa Tempel.

Jalankan dari folder ai_klasifikasi:  python train_model.py
Hasil: model_klasifikasi_desa.joblib, metrik_evaluasi.json, confusion_matrix_terbaik.png,
       distribusi_kategori.png
"""
import json
import os
import re

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset_desa_tempel.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_klasifikasi_desa.joblib")
SEED = 42


def bersihkan_teks(teks) -> str:
    """Harus identik dengan fungsi di service_klasifikasi.py."""
    if not isinstance(teks, str):
        return ""
    teks = teks.lower()
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    return re.sub(r"\s+", " ", teks).strip()


def gabung_teks(df: pd.DataFrame) -> pd.Series:
    # Judul diberi bobot ganda (ditulis dua kali) karena paling informatif.
    return (df["judul"] + " " + df["judul"] + " " + df["ringkasan"] + " " + df["konten"]).apply(bersihkan_teks)


def buat_pipeline(C=1.0, ngram=(1, 1)) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
    ])


df = pd.read_csv(CSV_PATH).dropna(subset=["judul", "konten", "kategori"]).copy()
for kol in ["judul", "ringkasan", "konten", "kategori"]:
    df[kol] = df[kol].fillna("").astype(str).str.strip()
if "sumber" not in df.columns:
    df["sumber"] = "asli"

X, y = gabung_teks(df), df["kategori"]
kelas = sorted(y.unique())
print(f"Data: {len(df)} baris | kelas: {kelas}")
print(df.groupby(["kategori", "sumber"]).size().unstack(fill_value=0), "\n")

# 1) Pemilihan hyperparameter dengan cross-validation (BUKAN dengan melihat nilai confidence)
rkf = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=SEED)
grid = GridSearchCV(
    buat_pipeline(),
    {"tfidf__ngram_range": [(1, 1), (1, 2)], "clf__C": [0.5, 1, 5, 10, 25]},
    scoring="f1_macro", cv=rkf, n_jobs=1, refit=False,
)
grid.fit(X, y)
hasil = pd.DataFrame(grid.cv_results_)[["param_tfidf__ngram_range", "param_clf__C", "mean_test_score", "std_test_score"]]
print(hasil.sort_values("mean_test_score", ascending=False).round(3).to_string(index=False), "\n")
terbaik = grid.best_params_
print("Parameter terbaik:", terbaik)

# 2) Estimasi performa pada data yang tidak dipakai melatih
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pred_cv = cross_val_predict(buat_pipeline(terbaik["clf__C"], terbaik["tfidf__ngram_range"]), X, y, cv=cv)
akurasi_cv, f1_cv = accuracy_score(y, pred_cv), f1_score(y, pred_cv, average="macro")
print(f"\nCross-validation 5-fold: akurasi={akurasi_cv:.3f}, F1-macro={f1_cv:.3f}")
print(classification_report(y, pred_cv, zero_division=0))

# 3) Uji pada data hold-out 25% (sekali pakai)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
m_ho = buat_pipeline(terbaik["clf__C"], terbaik["tfidf__ngram_range"]).fit(Xtr, ytr)
pred_ho = m_ho.predict(Xte)
akurasi_ho = accuracy_score(yte, pred_ho)
print(f"Hold-out 25% ({len(yte)} data): akurasi={akurasi_ho:.3f}, F1-macro={f1_score(yte, pred_ho, average='macro'):.3f}")

# 4) Uji generalisasi: latih pada data tambahan, uji pada 32 data asli
akurasi_silang = None
if (df["sumber"] == "tambahan").any() and (df["sumber"] == "asli").any():
    m_sil = buat_pipeline(terbaik["clf__C"], terbaik["tfidf__ngram_range"])
    m_sil.fit(X[df["sumber"] == "tambahan"], y[df["sumber"] == "tambahan"])
    p_sil = m_sil.predict(X[df["sumber"] == "asli"])
    akurasi_silang = accuracy_score(y[df["sumber"] == "asli"], p_sil)
    print(f"Latih pada data tambahan, uji pada data asli: akurasi={akurasi_silang:.3f}")

# 5) Distribusi confidence pada prediksi cross-validation (dasar penentuan ambang 'perlu ditinjau')
proba_cv = cross_val_predict(buat_pipeline(terbaik["clf__C"], terbaik["tfidf__ngram_range"]), X, y, cv=cv, method="predict_proba")
conf = proba_cv.max(axis=1)
benar = pred_cv == y.values
print(f"\nConfidence rata-rata: benar={conf[benar].mean():.3f}"
      + (f", salah={conf[~benar].mean():.3f}" if (~benar).any() else ""))
for ambang in (0.4, 0.5, 0.6):
    ditandai = conf < ambang
    print(f"Ambang {ambang}: {ditandai.mean()*100:.0f}% data ditandai 'perlu ditinjau' "
          f"({(~benar & ditandai).sum()} dari {(~benar).sum()} prediksi salah tertangkap)")

# 6) Model akhir: dilatih pada seluruh data
model = buat_pipeline(terbaik["clf__C"], terbaik["tfidf__ngram_range"]).fit(X, y)
joblib.dump({
    "model": model,
    "nama_model": "TF-IDF + Logistic Regression (Desa Tempel)",
    "classes": list(model.classes_),
    "parameter": {"C": terbaik["clf__C"], "ngram_range": list(terbaik["tfidf__ngram_range"])},
    "jumlah_data_latih": int(len(df)),
}, MODEL_PATH)

# 7) Simpan metrik dan gambar
metrik = {
    "jumlah_data": int(len(df)),
    "jumlah_per_kategori": y.value_counts().to_dict(),
    "parameter_terbaik": {"C": terbaik["clf__C"], "ngram_range": list(terbaik["tfidf__ngram_range"])},
    "cv_5fold": {"akurasi": round(akurasi_cv, 4), "f1_macro": round(f1_cv, 4)},
    "holdout_25persen": {"jumlah_uji": int(len(yte)), "akurasi": round(akurasi_ho, 4)},
    "latih_tambahan_uji_asli_akurasi": None if akurasi_silang is None else round(akurasi_silang, 4),
    "confidence_rata2_benar": round(float(conf[benar].mean()), 4),
    "laporan_per_kelas": classification_report(y, pred_cv, output_dict=True, zero_division=0),
}
with open(os.path.join(BASE_DIR, "metrik_evaluasi.json"), "w", encoding="utf-8") as f:
    json.dump(metrik, f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y, pred_cv, labels=kelas), display_labels=kelas).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=20)
ax.set_title("Confusion Matrix (cross-validation 5-fold)")
fig.tight_layout(); fig.savefig(os.path.join(BASE_DIR, "confusion_matrix_terbaik.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
df.groupby(["kategori", "sumber"]).size().unstack(fill_value=0).plot(kind="bar", stacked=True, ax=ax, rot=20)
ax.set_title("Distribusi data latih per kategori"); ax.set_xlabel(""); ax.set_ylabel("Jumlah data")
fig.tight_layout(); fig.savefig(os.path.join(BASE_DIR, "distribusi_kategori.png"), dpi=150); plt.close(fig)

print("\nModel tersimpan:", MODEL_PATH)
