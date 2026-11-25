"""Uji layanan: pytest test_service.py"""
import pytest
from fastapi.testclient import TestClient

import service_klasifikasi as svc

client = TestClient(svc.app)
KATEGORI = {"Berita", "Kegiatan", "Pengumuman", "Potensi Desa"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert set(r.json()["classes"]) == KATEGORI


@pytest.mark.parametrize("judul,konten,harapan", [
    ("Kebakaran Rumah Warga di RT 04", "Api melahap sebuah rumah dan dipadamkan unit pemadam kebakaran, tidak ada korban jiwa.", "Berita"),
    ("Kerja Bakti Membersihkan Makam Desa", "Warga bergotong royong membersihkan area makam sejak pagi hari.", "Kegiatan"),
    ("Pemberitahuan Jadwal Pelayanan Surat Keterangan", "Warga diimbau menyiapkan persyaratan dokumen administrasi sebelum datang ke kantor desa.", "Pengumuman"),
    ("Panen Melon dan Pengembangan Usaha Tani", "Kelompok tani memanen melon dan memasarkan produk ke pasar sehingga menambah penghasilan keluarga.", "Potensi Desa"),
])
def test_klasifikasi_contoh(judul, konten, harapan):
    r = client.post("/classify", json={"judul": judul, "konten": konten})
    assert r.status_code == 200
    d = r.json()
    assert d["kategori"] == harapan
    assert 0 <= d["confidence"] <= 1
    assert d["skor"][0]["kategori"] == d["kategori"]
    assert abs(sum(s["confidence"] for s in d["skor"]) - 1) < 0.01


def test_html_dan_angka_tidak_merusak():
    r = client.post("/classify", json={"judul": "Pengumuman 2026", "konten": "<p>Jadwal <b>pelayanan</b> 12/5/2026 di balai desa, harap membawa dokumen.</p>"})
    assert r.status_code == 200


@pytest.mark.parametrize("body", [{"judul": "", "konten": "x"}, {"judul": "x", "konten": "   "}])
def test_input_kosong_ditolak(body):
    assert client.post("/classify", json=body).status_code == 400


def test_teks_tanpa_huruf_ditolak():
    assert client.post("/classify", json={"judul": "123", "konten": "456 789"}).status_code == 422


def test_judul_terlalu_panjang_ditolak():
    assert client.post("/classify", json={"judul": "a" * 151, "konten": "isi"}).status_code == 422


def test_api_key(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "rahasia")
    body = {"judul": "Pengumuman", "konten": "Jadwal pelayanan administrasi desa"}
    assert client.post("/classify", json=body).status_code == 401
    assert client.post("/classify", json=body, headers={"X-API-Key": "salah"}).status_code == 401
    assert client.post("/classify", json=body, headers={"X-API-Key": "rahasia"}).status_code == 200
