const pool = require('../config/database');
const { badRequest } = require('../utils/helpers');

// kata kunci AI Mock
const MOCK_KEYWORDS = {
    'berita': ['runtuh', 'menimpa', 'tumbang', 'hujan', 'angin', 'banjir', 'longsor', 'bencana', 'kecelakaan',
        'kebakaran', 'korban', 'terjadi', 'peristiwa', 'kejadian', 'dilaporkan', 'warga melaporkan'],
    'potensi desa': ['panen', 'pisang', 'umkm', 'pertanian', 'kebun', 'produk', 'usaha', 'kemasan', 'ekonomi', 'pelatihan'],
    'pengumuman': ['jadwal', 'pelayanan', 'pembaruan', 'imbauan', 'diimbau', 'persyaratan', 'administrasi', 'pendataan', 'dokumen'],
    'kegiatan': ['gotong', 'royong', 'musyawarah', 'kerja bakti', 'karnaval', 'lomba', 'pentas', 'jalan sehat', 'kegiatan']
};

// Jika tidak ada kata kunci yang cocok.
const MOCK_FALLBACK = 'berita';
const FALLBACK_CONFIDENCE = 0.34;
// Ambang 'perlu ditinjau'. Diatur dari hasil evaluasi (lihat metrik_evaluasi.json); dapat diubah lewat .env
const REVIEW_THRESHOLD = Number(process.env.AI_REVIEW_THRESHOLD) || 0.4;

const round3 = (n) => Math.round(n * 1000) / 1000;

const mockClassify = (text, namaKategori) => {
    const t = text.toLowerCase();
    const scores = namaKategori.map((nama) => ({
        kategori: nama,
        hits: (MOCK_KEYWORDS[nama.toLowerCase()] || []).filter((w) => t.includes(w)).length
    }));
    const totalHits = scores.reduce((a, s) => a + s.hits, 0);

    if (totalHits === 0) {
        const fb = scores.find((s) => s.kategori.toLowerCase() === MOCK_FALLBACK) || scores[0];
        const kategori = fb ? fb.kategori : '';
        return { kategori, confidence: FALLBACK_CONFIDENCE, skor: kategori ? [{ kategori, confidence: FALLBACK_CONFIDENCE }] : [] };
    }

    // Confidence = porsi kategori dibanding kategori lain x kekuatan bukti (1 kata cocok = lemah)
    const skor = scores
        .filter((s) => s.hits > 0)
        .map((s) => ({ kategori: s.kategori, confidence: (s.hits / totalHits) * (s.hits / (s.hits + 2)) }))
        .sort((a, b) => b.confidence - a.confidence);
    return { kategori: skor[0].kategori, confidence: skor[0].confidence, skor };
};

// Memanggil layanan AI terpisah. Kontrak ada di README.
const callAiService = async (judul, konten) => {
    const base = process.env.AI_SERVICE_URL;
    if (!base) {
        const e = new Error('AI_SERVICE_URL belum dikonfigurasi.');
        e.status = 503;
        e.expose = true;
        throw e;
    }

    let response;
    try {
        response = await fetch(`${base.replace(/\/$/, '')}/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(process.env.AI_API_KEY ? { 'X-API-Key': process.env.AI_API_KEY } : {})
            },
            body: JSON.stringify({ judul, konten }),
            signal: AbortSignal.timeout(Number(process.env.AI_TIMEOUT_MS) || 5000)
        });
    } catch (err) {
        // timeout atau layanan tidak terjangkau
        const e = new Error('Layanan AI tidak tersedia. Silakan pilih kategori secara manual.');
        e.status = 503;
        e.expose = true;
        throw e;
    }

    if (!response.ok) {
        const e = new Error('Layanan AI mengembalikan kesalahan. Silakan pilih kategori secara manual.');
        e.status = 502;
        e.expose = true;
        throw e;
    }

    const json = await response.json().catch(() => null);
    if (!json || typeof json.kategori !== 'string' || typeof json.confidence !== 'number') {
        const e = new Error('Respons layanan AI tidak sesuai format.');
        e.status = 502;
        e.expose = true;
        throw e;
    }
    return json;
};

// Cek Klasifikasi
const klasifikasiInformasi = async (req, res, next) => {
    try {
        const { judul, konten } = req.body || {};
        if (typeof judul !== 'string' || !judul.trim() || typeof konten !== 'string' || !konten.trim()) {
            return badRequest(res, 'Judul dan konten wajib diisi untuk klasifikasi.');
        }
        if (judul.length > 150 || konten.length > 100000) {
            return badRequest(res, 'Judul atau konten terlalu panjang.');
        }

        // Daftar kategori diambil lebih dulu
        const { rows: kategori } = await pool.query('SELECT id, nama_kategori FROM kategori');

        const mock = process.env.AI_MOCK === 'true';
        const ai = mock
            ? mockClassify(`${judul} ${konten}`, kategori.map((k) => k.nama_kategori))
            : await callAiService(judul.trim(), konten);

        // Petakan nama kategori dari AI ke baris kategori di database
        const byName = new Map(kategori.map((k) => [k.nama_kategori.toLowerCase(), k]));
        const match = byName.get(ai.kategori.toLowerCase());

        if (!match) {
            return res.status(502).json({
                success: false,
                message: 'Kategori hasil AI tidak dikenali. Silakan pilih kategori secara manual.'
            });
        }

        const alternatif = Array.isArray(ai.skor)
            ? ai.skor
                .filter((s) => s && byName.has(String(s.kategori).toLowerCase()))
                .map((s) => ({
                    kategori_id: byName.get(s.kategori.toLowerCase()).id,
                    nama_kategori: byName.get(s.kategori.toLowerCase()).nama_kategori,
                    confidence: round3(Number(s.confidence))
                }))
            : [];

        res.status(200).json({
            success: true,
            message: 'Rekomendasi kategori berhasil dibuat',
            data: {
                rekomendasi: {
                    kategori_id: match.id,
                    nama_kategori: match.nama_kategori,
                    confidence: round3(ai.confidence)
                },
                // true jika AI kurang yakin
                perlu_ditinjau: ai.confidence < REVIEW_THRESHOLD,
                alternatif,
                sumber: mock ? 'mock' : 'ai'
            }
        });
    } catch (error) {
        next(error);
    }
};

module.exports = { klasifikasiInformasi };