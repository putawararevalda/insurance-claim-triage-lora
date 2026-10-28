# Rencana Proyek: Fine-Tuning LoRA untuk Insurance Claim & FNOL Triage

Dokumen rencana kerja bertahap (*phased roadmap*) untuk proyek portofolio **LoRA Fine-Tuning pada Dataset Domain Asuransi (First Notice of Loss / FNOL Triage)**, disesuaikan dengan lingkungan hardware laptop lokal (GPU NVIDIA RTX 3050 6GB VRAM) dan standar hiring manager AI Engineer.

---

## Ringkasan Proyek

| Komponen | Spesifikasi |
|---|---|
| **Domain** | Asuransi Jiwa & Kesehatan (Life & Health Insurance) di Indonesia |
| **Task Utama** | Klasifikasi narasi klaim nasabah & ekstraksi parameter urgensi ke format JSON terstruktur |
| **Base Model** | `Llama-3.2-3B-Instruct` atau `Qwen2.5-1.5B-Instruct` |
| **Metode Fine-Tuning** | 4-bit QLoRA (NF4) via Unsloth / Hugging Face PEFT + TRL |
| **Hardware Target** | NVIDIA GeForce RTX 3050 Laptop (6 GB VRAM, CUDA 12.x) |
| **Output Deliverable** | Pipeline training reproduktif, adapter LoRA, benchmark evaluasi kuantitatif, dan interactive demo Streamlit |

---

## Roadmap Pelaksanaan (3 Fase)

```mermaid
flowchart TD
    subgraph Fase 1: MVP [Fase 1: MVP - End-to-End Slice]
        F1_1[Setup Environment uv & PyTorch CUDA] --> F1_2[Kurasi Seed Dataset: 150 sampel]
        F1_2 --> F1_3[Script QLoRA Dasar: train_mvp.py]
        F1_3 --> F1_4[Validasi Awal: Base vs LoRA]
    end

    subgraph Fase 2: Alpha [Fase 2: Alpha - Skala & Rigor Evaluasi]
        F2_1[Ekspansi Dataset: 500-800 sampel] --> F2_2[Tracking W&B & Hyperparameter Sweep]
        F2_2 --> F2_3[Harness Evaluasi: F1, JSON Validity]
        F2_3 --> F2_4[Uji Catastrophic Forgetting]
    end

    subgraph Fase 3: Beta [Fase 3: Beta - Serving & Portofolio Packaging]
        F3_1[Export Adapter / GGUF & Serving Ollama/FastAPI] --> F3_2[Interactive UI Demo: Streamlit]
        F3_2 --> F3_3[README Industri & Video Walkthrough]
    end

    Fase 1 --> Fase 2 --> Fase 3
```

---

## Rincian Per Fase

### Fase 1: MVP (Minimal End-to-End Slice)
**Target Waktu**: 3–4 Hari  
**Tujuan**: Memastikan seluruh alur (data $\to$ training $\to$ inferensi) berjalan sukses di GPU RTX 3050 lokal tanpa kendala Out-Of-Memory (OOM).

1. **Inisialisasi Environment**:
   - Buat virtual environment menggunakan `uv` di direktori `E:/portfolios/lorafinetuning`.
   - Install dependensi: `torch`, `unsloth` (atau `peft` + `bitsandbytes` + `trl`), `transformers`, `datasets`, `pydantic`.
   - Konfigurasi CUDA 12.x dan verifikasi alokasi memory GPU.
2. **Kurasi Seed Dataset (150–200 Pasang Data)**:
   - Buat dataset instruksi dalam Bahasa Indonesia untuk 5 kelas klaim:
     - `Inpatient_Medical` (Rawat inap demam berdarah, tifus, operasi usus buntu)
     - `Outpatient_Emergency` (Jahit luka, rontgen kecelakaan ringan, IGD)
     - `Critical_Illness` (Kanker, serangan jantung, stroke, gagal ginjal)
     - `Personal_Accident` (Fraktur tulang, kecelakaan kerja/lalin)
     - `Inquiry_or_Dispute` (Tanya status klaim lama, batas plafon, komplain)
   - Format: ChatML / Alpaca (`instruction`, `input`, `output` JSON terstruktur).
   - Pembagian data: Train (80%), Val (10%), Test (10%).
3. **Script Training Dasar (`train_mvp.py`)**:
   - Konfigurasi QLoRA 4-bit (NF4).
   - Parameter LoRA: Rank $r=16$, Alpha $\alpha=32$, dropout $0.05$.
   - Target modul: Attention projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
   - Sequence length: Maksimal 512 token (sangat hemat VRAM).
   - Training setting: 3 epoch, per-device batch size 2, gradient accumulation 4.
4. **Smoke Test & Baseline Comparison**:
   - Jalankan 10 contoh teks uji coba.
   - Amati perbandingan: Model base (apakah format berantakan/banyak halusinasi) vs Model LoRA (apakah format JSON langsung terkunci dan label tepat).

---

### Fase 2: Alpha (Skala Dataset, Rigor Eksperimen & Evaluasi Kuantitatif)
**Target Waktu**: 4–5 Hari  
**Tujuan**: Mengembangkan pipeline ke standar produksi dengan metrik matematis yang solid untuk dipresentasikan saat interview.

1. **Ekspansi Dataset (500–800 Pasang Data)**:
   - Tambahkan variasi gaya bahasa: bahasa santai WhatsApp, bahasa panik keluarga pasien, bahasa formal agen asuransi.
   - Tambahkan *edge cases*:
     - Kasus ambigu (misal nasabah jatuh karena pusing darah tinggi).
     - Kasus klaim yang berpotensi ditolak / klausul masa tunggu (*waiting period*).
2. **Experiment Tracking (Weights & Biases / MLflow)**:
   - Log seluruh metrik: training loss, validation loss per checkpoint, pemakaian VRAM, dan durasi per epoch.
   - Jalankan hyperparameter sweep ringan:
     - Rank $r \in \{8, 16, 32\}$
     - Learning rate: $1\text{e-}4$ vs $2\text{e-}4$
   - Pilih checkpoint terbaik berdasarkan Validation Loss terendah untuk mencegah *overfitting*.
3. **Automated Evaluation Harness (`eval_harness.py`)**:
   - Evaluasi head-to-head pada 50–100 data uji *unseen*:
     - **Classification Metrics**: Macro F1-score, Precision, Recall, dan Confusion Matrix per kategori.
     - **JSON Schema Adherence**: Persentase output yang 100% valid JSON dan lolos validasi Pydantic.
     - **Inference Latency & Throughput**: Tokens per second di GPU lokal.
4. **Uji Catastrophic Forgetting**:
   - Evaluasi model fine-tuned pada 20 pertanyaan umum non-asuransi (penalaran umum / instruksi dasar) untuk memastikan model tidak kehilangan kapabilitas dasar.

---

### Fase 3: Beta & Portfolio Packaging (Serving, UI Demo & Storytelling)
**Target Waktu**: 3–4 Hari  
**Tujuan**: Membungkus hasil model menjadi artefak nyata yang siap didemokan langsung ke hiring manager dan VP of Engineering.

1. **Model Packaging & Serving**:
   - Simpan adapter weights secara terpisah (< 100 MB).
   - (Opsional) Export model gabungan ke format GGUF untuk dijalankan via **Ollama** (`Modelfile`) atau sajikan via endpoint **FastAPI** (`POST /v1/triage`).
2. **Interactive Streamlit / Gradio Demo**:
   - Antarmuka web sederhana:
     - Text area untuk memasukkan narasi klaim bebas (atau memilih template contoh kasus).
     - Tampilan side-by-side: Hasil Base Model vs Hasil Fine-Tuned Model.
     - Visual badge untuk kategori klaim, tingkat urgensi, dan waktu latensi (ms).
3. **Dokumentasi & Portofolio Storytelling**:
   - Tulis `README.md` berstandar dokumentasi internal tim AI:
     - Latar belakang bisnis & arsitektur teknis.
     - Tabel perbandingan hasil evaluasi (*The Headline Numbers*).
     - Panduan instalasi dan reproduksi eksperimen dengan 1 perintah.
   - Rekam video walkthrough Loom (2–3 menit):
     - Masalah klaim di industri asuransi.
     - Demo perbandingan base vs fine-tuned di UI.
     - Bukti dashboard metrik evaluasi (Confusion Matrix & W&B).
