# 📹 Video Walkthrough & Presentation Script (Loom)
## Indonesian Insurance Claim & FNOL Triage System via LoRA Fine-Tuning

Dokumen ini memuat panduan terstruktur dan naskah presentasi video berdurasi **2–3 menit** untuk perekaman Loom, dirancang untuk presentasi teknis enterprise bagi para decision maker dan engineering lead di industri asuransi jiwa & kesehatan (*Life & Health Insurance*).

---

## ⏱️ Video Structure & Timing Breakdown (Total: 2 Menit 45 Detik)

| Segmen | Durasi | Layar / Visual | Fokus Pembahasan |
| :--- | :---: | :--- | :--- |
| **1. Hook & Business Context** | 0:00 – 0:35 (35s) | Slide Judul / Webcam + Diagram FNOL Bottleneck | Masalah klaim asuransi tak terstruktur di Indonesia, inefisiensi biaya commercial API, dan pentingnya SLM lokal berdaulat. |
| **2. Live Interactive UI Demo** | 0:35 – 1:30 (55s) | Layar Streamlit (`app.py`) Side-by-Side | Komparasi langsung Base Model vs. LoRA Model pada kasus darurat (STEMI Jantung / DBD). Memperlihatkan kepatuhan skema JSON & latensi $<2.5$ detik. |
| **3. Architecture & QLoRA Engineering** | 1:30 – 2:15 (45s) | Diagram Arsitektur + Terminal Code | Optimasi VRAM pada RTX 3050 (6GB), 4-bit NF4, rank $r=16$, $\alpha=32$, merging ke standalone 16-bit, dan FastAPI serving. |
| **4. Headline Scorecard & Closing** | 2:15 – 2:45 (30s) | Tabel Benchmark Scorecard + GitHub Repo | Peningkatan Macro F1 dari 0.32 ke 0.98, retensi nalar umum (uji catastrophic forgetting), dan ajakan diskusi teknis. |

---

## 🎙️ Detailed Speaking Script

### Segmen 1: Hook & Business Context (0:00 – 0:35)
> **[Visual: Webcam menyala + menampilkan cover slide proyek]**  
> *"Halo semuanya, perkenalkan saya Revalda Putawara, AI Engineer. Hari ini saya mendemonstrasikan implementasi sistem AI terapan: **Automated FNOL & Insurance Claim Triage System** yang diadaptasi khusus untuk asuransi jiwa dan kesehatan di Indonesia.*
>
> *Setiap hari, perusahaan asuransi jiwa dan kesehatan di Indonesia menerima ribuan pesan klaim nasabah dalam bahasa percakapan bebas melalui WhatsApp atau mobile app. Tim operasional membutuhkan waktu lama untuk memilah urgensi, diagnosa medis, dan kelayakan penjaminan cashless. Menggunakan API komersial seperti OpenAI memakan biaya tinggi per token dan berisiko terhadap kerahasiaan data medis. Solusi yang saya bangun adalah **Small Language Model (SLM 1.5B)** yang dilatih menggunakan **4-bit QLoRA** agar dapat berjalan mandiri, cepat, dan 100% aman di infrastruktur lokal."*

---

### Segmen 2: Live Demo Interaktif Streamlit (0:35 – 1:30)
> **[Visual: Beralih ke layar browser Streamlit Web App (`http://localhost:8501`)]**  
> *"Mari kita lihat demonstrasi langsungnya pada aplikasi web interaktif yang saya bangun dengan Streamlit.*
>
> *Di sini, saya memiliki skenario kritis: seorang nasabah dilarikan ke IGD RS Harapan Kita akibat **Serangan Jantung STEMI Akut** dan butuh penjaminan kateterisasi jantung darurat.*
>
> *Ketika kita klik tombol **Jalankan Analisis Triage**, sistem mengeksekusi komparasi berdampingan (*side-by-side*) antara Base Foundation Model `Qwen2.5-1.5B` di sebelah kiri dan LoRA Golden Model kita di sebelah kanan.*
>
> *Perhatikan hasilnya:  
> Base Model menghasilkan teks yang bertele-tele dan gagal memenuhi skema JSON deterministik.  
> Sebaliknya, model **LoRA Fine-Tuned** kita secara instan dalam waktu **1.4 detik** menghasilkan ekstraksi parameter yang sempurna:  
> - Kategori terklasifikasi tepat sebagai **Critical Illness**  
> - Urgensi terdeteksi **High**  
> - Metode klaim teridentifikasi **Cashless**  
> - Diagnosa spesifik **STEMI Akut** berhasil diekstrak, lengkap dengan justifikasi operasionalnya.  
> Output ini 100% lolos validasi skema Pydantic dan siap diintegrasikan langsung ke sistem core insurance.*"

---

### Segmen 3: Technical Architecture & Hardware Optimization (1:30 – 2:15)
> **[Visual: Beralih ke diagram arsitektur di README / VS Code]**  
> *"Di balik layar, tantangan rekayasa terbesarnya adalah melatih model ini pada hardware lokal dengan kapasitas memori terbatas, yaitu laptop GPU **NVIDIA RTX 3050 dengan hanya 6GB VRAM**.*
>
> *Untuk mencapai ini tanpa mengalami CUDA Out-of-Memory, saya menerapkan strategi **QLoRA 4-bit NormalFloat (NF4)** dengan pustaka `bitsandbytes`, mengaktifkan *gradient checkpointing*, dan menginjeksi adapter rank 16 serta alpha 32 ke seluruh modul linear.*
>
> *Puncak pemakaian VRAM selama pelatihan terkendali ketat di **4.8 GB**.  
> Setelah menemukan checkpoint terbaik melalui hyperparameter sweep, bobot adapter difusikan kembali (*merge and unload*) menjadi standalone model 16-bit, yang kemudian disajikan melalui REST API performa tinggi menggunakan **FastAPI** pada endpoint `POST /v1/triage`, serta integrasi **Ollama Modelfile** untuk kemudahan deployment di edge."*

---

### Segmen 4: Headline Scorecard & Call to Action (2:15 – 2:45)
> **[Visual: Menampilkan tabel Quantitative Benchmark Scorecard & Profil GitHub]**  
> *"Secara kuantitatif, hasil pengujian pada 65 kasus uji unseen menunjukkan peningkatan performa yang sangat masif:  
> - **Macro F1-Score melonjak dari 0.3236 menjadi 0.9846**, peningkatan lebih dari 200%.  
> - **Akurasi kategori mencapai 98.5%**, dan akurasi metode klaim naik dari 26% menjadi **90.8%**.  
> - Pengujian *catastrophic forgetting* pada 25 soal non-domain membuktikan model tetap mempertahankan kemampuan penalaran umum dan logika pemrograman dengan regresi hanya 11%.*
>
> *Seluruh kode sumber, pipeline pelatihan, script evaluasi, dan endpoint API telah terdokumentasi lengkap di GitHub saya. Terima kasih banyak atas perhatian Anda, dan saya sangat terbuka untuk mendiskusikan bagaimana pendekatan ini dapat mengakselerasi otomatisasi operasional di perusahaan Anda."*

---

## 📱 Blurb LinkedIn & CV Highlight

### LinkedIn Post Copy:
```text
🚀 How I Fine-Tuned a Sovereign Small Language Model (SLM) for Indonesian Insurance FNOL Triage on a 6GB Laptop GPU! 🛡️

Handling unstructured First Notice of Loss (FNOL) claims in the Life & Health insurance industry requires speed, medical domain precision, and strict data privacy (OJK/UU PDP compliance). Commercial cloud APIs can be prohibitively expensive and pose data leakage risks.

To solve this, I developed an end-to-end domain-adapted triage pipeline:
🔹 Fine-Tuned Qwen2.5-1.5B-Instruct using 4-bit QLoRA on a consumer NVIDIA RTX 3050 Laptop GPU (6GB VRAM).
🔹 Boosted Macro F1-Score from 0.3236 (Base Zero-Shot) to 0.9846 (+0.6610) across 5 core insurance claim classes.
🔹 Enforced 100% deterministic Pydantic JSON schema extraction for instant core insurance integration.
🔹 Achieved sub-2.0s inference latency on local GPU via FastAPI serving and standalone weight merging.
🔹 Preserved general reasoning capability with an 11% controlled threshold on catastrophic forgetting tests.

Watch the 2.5-minute video walkthrough and explore the full production repository:
🔗 Loom Video: https://www.loom.com/share/lora-insurance-claim-fnol-triage
🔗 GitHub: https://github.com/putawararevalda/insurance-claim-triage-lora

#AIEngineering #MachineLearning #LoRA #QLoRA #FastAPI #Streamlit #InsurTech #GenerativeAI #PyTorch #DataScience
```

### Resume / CV Bullet Points:
```text
• Developed and deployed a sovereign Indonesian Insurance Claim & FNOL Triage SLM via 4-bit QLoRA fine-tuning on a 6GB VRAM GPU, increasing Macro F1-score from 0.3236 to 0.9846 (+204% relative gain) with 100% JSON schema validity.
• Engineered production inference microservice with FastAPI and interactive side-by-side Streamlit UI, achieving sub-2.0s inference latency and 90.8% accuracy on claim settlement method extraction.
• Implemented rigorous automated MLOps benchmark suite, including Weights & Biases experiment tracking and non-domain catastrophic forgetting retention evaluation.
```
