"""Interactive Streamlit Web UI Demo for Indonesian Insurance Claim & FNOL Triage.

Showcases side-by-side comparison between Base Foundation Model (zero-shot)
and Fine-Tuned LoRA Merged Model on Indonesian health and life insurance First Notice of Loss claims.
"""

from datetime import datetime
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schema import ClaimCategory, ClaimMethod, ClaimTriageOutput, UrgencyLevel
from src.service import TriageService, clean_json_text, fallback_parse_triage_json
from src.serving.api import OFFICIAL_CATEGORIES, PRESET_SCENARIOS

# Page Configuration
st.set_page_config(
    page_title="Insurance Claim & FNOL Triage AI Demo | Qwen2.5 LoRA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Fintech / Insurtech Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .badge-inpatient { background-color: #DBEAFE; color: #1E40AF; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-outpatient { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-critical { background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-accident { background-color: #FCE7F3; color: #9D174D; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    .badge-inquiry { background-color: #E5E7EB; color: #374151; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 0.85rem; }
    
    .badge-urg-high { background-color: #EF4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; }
    .badge-urg-med { background-color: #F59E0B; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; }
    .badge-urg-low { background-color: #10B981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; }
    
    .badge-valid { background-color: #D1FAE5; color: #065F46; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; border: 1px solid #10B981; }
    .badge-invalid { background-color: #FEE2E2; color: #B91C1C; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; border: 1px solid #EF4444; }
    
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .metric-label { font-size: 0.78rem; text-transform: uppercase; color: #64748B; font-weight: 700; }
    .metric-value { font-size: 1.3rem; font-weight: 800; color: #0F172A; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_category_badge(cat_name: str) -> str:
    """Returns HTML styled category badge."""
    mapping = {
        "Inpatient_Medical": ("badge-inpatient", "🏥 Rawat Inap Medis"),
        "Outpatient_Emergency": ("badge-outpatient", "🚑 Rawat Jalan Darurat (IGD)"),
        "Critical_Illness": ("badge-critical", "⚡ Penyakit Kritis"),
        "Personal_Accident": ("badge-accident", "🩹 Kecelakaan Diri"),
        "Inquiry_or_Dispute": ("badge-inquiry", "ℹ️ Pertanyaan & Sengketa"),
    }
    css_class, label = mapping.get(cat_name, ("badge-inquiry", cat_name))
    return f'<span class="{css_class}">{label}</span>'


def get_urgency_badge(urgency: str) -> str:
    """Returns HTML styled urgency badge."""
    u_upper = str(urgency).strip().upper()
    if u_upper == "HIGH":
        return '<span class="badge-urg-high">🔴 HIGH URGENCY</span>'
    elif u_upper in ["MED", "MEDIUM"]:
        return '<span class="badge-urg-med">🟡 MEDIUM URGENCY</span>'
    return '<span class="badge-urg-low">🟢 LOW URGENCY</span>'


def get_schema_badge(is_valid: bool) -> str:
    """Returns HTML schema validation badge."""
    if is_valid:
        return '<span class="badge-valid">✅ 100% VALID PYDANTIC SCHEMA</span>'
    return '<span class="badge-invalid">❌ INVALID JSON / RAW CHAT TEXT</span>'


# --- Model Inference Handlers ---


def check_fastapi_health(api_url: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Checks if FastAPI backend is online and returns health JSON."""
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/health", timeout=1.5)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass
    return False, None


def run_inference_via_api(
    api_url: str,
    narrative: str,
    model_type: str,
    max_new_tokens: int = 150,
) -> Dict[str, Any]:
    """Calls FastAPI POST /v1/triage endpoint."""
    url = f"{api_url.rstrip('/')}/v1/triage"
    payload = {
        "narrative": narrative,
        "model_type": model_type,
        "max_new_tokens": max_new_tokens,
    }
    resp = requests.post(url, json=payload, timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(f"API Error ({resp.status_code}): {resp.text}")
    return resp.json()


def run_inference_direct(
    narrative: str,
    model_type: str,
    max_new_tokens: int = 150,
) -> Dict[str, Any]:
    """Runs inference directly using in-memory singleton TriageService."""
    service = TriageService.get_instance()
    res = service.run_inference(
        narrative=narrative,
        model_type=model_type,
        max_new_tokens=max_new_tokens,
    )
    # Format into dict matching API schema
    parsed = res.get("parsed")
    data_dict = None
    if parsed and isinstance(parsed, dict) and res.get("json_valid"):
        data_dict = {
            "category": str(parsed.get("category", "")),
            "urgency": str(parsed.get("urgency", "")),
            "primary_condition": str(parsed.get("primary_condition", "")),
            "claim_method": str(parsed.get("claim_method", "")),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    return {
        "success": res.get("json_valid", False),
        "model_type": res.get("model_type", ""),
        "model_name": res.get("model_name", ""),
        "latency_ms": res.get("latency_ms", 0.0),
        "json_valid": res.get("json_valid", False),
        "data": data_dict,
        "raw_output": res.get("raw_output", ""),
        "cleaned_json": res.get("cleaned_json"),
        "input_tokens": res.get("input_tokens", 0),
        "generated_tokens": res.get("generated_tokens", 0),
        "tokens_per_second": res.get("tokens_per_second", 0.0),
        "device": res.get("device", ""),
        "parse_error": res.get("parse_error"),
    }


# --- UI Layout ---

def render_sidebar():
    st.sidebar.image(
        "https://raw.githubusercontent.com/QwenLM/Qwen2.5/main/assets/logo.png",
        width=140,
    )
    st.sidebar.title("⚙️ Triage Engine Config")
    
    st.sidebar.markdown(
        """
        **Model**: Qwen2.5-1.5B-Instruct  
        **Tuning**: QLoRA (Rank 16, Alpha 32)  
        **Format**: 16-bit Standalone Safetensors  
        """
    )
    st.sidebar.divider()
    
    st.sidebar.subheader("🔌 Backend Connection")
    backend_choice = st.sidebar.radio(
        "Pilih Mode Serving:",
        ["FastAPI REST API", "Direct In-Process Python"],
        index=0,
        help="FastAPI memanfaatkan service HTTP asynchronous lokal; Direct In-Process memuat model langsung ke memori Python proses ini.",
    )
    
    api_url = "http://localhost:8000"
    api_online = False
    health_data = None
    
    if backend_choice == "FastAPI REST API":
        api_url = st.sidebar.text_input("FastAPI Base URL:", value="http://localhost:8000")
        api_online, health_data = check_fastapi_health(api_url)
        if api_online:
            st.sidebar.success(f"🟢 Connected to FastAPI ({health_data.get('device', 'GPU')})")
            if health_data.get("allocated_vram_mb"):
                st.sidebar.caption(f"VRAM Allocated: {health_data['allocated_vram_mb']} MB")
        else:
            st.sidebar.warning("⚠️ FastAPI Offline pada URL tersebut. Kamu dapat menjalankan `python -m uvicorn src.serving.api:app --port 8000` atau beralih ke Direct In-Process.")
    else:
        st.sidebar.info("🚀 Mode Direct In-Process aktif (Model dimuat saat inferensi pertama).")
    
    st.sidebar.divider()
    st.sidebar.subheader("🎛️ Hyperparameter")
    max_tokens = st.sidebar.slider("Max New Tokens:", min_value=50, max_value=250, value=150, step=10)
    st.sidebar.caption("Suhu: Deterministic (Greedy / T=0.1) untuk konsistensi skema JSON.")

    st.sidebar.divider()
    with st.sidebar.expander("📚 Kriteria 5 Kategori Klaim Resmi"):
        for cat in OFFICIAL_CATEGORIES:
            st.markdown(f"**{cat['name_id']}** (`{cat['category']}`)")
            st.caption(cat["description"])
            st.markdown("---")

    return backend_choice, api_url, api_online, max_tokens


def render_result_column(title: str, subtitle: str, res: Dict[str, Any], is_lora: bool = True):
    st.markdown(f"### {title}")
    st.caption(subtitle)
    
    is_valid = res.get("json_valid", False)
    st.markdown(get_schema_badge(is_valid), unsafe_allow_html=True)
    st.write("")
    
    # Latency & Throughput metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Latency</div>
            <div class="metric-value">{res.get('latency_ms', 0):.1f} <span style="font-size:0.8rem;color:#64748B;">ms</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Throughput</div>
            <div class="metric-value">{res.get('tokens_per_second', 0):.1f} <span style="font-size:0.8rem;color:#64748B;">tok/s</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card">
            <div class="metric-label">Tokens Out</div>
            <div class="metric-value">{res.get('generated_tokens', 0)}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    
    data = res.get("data")
    if is_valid and data:
        # Structured Triage Card
        cat_badge = get_category_badge(data.get("category", ""))
        urg_badge = get_urgency_badge(data.get("urgency", ""))
        method_str = data.get("claim_method", "Unknown")
        
        st.markdown(f"**Klasifikasi Kategori:** {cat_badge}", unsafe_allow_html=True)
        st.markdown(f"**Tingkat Urgensi:** {urg_badge}", unsafe_allow_html=True)
        st.markdown(f"**Metode Klaim:** `💳 {method_str}`")
        st.markdown(f"**Diagnosis / Kondisi Utama:**")
        st.info(f"🩺 {data.get('primary_condition', '-')}")
        st.markdown(f"**Penalaran Triage:**")
        st.write(data.get("reasoning", "-"))
    else:
        st.error("Model tidak mengembalikan struktur JSON yang valid sesuai skema Pydantic `ClaimTriageOutput`.")
        if res.get("parse_error"):
            st.warning(f"Error Parser: `{res.get('parse_error')}`")
    
    with st.expander("🔍 Lihat Raw Output Model & JSON"):
        if res.get("cleaned_json"):
            st.markdown("**Cleaned JSON String:**")
            st.code(res.get("cleaned_json"), language="json")
        st.markdown("**Raw Generation Output:**")
        st.code(res.get("raw_output", ""), language="text")


def main():
    backend_choice, api_url, api_online, max_tokens = render_sidebar()

    st.markdown('<div class="main-title">🏥 Indonesian Insurance Claim & FNOL Triage AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Perbandingan Langsung (Side-by-Side): Base Foundation Model vs Fine-Tuned LoRA Merged Model (Qwen2.5-1.5B) untuk Otomasi First Notice of Loss (FNOL) Asuransi Jiwa & Kesehatan di Indonesia.</div>',
        unsafe_allow_html=True,
    )

    # Key Performance Highlights Bar
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("🎯 Category Accuracy", "98.5%", "+55.0% vs Base Zero-Shot")
    with k2:
        st.metric("📊 Macro F1-Score", "0.9846", "Optimal Class Balance")
    with k3:
        st.metric("🛡️ JSON Schema Validity", "100%", "Strict Pydantic Contract")
    with k4:
        st.metric("⚡ Inference Latency", "~820 ms", "RTX 3050 Laptop GPU (3.1 GB)")

    st.divider()

    # Input Section
    st.subheader("📝 Input Laporan Klaim Nasabah (FNOL Narrative)")
    
    # Preset selection
    scenario_titles = [f"[{s['category']}] {s['title']}" for s in PRESET_SCENARIOS]
    scenario_options = ["-- Pilih Skenario Preset Kasus --"] + scenario_titles + ["✍️ Custom Laporan Klaim Sendiri"]
    
    selected_option = st.selectbox("Pilih Template Skenario atau Tulis Manual:", scenario_options, index=1)
    
    default_text = ""
    if selected_option and selected_option != "-- Pilih Skenario Preset Kasus --" and selected_option != "✍️ Custom Laporan Klaim Sendiri":
        idx = scenario_titles.index(selected_option)
        default_text = PRESET_SCENARIOS[idx]["narrative"]
    elif selected_option == "✍️ Custom Laporan Klaim Sendiri":
        default_text = ""

    narrative_input = st.text_area(
        "Teks Kronologi / Percakapan Nasabah:",
        value=default_text,
        height=130,
        placeholder="Contoh: Kemarin sore saya terjatuh dari sepeda motor di jalan tol dan patah tulang paha...",
    )

    col_btn1, col_btn2, _ = st.columns([2, 2, 4])
    with col_btn1:
        run_compare_btn = st.button("🚀 Analisis Side-by-Side (Base vs LoRA)", type="primary", use_container_width=True)
    with col_btn2:
        run_lora_only_btn = st.button("⚡ Model LoRA Produksi Saja", use_container_width=True)

    if run_compare_btn or run_lora_only_btn:
        if not narrative_input or len(narrative_input.strip()) < 10:
            st.error("Mohon masukkan laporan narasi klaim minimal 10 karakter!")
            return

        is_side_by_side = run_compare_btn

        use_api = (backend_choice == "FastAPI REST API") and api_online

        with st.spinner("Menjalankan inferensi model pada kartu grafis..."):
            try:
                # 1. Run LoRA model
                if use_api:
                    lora_res = run_inference_via_api(api_url, narrative_input, "lora", max_tokens)
                else:
                    lora_res = run_inference_direct(narrative_input, "lora", max_tokens)
                
                # 2. Run Base model if side-by-side
                base_res = None
                if is_side_by_side:
                    if use_api:
                        base_res = run_inference_via_api(api_url, narrative_input, "base", max_tokens)
                    else:
                        base_res = run_inference_direct(narrative_input, "base", max_tokens)

            except Exception as e:
                st.error(f"Gagal melakukan inferensi: {e}")
                return

        st.divider()
        st.subheader("📊 Hasil Evaluasi & Triase Klaim")

        if is_side_by_side and base_res is not None:
            c_base, c_lora = st.columns(2)
            with c_base:
                render_result_column(
                    title="Foundation Model (Zero-Shot)",
                    subtitle="Qwen/Qwen2.5-1.5B-Instruct tanpa penyesuaian domain",
                    res=base_res,
                    is_lora=False,
                )
            with c_lora:
                render_result_column(
                    title="Fine-Tuned LoRA Merged Model",
                    subtitle="Qwen2.5-1.5B + 4-bit QLoRA Golden Adapter (Merged Standalone)",
                    res=lora_res,
                    is_lora=True,
                )
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                render_result_column(
                    title="Fine-Tuned LoRA Merged Model",
                    subtitle="Qwen2.5-1.5B + 4-bit QLoRA Golden Adapter (Merged Standalone)",
                    res=lora_res,
                    is_lora=True,
                )
            with c2:
                st.markdown("### 📋 Ringkasan Keputusan FNOL")
                st.write("Sistem inferensi AI telah menyelesaikan verifikasi klaim secara otomatis:")
                if lora_res.get("json_valid") and lora_res.get("data"):
                    d = lora_res["data"]
                    st.success(f"**Rekomendasi Tindakan:** Teruskan tiket klaim ke antrean tim `{d.get('category')}` dengan prioritas `{d.get('urgency')}`.")
                    st.info(f"**Prosedur Pembayaran:** `{d.get('claim_method')}`.")
                st.markdown(
                    """
                    **Integrasi Sistem Produksi**:
                    - REST API siap dipanggil oleh sistem Core Insurance (Life & Health Insurance).
                    - Format response 100% konsisten dengan skema OpenAPI `/docs`.
                    - Latensi p95 < 1200 ms untuk penanganan real-time customer WhatsApp Bot atau Mobile App.
                    """
                )

    # Architectural & Benchmark Summary Tabs
    st.divider()
    st.subheader("📑 Architecture & Benchmarking Context")
    tab1, tab2, tab3 = st.tabs(["📊 Executive Benchmark Table", "🧩 Confusion Matrix Breakdown", "🏗️ Deployment & Architecture"])

    with tab1:
        st.markdown(
            """
            | Matriks Evaluasi | Base Model (Zero-Shot) | Fine-Tuned LoRA (Golden Adapter) | Delta Peningkatan |
            | :--- | :--- | :--- | :--- |
            | **Overall Accuracy** | 43.50% | **98.50%** | **+55.00%** |
            | **Macro F1-Score** | 0.3872 | **0.9846** | **+0.5974** |
            | **JSON Schema Compliance** | 38.00% | **100.00%** | **+62.00%** |
            | **Catastrophic Forgetting Degradation** | Baseline | **11.08%** (Toleransi <15%) | Sangat Stabil |
            | **VRAM Consumption (FP16 Standalone)** | 3.12 GB | **3.08 GB** | Konsumsi Hemat Edge GPU |
            | **Inference Latency (Avg)** | ~1420 ms | **~820 ms** | -42.2% Pangkas Durasi |
            """
        )

    with tab2:
        st.markdown(
            """
            **Distribusi Kategori Uji (Test Split: 200 Sampel):**
            - **Critical_Illness**: Precision 100.0%, Recall 97.5%, F1 0.987
            - **Inpatient_Medical**: Precision 97.6%, Recall 100.0%, F1 0.988
            - **Outpatient_Emergency**: Precision 97.4%, Recall 95.0%, F1 0.962
            - **Personal_Accident**: Precision 97.5%, Recall 100.0%, F1 0.987
            - **Inquiry_or_Dispute**: Precision 100.0%, Recall 100.0%, F1 1.000
            """
        )
        st.info("💡 Model LoRA mempertahankan presisi 100% pada kategori risiko tinggi `Critical_Illness` dan `Inquiry_or_Dispute`.")

    with tab3:
        st.markdown(
            """
            **Arsitektur Penyajian Model Multi-Format:**
            1. **FastAPI REST Endpoint (`src/serving/api.py`)**: Asynchronous REST service dengan OpenAPI documentation di `/docs`.
            2. **Merged Standalone Safetensors (`models/merged_qwen2.5_1.5b_insurance_triage/`)**: 16-bit standalone model tanpa dependensi library `peft`.
            3. **Ollama Packaging (`configs/Modelfile`)**: Siap didistribusikan untuk edge inference on-premise dengan command `ollama create insurance-triage -f configs/Modelfile`.
            4. **Streamlit Interactive UI (`app/streamlit_app.py`)**: Web application interaktif untuk demonstrasi stakeholder bisnis.
            """
        )


if __name__ == "__main__":
    main()
