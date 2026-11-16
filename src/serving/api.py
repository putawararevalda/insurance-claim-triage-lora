"""Production FastAPI Inference REST API for Indonesian Insurance Claim & FNOL Triage.

Provides production-grade endpoints for automated insurance claim classification,
deterministic Pydantic schema validation, health and GPU/VRAM monitoring,
latency measurement middleware, and official triage criteria reference.
"""

from datetime import datetime, timezone
import logging
import os
import sys
import time
from typing import Any, Dict, List, Literal, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.schema import ClaimCategory, ClaimMethod, ClaimTriageOutput, UrgencyLevel
from src.service import TriageService, clean_json_text, fallback_parse_triage_json

logger = logging.getLogger("serving_api")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

API_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0-qwen2.5-1.5b-qlora"

# Initialize FastAPI application
app = FastAPI(
    title="Indonesian Insurance Claim & FNOL Triage API",
    description=(
        "Production-grade local inference REST API for First Notice of Loss (FNOL) triage "
        "using fine-tuned Qwen2.5-1.5B with 4-bit QLoRA Golden Adapter."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local Streamlit UI or web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Middleware: Request Latency Profiling ---


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures end-to-end request duration and sets X-Process-Time-Ms header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# --- Exception Handlers ---


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom structured response for validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "RequestValidationError",
            "detail": jsonable_encoder(exc.errors()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# --- Request & Response Schemas ---


class TriageRequest(BaseModel):
    narrative: str = Field(
        ...,
        min_length=10,
        description="Laporan atau kronologi klaim nasabah dalam Bahasa Indonesia.",
        examples=[
            "Kemarin sore saya terjatuh dari tangga saat membersihkan plafon rumah dan mengalami patah tulang lengan kanan tertutup. Saya langsung dilarikan ke IGD RS Medika dan ditangani gips. Mau klaim biaya perawatan dan rontgen."
        ],
    )
    model_type: Literal["lora", "base"] = Field(
        default="lora",
        description="Pilihan model: 'lora' (Golden Adapter Merged) atau 'base' (Zero-shot Base Foundation).",
    )
    max_new_tokens: int = Field(
        default=150,
        ge=30,
        le=300,
        description="Maksimal token baru yang dihasilkan.",
    )

    @field_validator("narrative")
    @classmethod
    def validate_narrative_not_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if len(trimmed) < 10:
            raise ValueError("Narrative klaim tidak boleh kosong atau kurang dari 10 karakter setelah spasi dihilangkan.")
        return trimmed


class TriageResultData(BaseModel):
    category: str = Field(..., description="Kategori klaim asuransi terverifikasi.")
    urgency: str = Field(..., description="Tingkat urgensi triage (Low, Medium, High).")
    primary_condition: str = Field(..., description="Kondisi medis atau pokok permasalahan utama.")
    claim_method: str = Field(..., description="Metode pengajuan klaim (Cashless, Reimbursement, Unknown).")
    reasoning: str = Field(..., description="Alasan singkat klasifikasi triage.")


class TriageResponse(BaseModel):
    success: bool
    model_type: str
    model_name: str
    model_version: str = MODEL_VERSION
    timestamp: str
    latency_ms: float
    json_valid: bool
    data: Optional[TriageResultData] = None
    raw_output: str
    cleaned_json: Optional[str] = None
    input_tokens: int
    generated_tokens: int
    tokens_per_second: float
    device: str
    parse_error: Optional[str] = None


class CompareRequest(BaseModel):
    narrative: str = Field(
        ...,
        min_length=10,
        description="Laporan atau kronologi klaim nasabah untuk dikomparasi.",
    )
    max_new_tokens: int = Field(default=150, ge=30, le=300)

    @field_validator("narrative")
    @classmethod
    def validate_narrative_not_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if len(trimmed) < 10:
            raise ValueError("Narrative klaim tidak boleh kosong atau kurang dari 10 karakter.")
        return trimmed


class CompareResponse(BaseModel):
    narrative: str
    lora: TriageResponse
    base: TriageResponse


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    device: str
    cuda_available: bool
    device_name: Optional[str] = None
    allocated_vram_mb: Optional[float] = None
    reserved_vram_mb: Optional[float] = None
    models_loaded: Dict[str, bool]


class CategoryCriteria(BaseModel):
    category: str
    name_id: str
    description: str
    typical_urgency: str
    typical_method: str
    criteria: List[str]
    sample_conditions: List[str]


class CategoriesResponse(BaseModel):
    total_categories: int
    categories: List[CategoryCriteria]


# --- Reference Data: 5 Official Claim Categories & Criteria ---

OFFICIAL_CATEGORIES: List[Dict[str, Any]] = [
    {
        "category": "Inpatient_Medical",
        "name_id": "Rawat Inap Medis",
        "description": "Perawatan medis di rumah sakit yang memerlukan rawat inap (opname/hospitalisasi > 24 jam) untuk penanganan penyakit akut, infeksi sistemik, atau tindakan bedah terencana.",
        "typical_urgency": "High / Medium",
        "typical_method": "Cashless / Reimbursement",
        "criteria": [
            "Memerlukan fasilitas kamar inap / bed rumah sakit lebih dari 24 jam",
            "Indikasi medis diagnosis berat seperti demam berdarah dengue, tifus abdominalis, pneumonia, atau apendisitis",
            "Melibatkan biaya kamar rawat inap, visit dokter spesialis, dan obat-obatan infus",
        ],
        "sample_conditions": [
            "Demam Berdarah Dengue (DBD) dengan Trombositopenia",
            "Tifus Abdominalis / Demam Tifoid",
            "Pneumonia Bakterial",
            "Apendisitis Akut (Operasi Usus Buntu)",
        ],
    },
    {
        "category": "Outpatient_Emergency",
        "name_id": "Rawat Jalan Darurat (IGD)",
        "description": "Penanganan medis gawat darurat yang ditangani di Instalasi Gawat Darurat (IGD) tanpa memerlukan opname, serta kontrol / perawatan pasca tindakan gawat darurat.",
        "typical_urgency": "Medium / High",
        "typical_method": "Reimbursement / Cashless",
        "criteria": [
            "Penanganan darurat langsung di IGD rumah sakit atau klinik 24 jam",
            "Tindakan hecting (jahit luka robek), debridement luka, atau observasi non-opname < 24 jam",
            "Klaim kontrol medis atau fisioterapi pasca tindakan darurat",
        ],
        "sample_conditions": [
            "Luka Robek Teriris Pisau (Hecting 6 Jahitan)",
            "Luka Bakar Ringan Derajat I-II",
            "Serangan Asma Akut Ringan di IGD",
            "Tertusuk Paku Karatan (Injeksi ATS / Tetanus)",
        ],
    },
    {
        "category": "Critical_Illness",
        "name_id": "Penyakit Kritis (Major Illness)",
        "description": "Kondisi penyakit kritis yang mengancam jiwa atau menyebabkan disabilitas permanen sesuai daftar klausul polis asuransi penyakit kritis katastropik.",
        "typical_urgency": "High",
        "typical_method": "Cashless Darurat / Akselerasi Manfaat Polis",
        "criteria": [
            "Diagnosis penyakit kritis katastropik (stadium lanjut atau tindakan intervensi bedah mayor)",
            "Memerlukan penanganan intensif ICU/ICCU/HCU",
            "Klaim pencairan uang pertanggungan penyakit kritis atau penjaminan rawat intensif seketika",
        ],
        "sample_conditions": [
            "Serangan Jantung Akut (STEMI / PCI Pasang Ring)",
            "Stroke Iskemik / Hemoragik Akut",
            "Kanker Stadium Lanjut (Kemoterapi / Radioterapi)",
            "Gagal Ginjal Terminal (Hemodialisis Rutin)",
        ],
    },
    {
        "category": "Personal_Accident",
        "name_id": "Kecelakaan Diri",
        "description": "Cedera fisik atau luka traumatik yang diakibatkan langsung oleh peristiwa kekerasan eksternal mendadak, tidak disengaja, dan kasat mata.",
        "typical_urgency": "High / Medium",
        "typical_method": "Cashless / Reimbursement",
        "criteria": [
            "Cedera traumatik akibat insiden eksternal mendadak (laka lantas, terjatuh, benturan fisik)",
            "Fraktur tulang (patah tulang terbuka/tertutup), dislokasi sendi, atau trauma kepala",
            "Memerlukan kronologi kejadian / berita acara kepolisian untuk verifikasi",
        ],
        "sample_conditions": [
            "Patah Tulang Lengan Kanan Tertutup Akibat Terjatuh dari Tangga",
            "Cedera Fraktur Klavikula Tabrakan Sepeda Motor",
            "Dislokasi Bahu Akibat Terpeleset",
            "Trauma Fisik Akibat Kecelakaan Kerja di Pabrik",
        ],
    },
    {
        "category": "Inquiry_or_Dispute",
        "name_id": "Pertanyaan & Sengketa Klaim",
        "description": "Permintaan informasi umum, konfirmasi klausul polis, status klaim berjalan, atau sanggahan penolakan klaim (dispute) tanpa klaim medis baru.",
        "typical_urgency": "Low",
        "typical_method": "Unknown",
        "criteria": [
            "Tidak ada tindakan medis aktif baru yang memerlukan penjaminan seketika",
            "Pertanyaan mengenai masa tunggu (waiting period) penyakit khusus atau batasan limit polis",
            "Sanggahan / banding atas klaim yang ditolak atau keluhan kelengkapan administrasi",
        ],
        "sample_conditions": [
            "Pertanyaan Masa Tunggu Polis Penyakit Khusus",
            "Konfirmasi Batasan Plafon Kamar Rawat Inap Polis Murni",
            "Sanggahan Penolakan Klaim Tindakan Kuretase",
            "Pengecekan Status Berkas Dokumen Reimbursement",
        ],
    },
]

# Preset sample scenarios for quick API / UI testing
PRESET_SCENARIOS = [
    {
        "id": "crit_01",
        "category": "Critical_Illness",
        "title": "Serangan Jantung Koroner Akut (ICU)",
        "narrative": "Suami saya usia 52 tahun mendadak nyeri dada hebat tembus ke punggung dan sesak napas. Hasil EKG RS Jantung Harapan Kita menyatakan STEMI akut dan dokter spesialis menyarankan tindakan pasang ring darurat (PCI). Kami butuh persetujuan penjaminan klaim kritis sekarang juga.",
        "expected_urgency": "High",
        "expected_method": "Cashless",
    },
    {
        "id": "acc_01",
        "category": "Personal_Accident",
        "title": "Patah Tulang Jatuh dari Tangga",
        "narrative": "Kemarin sore saya terjatuh dari tangga saat membersihkan plafon rumah dan mengalami patah tulang lengan kanan tertutup. Saya langsung dilarikan ke IGD RS Medika dan ditangani gips. Mau klaim biaya perawatan dan rontgen no polis POL-882190.",
        "expected_urgency": "Medium",
        "expected_method": "Cashless",
    },
    {
        "id": "inp_01",
        "category": "Inpatient_Medical",
        "title": "Rawat Inap Demam Berdarah Dengue",
        "narrative": "Anak saya panas tinggi 4 hari naik turun dan keluar bintik merah di tangan. Hasil lab trombosit turun ke 42.000. Dokter anak di RS Siloam mewajibkan opname rawat inap hari ini untuk infus cairan intravena.",
        "expected_urgency": "High",
        "expected_method": "Cashless",
    },
    {
        "id": "out_01",
        "category": "Outpatient_Emergency",
        "title": "Rawat Jalan Darurat Luka Robek Pisau Dapur",
        "narrative": "Tadi siang jari telunjuk teriris pisau dapur saat memotong daging, luka robek cukup dalam dan darah mengucur deras. Saya segera ke IGD terdekat untuk pembersihan luka dan dijahit 6 jahitan tanpa rawat inap. Biaya sudah saya bayar tunai Rp 850.000, ingin reimbursement.",
        "expected_urgency": "Medium",
        "expected_method": "Reimbursement",
    },
    {
        "id": "inq_01",
        "category": "Inquiry_or_Dispute",
        "title": "Pertanyaan Masa Tunggu Polis Penyakit Khusus",
        "narrative": "Halo customer care asuransi, saya mau menanyakan apakah tindakan kuretase polip rahim sudah tercover jika polis saya baru berjalan 8 bulan? Mohon informasi klausul masa tunggu penyakit khusus dan prosedur pengajuan pra-otorisasinya.",
        "expected_urgency": "Low",
        "expected_method": "Unknown",
    },
]


# --- Helper Function ---


def _format_service_result(res: Dict[str, Any]) -> TriageResponse:
    parsed = res.get("parsed")
    data_obj = None
    if parsed and isinstance(parsed, dict) and res.get("json_valid"):
        data_obj = TriageResultData(
            category=str(parsed.get("category", "")),
            urgency=str(parsed.get("urgency", "")),
            primary_condition=str(parsed.get("primary_condition", "")),
            claim_method=str(parsed.get("claim_method", "")),
            reasoning=str(parsed.get("reasoning", "")),
        )

    return TriageResponse(
        success=res.get("json_valid", False),
        model_type=res.get("model_type", ""),
        model_name=res.get("model_name", ""),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=res.get("latency_ms", 0.0),
        json_valid=res.get("json_valid", False),
        data=data_obj,
        raw_output=res.get("raw_output", ""),
        cleaned_json=res.get("cleaned_json"),
        input_tokens=res.get("input_tokens", 0),
        generated_tokens=res.get("generated_tokens", 0),
        tokens_per_second=res.get("tokens_per_second", 0.0),
        device=res.get("device", ""),
        parse_error=res.get("parse_error"),
    )


# --- Endpoints ---


@app.get("/", summary="API Root & Overview")
def root() -> Dict[str, Any]:
    """Root endpoint detailing API capabilities and service documentation."""
    return {
        "service": "Indonesian Insurance Claim & FNOL Triage API",
        "version": API_VERSION,
        "model_version": MODEL_VERSION,
        "framework": "FastAPI + Uvicorn + Qwen2.5-1.5B QLoRA",
        "endpoints": {
            "health": "/health",
            "categories": "GET /v1/categories",
            "triage": "POST /v1/triage",
            "compare": "POST /v1/compare",
            "scenarios": "GET /v1/scenarios",
            "documentation": "/docs",
        },
        "target_industry": "Life & Health Insurance (Indonesia)",
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check & VRAM Status")
def health_check() -> HealthResponse:
    """Returns service readiness, PyTorch device info, GPU VRAM allocation, and model status."""
    service = TriageService.get_instance()
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"

    alloc_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 2) if cuda_avail else None
    res_mb = round(torch.cuda.memory_reserved() / (1024 * 1024), 2) if cuda_avail else None

    return HealthResponse(
        status="healthy",
        service="Indonesian Insurance Claim & FNOL Triage API",
        version=API_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        device=service.device,
        cuda_available=cuda_avail,
        device_name=device_name,
        allocated_vram_mb=alloc_mb,
        reserved_vram_mb=res_mb,
        models_loaded={
            "lora_model": service.lora_model is not None,
            "base_model": service.base_model is not None,
        },
    )


@app.get("/v1/categories", response_model=CategoriesResponse, summary="Get Official Claim Categories & Criteria")
def get_categories() -> CategoriesResponse:
    """Returns the 5 official insurance claim categories with descriptions, triage criteria, and samples."""
    return CategoriesResponse(
        total_categories=len(OFFICIAL_CATEGORIES),
        categories=[CategoryCriteria(**cat) for cat in OFFICIAL_CATEGORIES],
    )


@app.get("/v1/scenarios", summary="Get Preset Claim Scenarios")
def get_preset_scenarios() -> List[Dict[str, Any]]:
    """Returns preset claim scenarios for testing and demonstration."""
    return PRESET_SCENARIOS


@app.post("/v1/triage", response_model=TriageResponse, summary="Perform Insurance Claim Triage")
def triage_claim(req: TriageRequest) -> TriageResponse:
    """Performs FNOL claim triage on the customer narrative, producing verified ClaimTriageOutput."""
    service = TriageService.get_instance()
    try:
        res = service.run_inference(
            narrative=req.narrative,
            model_type=req.model_type,
            max_new_tokens=req.max_new_tokens,
        )
        return _format_service_result(res)
    except Exception as e:
        logger.error(f"Inference error in /v1/triage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/v1/compare", response_model=CompareResponse, summary="Compare Base vs LoRA Side-by-Side")
def compare_claim(req: CompareRequest) -> CompareResponse:
    """Executes side-by-side inference with both Base model and LoRA fine-tuned model."""
    service = TriageService.get_instance()
    try:
        lora_res = service.run_inference(
            narrative=req.narrative,
            model_type="lora",
            max_new_tokens=req.max_new_tokens,
        )
        base_res = service.run_inference(
            narrative=req.narrative,
            model_type="base",
            max_new_tokens=req.max_new_tokens,
        )
        return CompareResponse(
            narrative=req.narrative,
            lora=_format_service_result(lora_res),
            base=_format_service_result(base_res),
        )
    except Exception as e:
        logger.error(f"Comparison error in /v1/compare: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison error: {str(e)}",
        )
