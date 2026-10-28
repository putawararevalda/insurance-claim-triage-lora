"""Pydantic schema definitions for Insurance Claim & FNOL Triage."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ClaimCategory(str, Enum):
    INPATIENT_MEDICAL = "Inpatient_Medical"
    OUTPATIENT_EMERGENCY = "Outpatient_Emergency"
    CRITICAL_ILLNESS = "Critical_Illness"
    PERSONAL_ACCIDENT = "Personal_Accident"
    INQUIRY_OR_DISPUTE = "Inquiry_or_Dispute"


class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ClaimMethod(str, Enum):
    CASHLESS = "Cashless"
    REIMBURSEMENT = "Reimbursement"
    UNKNOWN = "Unknown"


class ClaimTriageOutput(BaseModel):
    category: ClaimCategory = Field(
        ...,
        description="Kategori klaim asuransi: Inpatient_Medical, Outpatient_Emergency, Critical_Illness, Personal_Accident, Inquiry_or_Dispute."
    )
    urgency: UrgencyLevel = Field(
        ...,
        description="Tingkat urgensi penanganan klaim: Low, Medium, High."
    )
    primary_condition: str = Field(
        ...,
        description="Kondisi medis atau pokok permasalahan utama yang dilaporkan nasabah."
    )
    claim_method: ClaimMethod = Field(
        ...,
        description="Metode pengajuan klaim: Cashless, Reimbursement, atau Unknown."
    )
    reasoning: str = Field(
        ...,
        description="Alasan singkat klasifikasi triage dalam Bahasa Indonesia (1-2 kalimat)."
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for cat in ClaimCategory:
                if v_clean == cat.value.lower():
                    return cat
            # Domain normalization
            if "inpatient" in v_clean or "rawat_inap" in v_clean or "opname" in v_clean:
                return ClaimCategory.INPATIENT_MEDICAL
            if "outpatient" in v_clean or "emergency" in v_clean or "igd" in v_clean or "jalan" in v_clean or "cutan" in v_clean or "patient" in v_clean or "arrival" in v_clean:
                return ClaimCategory.OUTPATIENT_EMERGENCY
            if "critical" in v_clean or "kritis" in v_clean:
                return ClaimCategory.CRITICAL_ILLNESS
            if "accident" in v_clean or "kecelakaan" in v_clean or "injury" in v_clean or "trauma" in v_clean or "fraktur" in v_clean:
                return ClaimCategory.PERSONAL_ACCIDENT
            if "inquiry" in v_clean or "dispute" in v_clean or "tanya" in v_clean or "komplain" in v_clean or "claim" in v_clean or "reimburse" in v_clean or "eligib" in v_clean:
                return ClaimCategory.INQUIRY_OR_DISPUTE
            return ClaimCategory.OUTPATIENT_EMERGENCY
        return v

    @field_validator("urgency", mode="before")
    @classmethod
    def normalize_urgency(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for u in UrgencyLevel:
                if v_clean == u.value.lower():
                    return u
            if "high" in v_clean or "tinggi" in v_clean or "darurat" in v_clean or "kritis" in v_clean or "urgent" in v_clean:
                return UrgencyLevel.HIGH
            if "low" in v_clean or "rendah" in v_clean:
                return UrgencyLevel.LOW
            if "med" in v_clean or "sedang" in v_clean:
                return UrgencyLevel.MEDIUM
            return UrgencyLevel.MEDIUM
        return v

    @field_validator("claim_method", mode="before")
    @classmethod
    def normalize_claim_method(cls, v):
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for m in ClaimMethod:
                if v_clean == m.value.lower():
                    return m
            if "cash" in v_clean or "kartu" in v_clean:
                return ClaimMethod.CASHLESS
            if "reimburse" in v_clean or "tunai" in v_clean or "bayar" in v_clean or "klaim" in v_clean or "payment" in v_clean:
                return ClaimMethod.REIMBURSEMENT
            return ClaimMethod.UNKNOWN
        return v
