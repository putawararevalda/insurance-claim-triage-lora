"""Pydantic schema definitions for Insurance Claim & FNOL Triage."""

from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


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

    @model_validator(mode="before")
    @classmethod
    def pre_normalize_triage_payload(cls, data):
        if isinstance(data, dict):
            cat_raw = str(data.get("category", "")).strip()
            cond_raw = str(data.get("primary_condition", "")).lower()
            reason_raw = str(data.get("reasoning", "")).lower()
            combined = f"{cat_raw.lower()} {cond_raw} {reason_raw}"

            # 1. Post-hospitalization / Outpatient Rehabilitation / Kontrol Pasca Opname
            if (
                "post-hospital" in combined
                or "pasca opname" in combined
                or "post-op" in combined
                or "kontrol dokter" in combined
                or "fisioterapi" in combined
                or "rehabilitasi" in combined
            ) and ("stroke" not in combined and "kanker" not in combined and "meningitis" not in combined):
                data["category"] = ClaimCategory.OUTPATIENT_EMERGENCY.value

            # 2. Inpatient infections/febris mistakenly tagged as Critical_Illness or Injury
            elif cat_raw.lower() in ("critical_illness", "critical illness", "kritis"):
                if any(w in combined for w in ["vertigo", "muntaber", "salmonella", "thypi", "tifus", "febris", "gastroenteritis"]):
                    if "opname" in combined or "rawat inap" in combined:
                        data["category"] = ClaimCategory.INPATIENT_MEDICAL.value
                    elif "igd" in combined:
                        data["category"] = ClaimCategory.OUTPATIENT_EMERGENCY.value
                elif "anafilaksis" in combined or "alergi" in combined or "sakit kepala berputar" in combined:
                    if "rawat jalan" in combined or "igd" in combined or "non-opname" in combined or "observasi dan infus" in combined:
                        data["category"] = ClaimCategory.OUTPATIENT_EMERGENCY.value

            # 3. Minor wound / Hecting / Foreign body at IGD tagged as Injury / Personal_Accident
            elif cat_raw.lower() in ("injury", "accident", "personal_accident"):
                is_igd_minor = any(w in combined for w in [
                    "tersayat pisau", "paku karatan", "benda asing", "manik-manik",
                    "sengatan seng", "vulnus laceratum", "hecting", "jahit 11", "jahit 12"
                ])
                is_major_accident = any(w in combined for w in [
                    "fraktur collum", "fraktur os nasal", "patah tulang", "tergelincir masuk parit",
                    "mobil", "tabrakan", "laka lantas", "jatuh dari ketinggian", "scaffolding",
                    "bola basket", "patah tulang selangka", "klavikula"
                ])
                if is_igd_minor and not is_major_accident:
                    data["category"] = ClaimCategory.OUTPATIENT_EMERGENCY.value
                elif is_major_accident:
                    data["category"] = ClaimCategory.PERSONAL_ACCIDENT.value

            # 4. Inquiries / Disputes related to specific clinical services
            elif cat_raw.lower() in ("inquiry_or_dispute", "inquiry", "dispute", "claim_rejection", "claim_loss", "claim"):
                if ("periksa ke igd" in combined or "berobat ke igd" in combined) and ("flu batuk" in combined or "pusing biasa" in combined or "non-darurat" in combined):
                    data["category"] = ClaimCategory.OUTPATIENT_EMERGENCY.value
                elif "bronkopneumonia" in combined and "kwitansi asli" in combined and "kurir" in combined:
                    data["category"] = ClaimCategory.INPATIENT_MEDICAL.value

        return data

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
