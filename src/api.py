"""FastAPI Local Serving Application for Indonesian Insurance Claim & FNOL Triage.

Compatibility wrapper forwarding to src.serving.api.
"""

from src.serving.api import (
    app,
    API_VERSION,
    MODEL_VERSION,
    OFFICIAL_CATEGORIES,
    PRESET_SCENARIOS,
    CategoriesResponse,
    CategoryCriteria,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    TriageRequest,
    TriageResponse,
    TriageResultData,
    TriageService,
    _format_service_result,
)

__all__ = [
    "app",
    "API_VERSION",
    "MODEL_VERSION",
    "OFFICIAL_CATEGORIES",
    "PRESET_SCENARIOS",
    "CategoriesResponse",
    "CategoryCriteria",
    "CompareRequest",
    "CompareResponse",
    "HealthResponse",
    "TriageRequest",
    "TriageResponse",
    "TriageResultData",
    "TriageService",
    "_format_service_result",
]
