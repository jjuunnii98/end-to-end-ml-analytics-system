from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# =========================
# Request Schemas
# =========================

class ChurnPredictionRequest(BaseModel):
    """
    Input schema for churn prediction.

    This must match the model input columns defined in feature_schema.py
    (excluding id/target columns).
    """

    gender: str = Field(..., json_schema_extra={"example": "Female"})
    SeniorCitizen: int = Field(..., json_schema_extra={"example": 0})
    Partner: str = Field(..., json_schema_extra={"example": "Yes"})
    Dependents: str = Field(..., json_schema_extra={"example": "No"})
    tenure: int = Field(..., json_schema_extra={"example": 12})

    PhoneService: str = Field(..., json_schema_extra={"example": "Yes"})
    MultipleLines: str = Field(..., json_schema_extra={"example": "No"})

    InternetService: str = Field(..., json_schema_extra={"example": "Fiber optic"})
    OnlineSecurity: str = Field(..., json_schema_extra={"example": "No"})
    OnlineBackup: str = Field(..., json_schema_extra={"example": "Yes"})
    DeviceProtection: str = Field(..., json_schema_extra={"example": "No"})
    TechSupport: str = Field(..., json_schema_extra={"example": "No"})
    StreamingTV: str = Field(..., json_schema_extra={"example": "Yes"})
    StreamingMovies: str = Field(..., json_schema_extra={"example": "Yes"})

    Contract: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    PaperlessBilling: str = Field(..., json_schema_extra={"example": "Yes"})
    PaymentMethod: str = Field(..., json_schema_extra={"example": "Electronic check"})

    MonthlyCharges: float = Field(..., json_schema_extra={"example": 89.85})
    TotalCharges: float = Field(..., json_schema_extra={"example": 1081.25})


# =========================
# Response Schemas
# =========================

class PredictionItem(BaseModel):
    """
    Per-row prediction result.
    """

    predicted_label: int
    model_label: int
    churn_probability: float


class ChurnPredictionResponse(BaseModel):
    """
    Output schema for churn prediction.
    """

    model_name: str
    artifact_source: str

    input_shape: List[int]
    transformed_shape: List[int]
    threshold: float

    model_labels: List[int]
    probabilities: List[float]

    predictions: List[PredictionItem]


# =========================
# Health Check Schema
# =========================

class HealthResponse(BaseModel):
    status: str = "ok"
