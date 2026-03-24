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

    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)

    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")

    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="Yes")
    StreamingMovies: str = Field(..., example="Yes")

    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")

    MonthlyCharges: float = Field(..., example=89.85)
    TotalCharges: float = Field(..., example=1081.25)


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
