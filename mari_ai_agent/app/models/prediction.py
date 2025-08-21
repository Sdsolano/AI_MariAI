# app/api/v1/models/prediction.py
"""
Pydantic models for ML prediction endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class RiskLevel(str, Enum):
    """Risk level categories"""
    BAJO = "BAJO"
    MEDIO = "MEDIO" 
    ALTO = "ALTO"
    CRITICO = "CRITICO"

class PredictionRequest(BaseModel):
    """Request model for individual prediction"""
    student_id: int = Field(..., description="Student ID for prediction")
    
class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    student_ids: List[int] = Field(..., description="List of student IDs")
    
class KeyFactor(BaseModel):
    """Key factor affecting risk prediction"""
    factor: str = Field(..., description="Factor name")
    value: float = Field(..., description="Factor value")
    impact: str = Field(..., description="Impact description")
    
class RecommendedAction(BaseModel):
    """Recommended action for student"""
    action: str = Field(..., description="Action to take")
    priority: str = Field(..., description="Priority level")
    description: str = Field(..., description="Action description")

class PredictionResponse(BaseModel):
    """Response model for individual prediction"""
    student_id: int = Field(..., description="Student ID")
    risk_level: RiskLevel = Field(..., description="Predicted risk level")
    risk_probability: float = Field(..., ge=0, le=1, description="Risk probability (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    key_factors: List[KeyFactor] = Field(..., description="Top factors affecting prediction")
    recommended_actions: List[RecommendedAction] = Field(..., description="Recommended interventions")
    model_used: str = Field(..., description="ML model used for prediction")
    prediction_timestamp: str = Field(..., description="When prediction was made")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    total_students: int = Field(..., description="Total students processed")
    successful_predictions: int = Field(..., description="Successful predictions")
    failed_predictions: int = Field(..., description="Failed predictions")
    predictions: List[PredictionResponse] = Field(..., description="Individual predictions")
    processing_time_seconds: float = Field(..., description="Total processing time")

class ModelStatus(BaseModel):
    """Model status information"""
    model_name: str = Field(..., description="Model name")
    loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Path to model file")
    training_date: str = Field(..., description="When model was trained")
    accuracy_metrics: Dict[str, float] = Field(..., description="Model performance metrics")

class ModelsStatusResponse(BaseModel):
    """Response for models status endpoint"""
    available_models: List[ModelStatus] = Field(..., description="Available models")
    active_model: str = Field(..., description="Currently active model")
    last_updated: str = Field(..., description="Last model update")

class ModelEvaluationRequest(BaseModel):
    """Request for model evaluation"""
    model_name: Optional[str] = Field(None, description="Specific model to evaluate")
    test_student_ids: Optional[List[int]] = Field(None, description="Specific students to test")

class ModelEvaluationResponse(BaseModel):
    """Response for model evaluation"""
    model_name: str = Field(..., description="Evaluated model")
    accuracy: float = Field(..., description="Model accuracy")
    auc_score: float = Field(..., description="AUC score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1_score: float = Field(..., description="F1 score")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    evaluation_timestamp: str = Field(..., description="When evaluation was performed")
    total_samples: int = Field(..., description="Total samples evaluated")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    student_id: Optional[int] = Field(None, description="Student ID if applicable")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")