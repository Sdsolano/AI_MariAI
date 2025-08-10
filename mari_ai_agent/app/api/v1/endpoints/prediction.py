# mari_ai_agent/app/api/v1/endpoints/prediction.py
"""
Prediction endpoints for the Mari AI Agent
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def prediction_info():
    """
    Prediction service information endpoint
    """
    return {
        "status": "available",
        "service": "Prediction Service",
        "description": "Prediction endpoints will be implemented here"
    }