# mari_ai_agent/app/api/v1/endpoints/recommendation.py
"""
Recommendation endpoints for the Mari AI Agent
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def recommendation_info():
    """
    Recommendation service information endpoint
    """
    return {
        "status": "available",
        "service": "Recommendation Service",
        "description": "Recommendation endpoints will be implemented here"
    }