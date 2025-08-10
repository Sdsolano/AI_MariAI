# mari_ai_agent/app/api/v1/endpoints/chat.py
"""
Chat endpoints for the Mari AI Agent
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
async def chat_info():
    """
    Chat service information endpoint
    """
    return {
        "status": "available",
        "service": "Chat Service",
        "description": "Chat endpoints will be implemented here"
    }