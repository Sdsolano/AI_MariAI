# app/api/v1/api.py
"""
Main API router with all endpoints
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, academic, chat, prediction, recommendation, rag

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(academic.router, prefix="/academic", tags=["academic"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(prediction.router, prefix="/prediction", tags=["prediction"])
api_router.include_router(recommendation.router, prefix="/recommendation", tags=["recommendation"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])

# Root endpoint
@api_router.get("/")
async def root():
    """
    Mari AI Agent API Root
    """
    return {
        "message": "Mari AI Agent API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/api/v1/health/",
            "academic": "/api/v1/academic/",
            "chat": "/api/v1/chat/",
            "prediction": "/api/v1/prediction/",
            "recommendation": "/api/v1/recommendation/",
            "rag": "/api/v1/rag/",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }