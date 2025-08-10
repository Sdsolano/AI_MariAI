# mari_ai_agent/app/api/v1/api.py - ACTUALIZADO
from fastapi import APIRouter
from app.api.v1.endpoints import health, academic

api_router = APIRouter()

# Incluir routers principales
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(academic.router, prefix="/academic", tags=["academic"])

# Incluir routers opcionales (si existen y tienen routers válidos)
try:
    from app.api.v1.endpoints import chat
    if hasattr(chat, 'router'):
        api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
except ImportError:
    pass

try:
    from app.api.v1.endpoints import prediction
    if hasattr(prediction, 'router'):
        api_router.include_router(prediction.router, prefix="/prediction", tags=["prediction"])
except ImportError:
    pass

try:
    from app.api.v1.endpoints import recommendation
    if hasattr(recommendation, 'router'):
        api_router.include_router(recommendation.router, prefix="/recommendation", tags=["recommendation"])
except ImportError:
    pass
