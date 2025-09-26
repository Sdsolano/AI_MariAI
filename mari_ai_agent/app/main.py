# app/main.py
"""
Mari AI Agent - Main FastAPI Application with ML Prediction Support
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.api.v1.api import api_router
from app.db.connection import db_manager
from app.rag.vector_stores.chroma_store import generate_db, retrieve_db,generate_db_from_dict,obtener_cursos
from app.rag.retrievers.grade_retriever import preguntar_con_contexto

from app.services.prediction_service import ml_manager

# Pydantic models for RAG endpoints
class CarpetaRequest(BaseModel):
    path: str
    grade: str

class RetrieveRequest(BaseModel):
    grade: str
    query: str
    umbral: float
    k: int
class ProcessingRequest(BaseModel):
    db_url: str
    
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Mari AI Agent",
    description="Agente Integral de Inteligencia Artificial para plataforma educativa Mari AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("MARI AI AGENT - STARTING UP")
    logger.info("="*50)
    
    # Test database connection
    logger.info("🔌 Testing database connection...")
    try:
        if db_manager.test_connection():
            logger.info("Database connection successful")
        else:
            logger.error("Database connection failed")
            raise Exception("Database connection failed")
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Don't fail startup, but log the error
    
    # Load ML models
    logger.info(" Loading ML models...")
    try:
        success = ml_manager.load_models()
        if success:
            logger.info("ML models loaded successfully")
            logger.info(f"   Available models: {list(ml_manager.models.keys())}")
            logger.info(f"   Active model: {ml_manager.active_model}")
        else:
            logger.warning(" Some ML models failed to load")
    except Exception as e:
        logger.error(f" ML models loading error: {e}")
        # Don't fail startup, but log the error
    
    logger.info(" MARI AI AGENT - STARTUP COMPLETE")
    logger.info("="*50)
    logger.info(" API Documentation: http://localhost:8000/docs")
    logger.info(" Alternative docs: http://localhost:8000/redoc")
    logger.info(" Health check: http://localhost:8000/api/v1/health/")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info(" MARI AI AGENT - SHUTTING DOWN")
    
    # Cleanup database connections
    try:
        # db_manager.close_connections()  # Commented out as method doesn't exist
        logger.info(" Database connections closed")
    except Exception as e:
        logger.error(f" Error closing database connections: {e}")
    
    logger.info(" MARI AI AGENT - SHUTDOWN COMPLETE")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mari AI Agent API",
        "version": "1.0.0",
        "status": "operational",
        "features": {
            "academic_risk_prediction": " Operational",
            "personalized_recommendations": " In Development", 
            "chat_assistant": " Operational",
            "automated_alerts": " In Development"
        },
        "endpoints": {
            "health": "/api/v1/health/",
            "academic": "/api/v1/academic/", 
            "chat": "/api/v1/chat/",
            "prediction": "/api/v1/prediction/",
            "rag_procesar": "/procesar-carpeta/",
            "rag_retrieve": "/retrieve",
            "docs": "/docs"
        }
    }
@app.post("/procesar-carpeta-dict/")
def procesar_carpeta_endpoint(request: ProcessingRequest): # Se recibe el request
    """
    Endpoint dinámico que procesa los recursos de una base de datos específica
    y genera los vector stores correspondientes para ese tenant.
    """
    # Se usan los datos del request para llamar a las funciones
    diccionario_archivos = obtener_cursos(
        db_url=request.db_url
    )
    
    generate_db_from_dict(
        file_dict=diccionario_archivos, 
        db_url=request.db_url
    )
@app.get("/status")
async def system_status():
    """System status endpoint"""
    try:
        # Check database
        db_status = "healthy" if db_manager.test_connection() else "unhealthy"
        
        # Check ML models
        ml_status = "healthy" if ml_manager.models else "unhealthy"
        
        # Overall status
        overall_status = "healthy" if db_status == "healthy" and ml_status == "healthy" else "degraded"
        
        return {
            "overall_status": overall_status,
            "components": {
                "database": db_status,
                "ml_models": ml_status,
                "api": "healthy"
            },
            "ml_models": {
                "loaded_models": list(ml_manager.models.keys()),
                "active_model": ml_manager.active_model,
                "total_models": len(ml_manager.models)
            },
            "timestamp": "2025-08-10T10:00:00Z"
        }
        
    except Exception as e:
        logger.error(f" Error checking system status: {e}")
        raise HTTPException(status_code=500, detail="Error checking system status")

@app.post("/procesar-carpeta/")
def procesar_carpeta_endpoint(req: CarpetaRequest):
    """Endpoint existente para RAG - mantener compatibilidad"""
    try:
        if not os.path.exists(req.path) or not os.path.isdir(req.path):
            raise ValueError("Ruta inválida o no es una carpeta.")

        resultado = generate_db(req.path, req.grade)
        return {"mensaje": "Carpeta procesada", "detalles": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    """Endpoint existente para RAG - mantener compatibilidad"""
    try:
        resultados = retrieve_db(
            grade=req.grade,
            query=req.query,
            umbral=req.umbral,
            k=req.k
        )

        if not resultados:
            documentos = []
        else:
            documentos = [doc for doc, _ in resultados]

        respuesta_modelo = preguntar_con_contexto(documentos, req.query)

        return {"respuesta": respuesta_modelo}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
