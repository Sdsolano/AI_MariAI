# mari_ai_agent/app/main.py - ACTUALIZADO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from app.core.config.settings import settings
from app.api.v1.api import api_router

# Importar funcionalidades RAG existentes (mantener compatibilidad)
try:
    from app.rag.vector_stores.chroma_store import generate_db, retrieve_db
    from app.rag.retrievers.grade_retriever import preguntar_con_contexto
    rag_available = True
except ImportError:
    rag_available = False

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="Sistema integral de agentes de IA para educación"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers de API v1
app.include_router(api_router, prefix=settings.API_V1_STR)

# ========== MANTENER ENDPOINTS RAG EXISTENTES ==========

class CarpetaRequest(BaseModel):
    path: str
    grade: str

class RetrieveRequest(BaseModel):
    grade: str
    query: str
    umbral: float
    k: int

@app.post("/procesar-carpeta/")
def procesar_carpeta_endpoint(req: CarpetaRequest):
    """Endpoint existente para RAG - mantener compatibilidad"""
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG functionality not available")
    
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
    if not rag_available:
        raise HTTPException(status_code=503, detail="RAG functionality not available")
    
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

# ========== NUEVO ENDPOINT DE ESTADO ==========

@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "service": "Mari AI - Academic Intelligence Agent",
        "version": settings.PROJECT_VERSION,
        "status": "active",
        "features": {
            "rag_system": "✅ Active" if rag_available else "❌ Unavailable",
            "academic_data": "✅ Active", 
            "prediction_engine": "🔄 In Development",
            "recommendation_engine": "🔄 In Development"
        },
        "api_docs": f"{settings.API_V1_STR}/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)