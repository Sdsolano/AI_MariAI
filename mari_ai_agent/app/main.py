from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag.vector_stores.chroma_store import generate_db, retrieve_db
import os
from rag.retrievers.grade_retriever import preguntar_con_contexto
app = FastAPI()

class CarpetaRequest(BaseModel):
    path: str
    grade: str

@app.post("/procesar-carpeta/")
def procesar_carpeta_endpoint(req: CarpetaRequest):
    try:
        if not os.path.exists(req.path) or not os.path.isdir(req.path):
            raise ValueError("Ruta inválida o no es una carpeta.")

        resultado = generate_db(req.path,req.grade)

        return {"mensaje": "Carpeta procesada", "detalles": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




class RetrieveRequest(BaseModel):
    grade: str
    query: str
    umbral: float
    k: int

@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
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

        # Formatear para que sea JSON serializable
        return {
            "respuesta": respuesta_modelo,
            # "resultados": [
            #     {
            #         "contenido": doc.page_content,
            #         "metadatos": doc.metadata,
            #         "score": score
            #     }
            #     for doc, score in resultados
            # ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))