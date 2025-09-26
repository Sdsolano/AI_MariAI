# app/api/v1/endpoints/chat.py
"""
Main Chat endpoint for Mari AI Agent - Gateway to all AI functionality
"""
import uuid
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
import logging
import json
from datetime import datetime
from openai import OpenAI
import os
from enum import Enum
import sqlite3
# Import our internal services
from app.services.prediction_service import ml_manager
from app.services.rag_service import RAGService
from app.db.connection import db_manager
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
model_manager = ml_manager
rag_service = RAGService()

# OpenAI Configuration
client = OpenAI(api_key=os.environ["API_KEY"] )

async def get_student_id_from_user_id(user_id: str, db_url: str) -> Optional[int]:
    """
    Convert user_id to student_id (idmatricula) by following the database relationships:
    user_id → usua_usuarios.id → estu_estudiantes.idusuario → acad_estumatricula.idestudiante → idmatricula
    """
    try:
        with db_manager.get_session_for_url(db_url) as session:
            # Query to get idmatricula from user_id
            query = text("""
                SELECT em.id as idmatricula, e.identificacion, e.primer_nombre, e.primer_apellido
                FROM usua_usuarios u
                JOIN estu_estudiantes e ON u.id = e.idusuario
                JOIN acad_estumatricula em ON e.id = em.idestudiante
                WHERE u.id = :user_id 
                AND u.estado = 'on'
                AND e.estado = 'on'
                AND em.estado = 'on'
                ORDER BY em.fecha_matricula DESC
                LIMIT 1
            """)
            
            result = session.execute(query, {"user_id": int(user_id)})
            row = result.fetchone()
            
            if row:
                logger.info(f"✅ Found student for user_id {user_id}: {row.primer_nombre} {row.primer_apellido} (idmatricula: {row.idmatricula})")
                return row.idmatricula
            else:
                logger.warning(f"⚠️ No active student found for user_id: {user_id}")
                return None
                
    except Exception as e:
        logger.error(f"❌ Error resolving user_id to student_id: {e}")
        return None

async def get_student_info_from_matricula_id(matricula_id: str, db_url: str) -> Optional[Dict[str, Any]]:
    """
    Get complete student information from matricula_id
    """
    logger.info(f"🔍 [DEBUG] Getting student info from matricula_id: {matricula_id}")
    
    try:
        with db_manager.get_session_for_url(db_url) as session:
            
            # Query directa con matricula_id - simple y directo
            query = text("""
                SELECT 
                    em.id as idmatricula,
                    e.id as student_id,
                    e.identificacion,
                    e.primer_nombre,
                    e.segundo_nombre,
                    e.primer_apellido,
                    e.segundo_apellido,
                    e.estrato,
                    e.fecha_nacimiento,
                    e.email as correo,
                    g.nombre as grado,
                    gg.nombre as grupo,
                    gg.id as grupo_id,
                    g.id as grado_id
                FROM acad_estumatricula em
                JOIN estu_estudiantes e ON em.idestudiante = e.id
                JOIN acad_gradosgrupos gg ON em.idgrados_grupos = gg.id
                JOIN acad_grados g ON gg.idgrado = g.id
                WHERE em.id = :matricula_id
                AND e.estado = 'on'
                AND em.estado = 'on'
                LIMIT 1
            """)
            
            result = session.execute(query, {"matricula_id": int(matricula_id)})
            row = result.fetchone()
            
            if row:
                student_info = {
                    "idmatricula": row.idmatricula,
                    "student_id": row.student_id,
                    "identificacion": row.identificacion,
                    "nombre_completo": f"{row.primer_nombre} {row.segundo_nombre or ''} {row.primer_apellido} {row.segundo_apellido or ''}".strip(),
                    "primer_nombre": row.primer_nombre,
                    "correo": row.correo,
                    "estrato": row.estrato,
                    "fecha_nacimiento": row.fecha_nacimiento,
                    "grado": row.grado,
                    "grupo": row.grupo,
                    "grupo_id": row.grupo_id,
                    "grado_id": row.grado_id
                }
                logger.info(f"✅ SUCCESS: Student {student_info['nombre_completo']} found - Grade: {student_info['grado']}")
                return student_info
            else:
                logger.warning(f"❌ No student info found for matricula_id {matricula_id}")
                return None
                
    except Exception as e:
        logger.error(f"❌ Error getting student info from matricula_id {matricula_id}: {e}")
        return None

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str
    db_url: str
    # AÑADIMOS ESTO para identificar la conversación
    conversation_id: Optional[str] = None 
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    user_id: str
    # AÑADIMOS ESTO para que el cliente sepa qué conversación seguir
    conversation_id: str
    risk_analysis: Optional[Dict[str, Any]] = None
    function_calls_made: Optional[List[str]] = []
    timestamp: str
    confidence: Optional[float] = None

class FunctionCall(BaseModel):
    name: str
    arguments: str







# System prompt for the educational assistant
EDUCATIONAL_ASSISTANT_PROMPT = """
Eres Mari AI, un asistente educativo inteligente especializado en ayudar a estudiantes a tener éxito académico. Tu objetivo principal es garantizar el éxito del estudiante.

PERSONALIDAD Y ESTILO:
- Empático, comprensivo y motivacional
- Profesional pero cercano
- Enfocado en soluciones prácticas
- Proactivo en ofrecer ayuda

CAPACIDADES PRINCIPALES:
1. **Análisis de Riesgo**: Puedes evaluar el riesgo académico de estudiantes usando datos históricos
2. **Sistema RAG por Grado**: Puedes buscar información específica del grado académico del estudiante
3. **Recomendaciones Personalizadas**: Basadas en el perfil de riesgo y grado del estudiante

PROTOCOLO DE CONVERSACIÓN:
1. **Al inicio**: SIEMPRE obtén un análisis de riesgo del estudiante para entender su situación, ademas saluda al estudiante por su nombre
2. **Para consultas académicas**: SIEMPRE usa el sistema RAG específico del grado del estudiante
3. **Enfoque**: Proporciona recursos, estrategias y apoyo personalizado según el nivel de riesgo y grado


FUNCIONES DISPONIBLES:
- `get_risk_prediction(user_id)`: Obtiene predicción de riesgo académico para un estudiante
- `search_academic_resources(query, user_id)`: Busca información en el contenido específico del grado del estudiante

INSTRUCCIONES CRÍTICAS:
- **SIEMPRE usa NÚMEROS como user_id** - NUNCA uses nombres de estudiantes
- **SIEMPRE incluye el user_id numérico del estudiante cuando uses search_academic_resources** - esto permite buscar en el contenido específico de su grado
- Si el estudiante tiene ALTO riesgo: Sé más proactivo, ofrece recursos inmediatos y estrategias de recuperación
- Si el estudiante tiene BAJO riesgo: Enfócate en mantener el rendimiento y ofrecer recursos de mejora
- Para consultas sobre materias, conceptos, tareas: USA search_academic_resources con el user_id NUMÉRICO para obtener contenido del grado correcto
- Personaliza tus respuestas según el contexto del estudiante y su grado académico
- Sé específico y práctico en tus recomendaciones

CONTEXTO IMPORTANTE: El sistema puede buscar en contenido académico específico por grado (6°, 7°, 8°, etc.). Siempre que un estudiante pregunte sobre contenido académico, usa search_academic_resources con su user_id para obtener información relevante a su nivel educativo.

Recuerda: Tu misión es ser el compañero de estudios inteligente que cada estudiante necesita para tener éxito.
"""

# Function definitions for OpenAI
FUNCTION_DEFINITIONS = [
    {
        "name": "get_risk_prediction",
        "description": "Obtiene la predicción de riesgo académico para un estudiante específico usando su user_id",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "User ID del estudiante para obtener predicción de riesgo (se resuelve automáticamente al student_id)"
                }
            },
            "required": ["user_id", "db_url"]
        }
    },
    {
        "name": "search_academic_resources",
        "description": "Busca información relevante en la base de conocimientos académica específica del grado del estudiante usando RAG",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta o pregunta para buscar en los recursos académicos"
                },
                "user_id": {
                    "type": "string",
                    "description": "User ID del estudiante para buscar en el contenido específico de su grado"
                },
                "context_type": {
                    "type": "string",
                    "description": "Tipo de contexto a buscar",
                    "enum": ["academic", "administrative", "policies"]
                }
            },
            "required": ["query", "user_id"]
        }
    }
]











DATABASE_FILE = "chat_history.db"

def init_db():
    """Inicializa la DB y crea la tabla si no existe."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS historial_chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idmatricula INTEGER NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_historial_chat_conv ON historial_chat (conversation_id, timestamp);
    """)
    conn.commit()
    conn.close()
    logger.info(f"✅ Base de datos local '{DATABASE_FILE}' inicializada.")

def save_message_to_db(matricula_id: int, db_url: str, role: str, content: str):
    """
    Guarda un mensaje en la base de datos SQLite.
    La db_url se guarda en la columna conversation_id.
    """
    # <<< CAMBIO: Se renombró el parámetro 'conversation_id' a 'db_url' para mayor claridad.
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO historial_chat (idmatricula, conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
        # <<< CAMBIO: Se pasa 'db_url' en la posición de 'conversation_id'.
        (matricula_id, db_url, role, content, timestamp)
    )
    conn.commit()
    conn.close()

def load_history_from_db(matricula_id: int, db_url: str, limit: int = 10) -> List[Dict[str, str]]:
    """Carga el historial de una conversación usando matricula_id y db_url."""
    # <<< CAMBIO: La función ahora requiere 'matricula_id' y 'db_url'.
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # <<< CAMBIO: El WHERE ahora filtra por ambas columnas.
    query = """
        SELECT role, content 
        FROM historial_chat 
        WHERE idmatricula = ? AND conversation_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """
    
    # <<< CAMBIO: Se pasan ambos identificadores a la consulta.
    cursor.execute(query, (matricula_id, db_url, limit))
    
    history = [dict(row) for row in reversed(cursor.fetchall())]
    conn.close()
    return history

def trim_history_in_db(matricula_id: int, db_url: str, max_size: int = 10):
    """Borra los mensajes más antiguos de una conversación si supera max_size."""
    # <<< CAMBIO: La función ahora requiere 'matricula_id' y 'db_url'.
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # <<< CAMBIO: El WHERE se actualiza en la subconsulta para usar ambos identificadores.
    query = """
        DELETE FROM historial_chat
        WHERE id IN (
            SELECT id FROM historial_chat
            WHERE idmatricula = ? AND conversation_id = ?
            ORDER BY timestamp ASC
            LIMIT MAX(0, (SELECT COUNT(*) FROM historial_chat WHERE idmatricula = ? AND conversation_id = ?) - ?)
        )
    """
    
    # <<< CAMBIO: Se pasan los parámetros correspondientes a cada '?' en la consulta.
    cursor.execute(query, (matricula_id, db_url, matricula_id, db_url, max_size))
    
    conn.commit()
    conn.close()
    logger.info(f"🧹 Historial recortado para la conversación de matricula {matricula_id}")

# def save_message_to_db(matricula_id: int, conversation_id: str, role: str, content: str):
#     """Guarda un mensaje en la base de datos SQLite."""
#     conn = sqlite3.connect(DATABASE_FILE)
#     cursor = conn.cursor()
#     timestamp = datetime.now().isoformat()
#     cursor.execute(
#         "INSERT INTO historial_chat (idmatricula, conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
#         (matricula_id, conversation_id, role, content, timestamp)
#     )
#     conn.commit()
#     conn.close()

# def load_history_from_db(conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
#     """Carga el historial de una conversación desde SQLite."""
#     conn = sqlite3.connect(DATABASE_FILE)
#     conn.row_factory = sqlite3.Row
#     cursor = conn.cursor()
#     cursor.execute(
#         "SELECT role, content FROM historial_chat WHERE conversation_id = ? ORDER BY timestamp DESC LIMIT ?",
#         (conversation_id, limit)
#     )
#     history = [dict(row) for row in reversed(cursor.fetchall())]
#     conn.close()
#     return history

# def trim_history_in_db(conversation_id: str, max_size: int = 10):
#     """Borra los mensajes más antiguos de una conversación en SQLite si supera max_size."""
#     conn = sqlite3.connect(DATABASE_FILE)
#     cursor = conn.cursor()
#     query = """
#         DELETE FROM historial_chat
#         WHERE id IN (
#             SELECT id FROM historial_chat
#             WHERE conversation_id = ?
#             ORDER BY timestamp ASC
#             LIMIT MAX(0, (SELECT COUNT(*) FROM historial_chat WHERE conversation_id = ?) - ?)
#         )
#     """
#     cursor.execute(query, (conversation_id, conversation_id, max_size))
#     conn.commit()
#     conn.close()
#     logger.info(f"🧹 Historial recortado para la conversación {conversation_id}")

























# <<< CAMBIO 1: Añadir 'db_url' a la firma de la función.
async def execute_function_call(function_name: str, arguments: str, db_url: str) -> Dict[str, Any]:
    """
    Execute function calls from GPT
    """
    try:
        args = json.loads(arguments)
        logger.info(f"🔧 Executing function: {function_name} with args: {args}")
        
        if function_name == "get_risk_prediction":
            matricula_id = args.get("user_id") or args.get("student_id")
            if not matricula_id:
                return {"error": "Missing user_id parameter"}
            
            logger.info(f"🔍 Processing risk prediction for matricula_id: {matricula_id}")
            
            student_info = None
            try:
                # <<< CAMBIO 2: Pasar 'db_url' a la función que busca información.
                student_info = await get_student_info_from_matricula_id(str(matricula_id), db_url)
                if student_info:
                    logger.info(f"✅ Found student: {student_info['nombre_completo']} - Grade: {student_info['grado']}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get student info for matricula_id {matricula_id}: {e}")
            
            logger.info(f"🎯 Calling ML model with matricula_id: {matricula_id}")
            
            # <<< CAMBIO 3: Pasar 'db_url' al servicio de predicción.
            prediction_result = model_manager.predict_risk(int(matricula_id), db_url)
            
            # ... (el resto de la lógica para construir la respuesta es igual)
            prediction_dict = None
            if prediction_result:
                prediction_dict = prediction_result.model_dump() # Forma más limpia con Pydantic v2
                
            return {
                "function": "get_risk_prediction",
                "user_id": matricula_id,
                "student_info": student_info,
                "result": prediction_dict
            }
            
        elif function_name == "search_academic_resources":
            query = args.get("query")
            context_type = args.get("context_type", "academic")
            user_id = args.get("user_id")
            
            if not query:
                return {"error": "Missing query parameter"}
            
            student_grade = None
            if user_id:
                try:
                    # <<< CAMBIO 4: Pasar 'db_url' también aquí.
                    student_info = await get_student_info_from_matricula_id(str(user_id), db_url)
                    if student_info:
                        student_grade = student_info.get("grado")
                        logger.info(f"📚 Student grade identified: {student_grade} for matricula_id {user_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not get student grade for matricula_id {user_id}: {e}")
            
            rag_result = await rag_service.query_by_grade(
            query=query, 
            db_url=db_url, # <-- Aquí se conecta todo
            grade=student_grade,
            context_type=context_type
        )
            return {
                "function": "search_academic_resources",
                "query": query,
                "student_grade": student_grade,
                "context_type": context_type,
                "result": rag_result
            }
            
        else:
            return {"error": f"Unknown function: {function_name}"}
            
    except Exception as e:
        logger.error(f"❌ Error executing function {function_name}: {e}")
        return {"error": f"Function execution failed: {str(e)}"}




@router.post("/", response_model=ChatResponse)
async def chat_with_mari(request: ChatRequest):
    """
    Endpoint de chat principal con memoria persistente en DB,
    limitada a los últimos 10 mensajes (estilo pila).
    """
    try:
        matricula_id = int(request.user_id)
        MAX_MESSAGES = 10
        
        logger.info(f"💬 Chat request from user: {matricula_id}")

        # 1. GESTIONAR ID DE CONVERSACIÓN Y CARGAR HISTORIAL
        conversation_id = request.db_url or str(uuid.uuid4())
        history_from_db = load_history_from_db(matricula_id,conversation_id, limit=MAX_MESSAGES)
        logger.info(f"🕰️ Loaded {request.db_url} messages from history for conversation {conversation_id}")
        # 2. CONSTRUIR EL PROMPT PARA OPENAI (VERSIÓN CORREGIDA)
        messages = [{"role": "system", "content": EDUCATIONAL_ASSISTANT_PROMPT}]
        messages.extend(history_from_db)
        
        # --- CAMBIO 1: VOLVEMOS A AÑADIR EL ID AL MENSAJE ---
        user_message_with_context = f"[STUDENT_ID: {matricula_id}] {request.message}"
        messages.append({"role": "user", "content": user_message_with_context})
        
        # --- CAMBIO 2: VOLVEMOS A AÑADIR EL MENSAJE DE SISTEMA FORZANDO EL ID ---
        system_context = f"""
IDENTIFICADOR PERSISTENTE DEL ESTUDIANTE: {request.user_id}
REGLA CRÍTICA: SIEMPRE usa EXACTAMENTE el número {request.user_id} como user_id en TODAS las funciones.
NO INVENTES números. USA SOLAMENTE: {request.user_id}
        """
        messages.append({"role": "system", "content": system_context})
        
        # --- CAMBIO 3: VOLVEMOS A DETECTAR SI ES UNA CONVERSACIÓN NUEVA ---
        # Si no hay historial en la DB, es el primer mensaje.
        is_new_conversation = not history_from_db
        if is_new_conversation:
            initial_context = f"""
ACCIÓN INICIAL: Este es el inicio de una nueva conversación.
Debes INMEDIATAMENTE obtener un análisis de riesgo académico usando get_risk_prediction con user_id "{request.user_id}".
            """
            messages.append({"role": "system", "content": initial_context})

        # 3. LLAMAR A OPENAI Y PROCESAR RESPUESTA (tu lógica original completa)
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            tools=[{"type": "function", "function": func} for func in FUNCTION_DEFINITIONS],
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1500
        )
        
        assistant_message = response.choices[0].message
        function_calls_made = []
        risk_analysis = None
        final_content = ""

        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            function_call = tool_call.function
            function_name, function_args = function_call.name, function_call.arguments
            
            logger.info(f"🎯 GPT requested function call: {function_name}")
            function_result = await execute_function_call(
                function_name, 
                function_args, 
                request.db_url # <-- Aquí se conecta todo
            )
            function_calls_made.append(function_name)

            if function_name == "get_risk_prediction":
                risk_analysis = function_result.get("result")

            messages.append(assistant_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(function_result, ensure_ascii=False, default=str)
            })

            final_response = client.chat.completions.create(
                model="gpt-4-1106-preview", messages=messages, temperature=0.7, max_tokens=1500
            )
            final_content = final_response.choices[0].message.content
        else:
            final_content = assistant_message.content
        
        # 4. GUARDAR Y RECORTAR EN LA DB
        save_message_to_db(matricula_id, conversation_id, "user", request.message)
        save_message_to_db(matricula_id, conversation_id, "assistant", final_content)
        trim_history_in_db(conversation_id, conversation_id,max_size=MAX_MESSAGES)
        
        logger.info(f"✅ Respuesta generada para el usuario: {matricula_id}")
        
        # 5. DEVOLVER LA RESPUESTA AL CLIENTE
        return ChatResponse(
            response=final_content,
            user_id=request.user_id,
            conversation_id=conversation_id,
            risk_analysis=risk_analysis,
            function_calls_made=function_calls_made,
            timestamp=datetime.now().isoformat(),
            confidence=0.9
        )
        
    except Exception as e:
        logger.error(f"❌ Error en el endpoint de chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar la solicitud: {str(e)}")














@router.get("/health")
async def chat_health_check():
    """
    Health check for chat system
    """
    try:
        # Check OpenAI API key
        openai_configured = bool(os.getenv('OPENAI_API_KEY'))
        
        # Check services
        prediction_available = model_manager is not None
        rag_available = rag_service is not None
        
        return {
            "status": "operational",
            "service": "Mari AI Chat Gateway",
            "components": {
                "openai_api": "configured" if openai_configured else "missing_api_key",
                "prediction_service": "available" if prediction_available else "unavailable",
                "rag_service": "available" if rag_available else "unavailable",
                "functions_available": len(FUNCTION_DEFINITIONS)
            },
            "capabilities": [
                "risk_prediction",
                "academic_rag_search",
                "conversational_ai",
                "personalized_recommendations"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in chat health check: {e}")
        raise HTTPException(status_code=500, detail="Error checking chat system health")
@router.on_event("startup")
async def startup_event():
    """Función que se ejecuta al iniciar la aplicación para crear la DB."""
    init_db()
@router.get("/config")
async def get_chat_config():
    """
    Get current chat system configuration
    """
    return {
        "model": "gpt-4-1106-preview",
        "max_tokens": 1500,
        "temperature": 0.7,
        "functions_available": [f["name"] for f in FUNCTION_DEFINITIONS],
        "auto_risk_analysis": True,
        "rag_integration": True,
        "conversation_memory": True
    }