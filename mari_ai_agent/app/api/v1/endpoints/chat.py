# app/api/v1/endpoints/chat.py
"""
Main Chat endpoint for Mari AI Agent - Gateway to all AI functionality
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
import logging
import json
from datetime import datetime
from openai import OpenAI
import os
from enum import Enum

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

async def get_student_id_from_user_id(user_id: str) -> Optional[int]:
    """
    Convert user_id to student_id (idmatricula) by following the database relationships:
    user_id → usua_usuarios.id → estu_estudiantes.idusuario → acad_estumatricula.idestudiante → idmatricula
    """
    try:
        with db_manager.get_session() as session:
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

async def get_student_info_from_user_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get complete student information from user_id
    """
    try:
        with db_manager.get_session() as session:
            query = text("""
                SELECT 
                    em.id as idmatricula,
                    e.identificacion,
                    e.primer_nombre,
                    e.segundo_nombre,
                    e.primer_apellido,
                    e.segundo_apellido,
                    e.estrato,
                    e.fecha_nacimiento,
                    u.correo,
                    g.nombre as grado,
                    gg.nombre as grupo
                FROM usua_usuarios u
                JOIN estu_estudiantes e ON u.id = e.idusuario
                JOIN acad_estumatricula em ON e.id = em.idestudiante
                LEFT JOIN acad_gradosgrupos gg ON em.idgrados_grupos = gg.id
                LEFT JOIN acad_grados g ON gg.idgrado = g.id
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
                return {
                    "idmatricula": row.idmatricula,
                    "identificacion": row.identificacion,
                    "nombre_completo": f"{row.primer_nombre} {row.segundo_nombre or ''} {row.primer_apellido} {row.segundo_apellido or ''}".strip(),
                    "primer_nombre": row.primer_nombre,
                    "correo": row.correo,
                    "estrato": row.estrato,
                    "fecha_nacimiento": row.fecha_nacimiento,
                    "grado": row.grado,
                    "grupo": row.grupo
                }
            else:
                return None
                
    except Exception as e:
        logger.error(f"❌ Error getting student info: {e}")
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
    conversation_history: Optional[List[ChatMessage]] = []
    context: Optional[Dict[str, Any]] = {}

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ChatResponse(BaseModel):
    response: str
    user_id: str
    risk_analysis: Optional[Dict[str, Any]] = None
    function_calls_made: Optional[List[str]] = []
    conversation_history: List[ChatMessage]
    timestamp: str
    confidence: Optional[float] = None

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
2. **Sistema RAG**: Puedes buscar información relevante en la base de conocimientos académica
3. **Recomendaciones Personalizadas**: Basadas en el perfil de riesgo del estudiante

PROTOCOLO DE CONVERSACIÓN:
1. **Al inicio**: SIEMPRE obtén un análisis de riesgo del estudiante para entender su situación
2. **Para consultas**: SIEMPRE usa el sistema RAG para obtener información precisa y actualizada
3. **Enfoque**: Proporciona recursos, estrategias y apoyo personalizado según el nivel de riesgo

FUNCIONES DISPONIBLES:
- `get_risk_prediction`: Obtiene predicción de riesgo académico para un estudiante
- `search_academic_resources`: Busca información en la base de conocimientos usando RAG

INSTRUCCIONES ESPECÍFICAS:
- Si el estudiante tiene ALTO riesgo: Sé más proactivo, ofrece recursos inmediatos y estrategias de recuperación
- Si el estudiante tiene BAJO riesgo: Enfócate en mantener el rendimiento y ofrecer recursos de mejora
- SIEMPRE usa RAG para consultas específicas sobre políticas, procedimientos, recursos académicos
- Personaliza tus respuestas según el contexto del estudiante
- Sé específico y práctico en tus recomendaciones

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
            "required": ["user_id"]
        }
    },
    {
        "name": "search_academic_resources",
        "description": "Busca información relevante en la base de conocimientos académica usando RAG",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Consulta o pregunta para buscar en los recursos académicos"
                },
                "context_type": {
                    "type": "string",
                    "description": "Tipo de contexto a buscar",
                    "enum": ["academic", "administrative", "policies"]
                }
            },
            "required": ["query"]
        }
    }
]

async def execute_function_call(function_name: str, arguments: str) -> Dict[str, Any]:
    """
    Execute function calls from GPT
    """
    try:
        args = json.loads(arguments)
        logger.info(f"🔧 Executing function: {function_name} with args: {args}")
        
        if function_name == "get_risk_prediction":
            # The function should receive user_id and resolve to student_id internally
            user_id = args.get("user_id") or args.get("student_id")  # Support both for compatibility
            if not user_id:
                return {"error": "Missing user_id parameter"}
            
            # If it's already a numeric student_id (for backward compatibility), use directly
            try:
                if str(user_id).isdigit() and int(user_id) > 10000:  # Assume student IDs are > 10000
                    student_id = int(user_id)
                    logger.info(f"Using direct student_id: {student_id}")
                else:
                    # Resolve user_id to student_id (idmatricula)
                    student_id = await get_student_id_from_user_id(str(user_id))
                    if not student_id:
                        return {"error": f"No active student found for user_id: {user_id}"}
                
                # Call our prediction service
                prediction_result = model_manager.predict_risk(student_id)
                
                # Get additional student context
                student_info = await get_student_info_from_user_id(str(user_id)) if not str(user_id).isdigit() else None
                
                # Convert prediction result to dict for JSON serialization
                prediction_dict = None
                if prediction_result:
                    prediction_dict = {
                        "student_id": prediction_result.student_id,
                        "risk_level": prediction_result.risk_level.value,
                        "risk_probability": prediction_result.risk_probability,
                        "confidence": prediction_result.confidence,
                        "key_factors": [
                            {
                                "factor": kf.factor,
                                "value": kf.value,
                                "impact": kf.impact
                            } for kf in prediction_result.key_factors
                        ],
                        "recommended_actions": [
                            {
                                "action": ra.action,
                                "priority": ra.priority,
                                "description": ra.description
                            } for ra in prediction_result.recommended_actions
                        ],
                        "model_used": prediction_result.model_used,
                        "prediction_timestamp": prediction_result.prediction_timestamp
                    }
                
                return {
                    "function": "get_risk_prediction",
                    "user_id": user_id,
                    "student_id": student_id,
                    "student_info": student_info,
                    "result": prediction_dict
                }
                
            except ValueError as e:
                return {"error": f"Invalid user_id format: {user_id}"}
            
        elif function_name == "search_academic_resources":
            query = args.get("query")
            context_type = args.get("context_type", "academic")
            
            if not query:
                return {"error": "Missing query parameter"}
            
            # Call our RAG service
            rag_result = await rag_service.query(query, context_type=context_type)
            return {
                "function": "search_academic_resources",
                "query": query,
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
    Main chat endpoint - Gateway to Mari AI Agent
    
    This is the primary endpoint that integrates:
    - Risk prediction analysis
    - RAG system for academic queries  
    - GPT-4.1 conversational AI
    
    The system automatically:
    1. Analyzes student risk at conversation start
    2. Uses RAG for any academic queries
    3. Provides personalized recommendations
    """
    try:
        logger.info(f"💬 Chat request from user: {request.user_id}")
        
        # Build conversation history
        messages = [
            {"role": "system", "content": EDUCATIONAL_ASSISTANT_PROMPT}
        ]
        
        # Add conversation history
        for msg in request.conversation_history:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current user message
        messages.append({
            "role": "user", 
            "content": request.message
        })
        
        # Check if this is the start of conversation (no history)
        is_new_conversation = len(request.conversation_history) == 0
        
        # If new conversation, add context about getting risk analysis
        if is_new_conversation:
            system_context = f"""
CONTEXTO INICIAL: Este es el inicio de una nueva conversación con el estudiante (user_id: {request.user_id}).
ACCIÓN REQUERIDA: Debes INMEDIATAMENTE obtener un análisis de riesgo académico del estudiante usando la función get_risk_prediction con el user_id "{request.user_id}" antes de responder.
Esto te permitirá personalizar tu respuesta inicial basada en su situación académica actual.
            """
            messages.append({
                "role": "system",
                "content": system_context
            })
        
        # Call GPT-4.1 with function calling
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # GPT-4 Turbo (latest available)
            messages=messages,
            tools=[{"type": "function", "function": func} for func in FUNCTION_DEFINITIONS],
            tool_choice="auto",
            temperature=0.7,
            max_tokens=1500
        )
        
        assistant_message = response.choices[0].message
        function_calls_made = []
        risk_analysis = None
        
        # Handle function calls
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            function_call = tool_call.function
            function_name = function_call.name
            function_args = function_call.arguments
            
            logger.info(f"🎯 GPT requested function call: {function_name}")
            
            # Execute the function
            function_result = await execute_function_call(function_name, function_args)
            function_calls_made.append(function_name)
            
            # Store risk analysis if it was called
            if function_name == "get_risk_prediction":
                risk_analysis = function_result.get("result")
            
            # Add function result back to conversation
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or f"Llamando función {function_name}...",
                "tool_calls": assistant_message.tool_calls
            })
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(function_result, ensure_ascii=False, default=str)
            })
            
            # Get the final response from GPT with function result
            final_response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            final_content = final_response.choices[0].message.content
            
        else:
            final_content = assistant_message.content
        
        # Build updated conversation history
        updated_history = []
        for msg in request.conversation_history:
            updated_history.append(msg)
        
        # Add user message
        updated_history.append(ChatMessage(
            role=MessageRole.USER,
            content=request.message,
            timestamp=datetime.now().isoformat()
        ))
        
        # Add assistant response
        updated_history.append(ChatMessage(
            role=MessageRole.ASSISTANT,
            content=final_content,
            timestamp=datetime.now().isoformat()
        ))
        
        logger.info(f"✅ Chat response generated for user: {request.user_id}")
        
        return ChatResponse(
            response=final_content,
            user_id=request.user_id,
            risk_analysis=risk_analysis,
            function_calls_made=function_calls_made,
            conversation_history=updated_history,
            timestamp=datetime.now().isoformat(),
            confidence=0.9
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

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