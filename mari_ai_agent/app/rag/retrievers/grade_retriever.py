import openai
from typing import List
from langchain.schema import Document  # o reemplaza si es otra clase

 # Reemplaza por tu clave real o ponla como variable de entorno
client = openai.Client(api_key="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")
def construir_contexto_desde_docs(documentos: List[Document]) -> str:
    """
    Construye un contexto textual a partir de una lista de documentos.
    """
    contexto = "\n\n".join(
        f"(Fuente: {doc.metadata.get('source', 'desconocida')}, página {doc.metadata.get('page', '?')}):\n{doc.page_content}"
        for doc in documentos
    )
    return contexto

def preguntar_con_contexto(documentos: List[Document], pregunta: str, modelo: str = "gpt-3.5-turbo") -> str:
    """
    Usa OpenAI para responder una pregunta usando el contexto dado, con mensajes estructurados.
    La respuesta incluirá las fuentes y páginas como referencias.
    """
    contexto = construir_contexto_desde_docs(documentos)
    
    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente experto. Responde solo con la información proporcionada en el contexto.\n"
                "Incluye referencias claras a las fuentes y páginas cuando sea posible usando el formato: "
                "'(Fuente: [source], página [page])'. Si no hay suficiente información, responde 'No tengo suficiente información'."
            )
        },
        {
            "role": "user",
            "content": f"""Contexto:
{contexto}

Pregunta:
{pregunta}"""
        }
    ]
    
    respuesta = client.chat.completions.create(
        model=modelo,
        messages=messages,
        temperature=0,
        max_tokens=500
    )
    
    return respuesta.choices[0].message.content
