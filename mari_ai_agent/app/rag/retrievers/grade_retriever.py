import openai
from typing import List
from langchain.schema import Document
import os  # o reemplaza si es otra clase
from dotenv import load_dotenv

# Carga las variables ANTES de usarlas
load_dotenv()
api_key = os.environ["API_KEY"] 
 # Reemplaza por tu clave real o ponla como variable de entorno
client = openai.Client(api_key=api_key)
def construir_contexto_desde_docs(documentos: List[Document]) -> str:
    """
    Construye un contexto textual a partir de una lista de documentos.
    """
    contexto = "\n\n".join(
        f"(Fuente: {doc.metadata.get('source', 'desconocida')}, página {doc.metadata.get('page', '?')}):\n{doc.page_content}"
        for doc in documentos
    )
    return contexto

def extraer_respuesta_y_citas(response):
    texto = ""
    citas = []

    for item in response.output:
        if item.type == "message":
            for contenido in item.content:
                if contenido.type == "output_text":
                    texto = contenido.text
                    
                    # Extraer citas
                    if hasattr(contenido, "annotations"):
                        for ann in contenido.annotations:
                            if ann.type == "url_citation":
                                citas.append({
                                    "titulo": ann.title,
                                    "url": ann.url
                                })

    return texto, citas

def buscar_con_websearch(prompt: str, location: dict = None) -> str:
    tools = [{
        "type": "web_search_preview"
    }]

    if location:
        tools[0]["user_location"] = {
            "type": "approximate",
            **location
        }

    response = client.responses.create(
        model="gpt-4.1",  # También puedes usar "gpt-4o", "o4-mini"
        input=prompt,
        tools=tools,
        tool_choice={ "type": "web_search_preview" }  # 👈 Forzar uso de la herramienta
    )
    texto, citas = extraer_respuesta_y_citas(response)

# Mostrar resultado
    print("🧾 Respuesta:")
    print(texto[:1000] + "...\n")  # solo los primeros 1000 caracteres

    print("🔗 Fuentes citadas:")
    for i, c in enumerate(citas, 1):
        print(f"{i}. {c['titulo']} — {c['url']}")
    return texto


def preguntar_con_contexto(documentos: List[Document], pregunta: str, modelo: str = "gpt-3.5-turbo") -> str:
    """
    Usa OpenAI para responder una pregunta usando el contexto dado, con mensajes estructurados.
    La respuesta incluirá las fuentes y páginas como referencias.
    """
    if documentos is None or len(documentos) == 0:
        print("Contexto obtenido de búsqueda web.")
        contexto= buscar_con_websearch(pregunta+". Importante, Busca unicamente informacion en google academico, evita sitios como wikipedia, foros, etc.")
        
    else:
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
