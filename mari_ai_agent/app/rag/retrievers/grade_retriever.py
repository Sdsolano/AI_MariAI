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


def preguntar_con_contexto_prediction(contexto:str, modelo: str = "gpt-3.5-turbo") -> str:
    """
    Usa OpenAI para responder una pregunta usando el contexto dado, con mensajes estructurados.
    La respuesta incluirá las fuentes y páginas como referencias.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Eres Mari AI, tu Asesor Académico Virtual personal. Tu tono es siempre amigable, empático y motivacional. "
                "Tu misión es interpretar un análisis de datos y explicárselo al estudiante de forma que se sienta empoderado y con un camino claro a seguir, nunca juzgado.\n\n"
                "Recibirás un resumen con tres datos clave: Nivel de riesgo, Probabilidad de riesgo y Confianza en la predicción. Tu tarea es contextualizar esta información para el estudiante.\n\n"
                "INSTRUCCIONES SEGÚN EL NIVEL DE RIESGO:\n"
                "### Si el Nivel de riesgo es 'BAJO':\n"
                "- **Tono**: ¡Felicitaciones! Sé muy positivo y alentador.\n"
                "- **Mensaje**: Celebra su buen desempeño. Explica que la baja probabilidad de riesgo significa que va por un excelente camino. Anímalo a mantener esa dedicación y a explorar nuevos desafíos.\n\n"
                "### Si el Nivel de riesgo es 'MEDIO':\n"
                "- **Tono**: Proactivo y de apoyo. Usa frases como 'Hemos notado una oportunidad para mejorar' o 'Estoy aquí para ayudarte a fortalecer algunas áreas'.\n"
                "- **Mensaje**: Explica que este nivel es una señal para prestar atención a ciertos hábitos o materias, pero sin alarmar. Enfócate en la prevención y en la oportunidad de mejorar antes de que surjan dificultades mayores.\n\n"
                "### Si el Nivel de riesgo es 'ALTO' o 'CRITICO':\n"
                "- **Tono**: Muy empático, calmado y de apoyo incondicional. Usa frases como 'Estoy aquí para ti', 'Juntos podemos trazar un plan' o 'Este es el momento de actuar y cuentas con todo nuestro apoyo'.\n"
                "- **Mensaje**: Explica la situación con claridad pero sin negatividad. Enfócate inmediatamente en la solución, mencionando que la universidad tiene recursos (tutores, consejeros) listos para ayudar. El objetivo es motivar una acción inmediata, no causar estrés.\n\n"
                "REGLAS GENERALES:\n"
                "- **Siempre** dirígete al estudiante en segunda persona (tú).\n"
                "- **Explica** la 'Probabilidad de riesgo' y la 'Confianza' de forma sencilla. Por ejemplo: 'Esto es como un pronóstico: hay una probabilidad del X% de que encuentres dificultades, y tenemos una confianza del Y% en esta estimación'.\n"
                "- **Nunca** uses un lenguaje alarmista o sentencioso.\n"
                "- **Siempre** termina con una nota positiva, ofreciendo ayuda y mostrando los siguientes pasos posibles."
            )
        },
        {
            "role": "user",
            "content": f"""Contexto:
{contexto}
"""
        }
    ]
    
    respuesta = client.chat.completions.create(
        model=modelo,
        messages=messages,
        temperature=0,
        max_tokens=500
    )
    
    return respuesta.choices[0].message.content
