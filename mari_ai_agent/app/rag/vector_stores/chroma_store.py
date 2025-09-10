from app.rag.content_processors.content_indexer import get_loader, excel_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from collections import defaultdict
import psycopg2
def obtener_cursos():
    conn = psycopg2.connect(
       dbname=os.environ["ACADEMIC_DB_NAME"] ,
        user=os.environ["ACADEMIC_DB_USER"],
        host=os.environ["ACADEMIC_DB_HOST"],
        port=os.environ["ACADEMIC_DB_PORT"]
    )

    cur = conn.cursor()

    query = """
    SELECT 
        r.file_path,
        g.nombre AS grado_grupo_nombre
    FROM centro_acad_recursos r
    JOIN acad_gradosasignaturas a
        ON r.idgrados_asignatura = a.id
    JOIN acad_gradosgrupos g
        ON a.idgrados_grupos = g.id;
    """

    cur.execute(query)
    rows = cur.fetchall()
    # Convertir a diccionario {file_path: grado}
    result = {os.path.join(os.environ["FOLDER_PATH"],row[0]): row[1] for row in rows}
    print(result)


    cur.close()
    conn.close()
    return result
def take_key():
    load_dotenv()
    api_key = os.environ["API_KEY"] 
    return api_key


def generate_db_from_dict(file_dict):
    """
    file_dict: diccionario {file_path: grado}
    Genera una base Chroma por cada grado, con todos los archivos de ese grado.
    """
    OPENAI_API_KEY = take_key()  # tu función para obtener la API key
    all_docs_per_grade = defaultdict(list)

    # Agrupar archivos por grado
    for file_path, grade in file_dict.items():
        if not os.path.exists(file_path):
            print(f"Archivo no encontrado: {file_path}")
            continue
        loader_type = get_loader(file_path)
        try:
            if loader_type == "custom_excel_loader":
                docs = excel_loader(file_path)
            else:
                docs = loader_type.load()
                for doc in docs:
                    doc.metadata.setdefault("source", os.path.basename(file_path))
            all_docs_per_grade[grade].extend(docs)
        except Exception as e:
            print(f"Error cargando {file_path}: {e}")

    # Para cada grado, generar la base vectorial
    for grade, docs in all_docs_per_grade.items():
        if not docs:
            print(f"No hay documentos para el grado {grade}, se omite.")
            continue

        # Dividir en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)

        # Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Crear base Chroma
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=f"mari_ai_grado_{grade}"
        )
        vectordb.persist()
        print(f"Base vectorial creada correctamente para grado {grade}.")








def generate_db(folder_path,grade:str):
# 1. Cargar todos los documentos
    OPENAI_API_KEY = take_key() 
    all_documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".pdf", ".docx", ".pptx", ".txt", ".csv", ".xlsx", ".xlsm")):
            full_path = os.path.join(folder_path, filename)
            loader_type = get_loader(full_path)
            try:
                if loader_type == "custom_excel_loader":
                    docs = excel_loader(full_path)
                else:
                    docs = loader_type.load()
                    for doc in docs:
                        doc.metadata.setdefault("source", filename) 
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error cargando {filename}: {e}")

    # 2. Dividir en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)

    # 3. Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # 4. Crear base Chroma
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=f"mari_ai_grado_{grade}"
    )

    vectordb.persist()
    print("Base vectorial creada correctamente.")



def retrieve_db(grade:str,query:str,umbral:float,k:int):
    OPENAI_API_KEY = take_key()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print("📂 Existe la carpeta?", os.path.exists(f"mari_ai_grado_{grade}"))
    db = Chroma(
        persist_directory=f"mari_ai_grado_{grade}",
        embedding_function=embeddings
    )
    resultados_con_score = db.similarity_search_with_score(query, k=k)
    
    resultados_filtrados = [(doc, score) for doc, score in resultados_con_score if score <= umbral]
    print("Resultados encontrados:", len(resultados_filtrados))
    if not resultados_filtrados:
        print("No se encontraron resultados con ese umbral.")
        return None
    else:
        return resultados_filtrados
