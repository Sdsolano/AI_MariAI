from rag.content_processors.content_indexer import get_loader, excel_loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

def generate_db(folder_path,grade:str):
# 1. Cargar todos los documentos
    OPENAI_API_KEY = "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"  # reemplaza con tu key real
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
                        doc.metadata.setdefault("source", filename)  # No sobrescribe si ya existe


                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ Error cargando {filename}: {e}")

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
    print("✅ Base vectorial creada correctamente con Excel incluido.")



def retrieve_db(grade:str,query:str,umbral:float,k:int):
    OPENAI_API_KEY = "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" 
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    print("📂 Existe la carpeta?", os.path.exists(f"mari_ai_grado_{grade}"))
    db = Chroma(
        persist_directory=f"mari_ai_grado_{grade}",
        embedding_function=embeddings
    )
    resultados_con_score = db.similarity_search_with_score(query, k=k)

    resultados_filtrados = [(doc, score) for doc, score in resultados_con_score if score <= umbral]

    if not resultados_filtrados:
        return None
    else:
        return resultados_filtrados
