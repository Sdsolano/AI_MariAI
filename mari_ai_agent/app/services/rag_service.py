# app/services/rag_service.py
"""
Real RAG Service implementation for Mari AI Agent
"""

import os
import logging
import json
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Import websearch functionality
from app.rag.retrievers.grade_retriever import buscar_con_websearch, extraer_respuesta_y_citas

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        """Initialize RAG service with real components"""
        load_dotenv()
        self.api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set API_KEY or OPENAI_API_KEY environment variable.")
        
        self.client = openai.Client(api_key=self.api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
        )
        
        # Vector stores by grade/context
        self.vector_stores: Dict[str, Chroma] = {}
        self._load_existing_stores()
    
    def _load_existing_stores(self):
        """Load existing Chroma vector stores"""
        try:
            # Look for existing vector stores
            base_path = Path.cwd()
            for store_dir in base_path.glob("mari_ai_grado_*"):
                if store_dir.is_dir():
                    grade = store_dir.name.replace("mari_ai_grado_", "")
                    try:
                        vectordb = Chroma(
                            persist_directory=str(store_dir),
                            embedding_function=self.embeddings
                        )
                        self.vector_stores[grade] = vectordb
                        logger.info(f"✅ Loaded vector store for grade: {grade}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load vector store for grade {grade}: {e}")
                        
            # If no stores found, create a default one
            if not self.vector_stores:
                logger.info("No existing vector stores found. Creating default academic store.")
                self._create_default_store()
                
        except Exception as e:
            logger.error(f"❌ Error loading vector stores: {e}")
    
    def _create_default_store(self):
        """Create a default vector store with sample academic content"""
        try:
            # Create sample documents for testing
            sample_docs = [
                Document(
                    page_content="Las políticas académicas de la universidad establecen que los estudiantes deben mantener un promedio mínimo de 3.0 para continuar en el programa.",
                    metadata={"source": "academic_policies.pdf", "section": "Políticas Académicas", "page": 1}
                ),
                Document(
                    page_content="El proceso de matrícula se realiza en línea a través del portal estudiantil. Los estudiantes deben completar la matrícula antes de las fechas límite establecidas.",
                    metadata={"source": "enrollment_guide.pdf", "section": "Matrícula", "page": 3}
                ),
                Document(
                    page_content="Los estudiantes pueden solicitar apoyo académico a través del centro de tutoría. Se ofrecen sesiones individuales y grupales.",
                    metadata={"source": "student_support.pdf", "section": "Apoyo Académico", "page": 2}
                )
            ]
            
            vectordb = Chroma.from_documents(
                documents=sample_docs,
                embedding=self.embeddings,
                persist_directory="mari_ai_grado_academic"
            )
            vectordb.persist()
            self.vector_stores["academic"] = vectordb
            logger.info("✅ Created default academic vector store")
            
        except Exception as e:
            logger.error(f"❌ Error creating default vector store: {e}")
    
    async def retrieve_documents(
        self, 
        query: str, 
        context_type: str = "academic",
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using real vector search"""
        try:
            logger.info(f"🔍 Retrieving documents for query: '{query}' in context: {context_type}")
            
            # Get appropriate vector store
            vectordb = self.vector_stores.get(context_type) or self.vector_stores.get("academic")
            if not vectordb:
                logger.warning("No vector store available, creating default")
                self._create_default_store()
                vectordb = self.vector_stores.get("academic")
            
            # Perform similarity search
            docs_with_scores = vectordb.similarity_search_with_score(
                query, 
                k=max_results * 2  # Get more results to filter by threshold
            )
            
            # Filter by similarity threshold and format results
            retrieved_docs = []
            for doc, score in docs_with_scores:
                # Convert distance to similarity (Chroma returns distance, lower is better)
                similarity_score = 1 / (1 + score)  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold and len(retrieved_docs) < max_results:
                    retrieved_docs.append({
                        "content": doc.page_content,
                        "similarity_score": round(similarity_score, 3),
                        "metadata": doc.metadata,
                        "doc_id": doc.metadata.get("source", f"doc_{len(retrieved_docs)}")
                    })
            
            logger.info(f"✅ Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            raise Exception(f"Error retrieving documents: {str(e)}")
    
    def _normalize_grade_name(self, grade: str) -> str:
        """Normalize grade names to match ChromaDB directory naming"""
        if not grade:
            return "academic"  # Default fallback
        
        # Remove extra spaces and standardize format
        normalized = grade.strip().lower()
        
        # Common grade mappings
        grade_mappings = {
            "sexto": "6°",
            "6": "6°",
            "grado 6": "6°",
            "septimo": "7°", 
            "séptimo": "7°",
            "7": "7°",
            "grado 7": "7°",
            "octavo": "8°",
            "8": "8°",
            "grado 8": "8°",
            "noveno": "9°",
            "9": "9°",
            "grado 9": "9°",
            "decimo": "10°",
            "décimo": "10°",
            "10": "10°",
            "grado 10": "10°",
            "once": "11°",
            "11": "11°",
            "grado 11": "11°"
        }
        
        # Try to find exact match first
        if normalized in grade_mappings:
            return grade_mappings[normalized]
        
        # Try to extract number pattern
        import re
        match = re.search(r'(\d{1,2})', normalized)
        if match:
            number = match.group(1)
            if number in grade_mappings:
                return grade_mappings[number]
            return f"{number}°"
        
        # If no pattern matches, try original grade name
        return grade.strip()
    
    async def retrieve_documents_by_grade(
        self, 
        query: str, 
        db_url: str,
        grade: Optional[str] = None,
        context_type: str = "academic",
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Retrieve documents from grade-specific ChromaDB store"""
        try:
            logger.info(f"🎓 Retrieving documents by grade: '{grade}' for query: '{query}'")
            logger.info("xxx"*50)
            logger.info(f" '{db_url.rsplit('/', 1)[-1]}'")
            logger.info("xxx"*50)
            # Normalize grade name
            normalized_grade = self._normalize_grade_name(grade) if grade else "academic"
            
            # Try to load grade-specific vector store
            from pathlib import Path
            from langchain.vectorstores import Chroma
            from langchain.embeddings import OpenAIEmbeddings
            
            grade_db_path = Path(f"mari_ai_grado_{normalized_grade}_{db_url.rsplit('/', 1)[-1]}")
            
            if grade_db_path.exists():
                logger.info(f"📚 Using grade-specific database: {grade_db_path}")
                embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
                vectordb = Chroma(
                    persist_directory=str(grade_db_path),
                    embedding_function=embeddings
                )
            else:
                # Fallback to general academic database or default store
                logger.warning(f"⚠️ Grade database {grade_db_path} not found, using fallback")
                fallback_path = Path("mari_ai_grado_academic")
                if fallback_path.exists():
                    embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
                    vectordb = Chroma(
                        persist_directory=str(fallback_path),
                        embedding_function=embeddings
                    )
                else:
                    # Use existing vector store from memory
                    vectordb = self.vector_stores.get(context_type) or self.vector_stores.get("academic")
                    if not vectordb:
                        logger.warning("No vector store available, creating default")
                        self._create_default_store()
                        vectordb = self.vector_stores.get("academic")
            
            # Perform similarity search
            docs_with_scores = vectordb.similarity_search_with_score(
                query, 
                k=max_results * 2  # Get more results to filter by threshold
            )
            
            # Filter by similarity threshold and format results
            retrieved_docs = []
            for doc, score in docs_with_scores:
                # Convert distance to similarity (Chroma returns distance, lower is better)
                similarity_score = 1 / (1 + score)
                
                if similarity_score >= similarity_threshold and len(retrieved_docs) < max_results:
                    retrieved_docs.append({
                        "content": doc.page_content,
                        "similarity_score": round(similarity_score, 3),
                        "metadata": {
                            **doc.metadata,
                            "grade_context": normalized_grade
                        },
                        "doc_id": doc.metadata.get("source", f"doc_{len(retrieved_docs)}")
                    })
            
            logger.info(f"✅ Retrieved {len(retrieved_docs)} documents from grade {normalized_grade}")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents by grade: {e}")
            # Fallback to general search
            logger.info("🔄 Falling back to general document retrieval")
            return await self.retrieve_documents(query, context_type, max_results, similarity_threshold)
    
    async def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]],
        websearch_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response using OpenAI with retrieved context"""
        try:
            # Build context from retrieved documents
            print("retrieved_docs"*20)
            print(retrieved_docs)
            print("retrieved_docs"*20)
            print('2'*50)
            print("websearch_context", websearch_context)
            print('2'*50)
            context = "\n\n".join([
                f"Documento {i+1} (Fuente: {doc['metadata'].get('source', 'desconocida')}):\n{doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Create prompt with context
            prompt = f"""Eres Mari AI, un asistente académico inteligente. Responde la pregunta del estudiante basándote únicamente en el contexto proporcionado.

Contexto disponible:
{f'\n\nInformación encontrada en la base de datos:\n{context}' if context else ''}, información encontrada en la web {websearch_context},

Pregunta del estudiante: {query}

Instrucciones:
- Tu función es basar tu respuesta ÚNICAMENTE en la información del contexto disponible, es decir solo vas a parafrasear sin agregar nada.
- IMPORTANTE: agrega todos los enlaces (videos, documentos, etc.) que se encuentren en el contexto.
- Organiza la respuesta de forma clara y estética. Utiliza encabezados, negritas y listas para facilitar la lectura.
- Usa un tono amigable y profesional.
- Si el contexto no contiene información relevante para responder la pregunta, menciónalo directamente.

Respuesta:"""

            # Generate response with OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres Mari AI, un asistente académico que ayuda a estudiantes universitarios."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            print("generated_text"*20)
            print(generated_text)
            print("generated_text"*20)  
            # Calculate confidence based on number and quality of retrieved docs
            confidence = min(0.9, 0.5 + (len(retrieved_docs) * 0.1) + 
                           (sum(doc['similarity_score'] for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0) * 0.3)
            
            return {
                "generated_response": generated_text,
                "confidence_score": round(confidence, 2),
                "context_used": [doc['content'] for doc in retrieved_docs]
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise Exception(f"Error generating response: {str(e)}")
    
    async def add_document(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: str = "academic"
    ) -> Dict[str, Any]:
        """Add document to vector store"""
        try:
            logger.info(f"📄 Adding document to {doc_type} knowledge base")
            
            # Split document into chunks
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[metadata or {}]
            )
            
            # Get or create vector store for doc_type
            if doc_type not in self.vector_stores:
                # Create new vector store
                vectordb = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=f"mari_ai_grado_{doc_type}"
                )
                self.vector_stores[doc_type] = vectordb
            else:
                # Add to existing vector store
                vectordb = self.vector_stores[doc_type]
                vectordb.add_documents(docs)
            
            # Persist changes
            vectordb.persist()
            
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return {
                "doc_id": doc_id,
                "status": "processed",
                "chunks_created": len(docs),
                "content_length": len(content),
                "doc_type": doc_type,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error adding document: {e}")
            raise Exception(f"Error adding document: {str(e)}")
    
    async def query(
        self, 
        query: str, 
        context_type: str = "academic",
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve documents and generate response
        This is the main method used by the chat endpoint
        """
        try:
            logger.info(f"🤖 Processing RAG query: '{query}' in context: {context_type}")
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = await self.retrieve_documents(
                query=query,
                context_type=context_type,
                max_results=max_results,
                similarity_threshold=similarity_threshold
            )
            
            # Step 2: Generate response based on retrieved context
            if retrieved_docs:
                response_data = await self.generate_response(query, retrieved_docs)
            else:
                # No relevant documents found, use websearch fallback
                logger.warning("No relevant documents found, using websearch fallback")
                
                try:
                    search_prompt = f"{query}. Importante, Busca únicamente información en Google académico, evita sitios como wikipedia, foros, etc."
                    websearch_context = buscar_con_websearch(search_prompt)
                    
                    response_data = {
                        "generated_response": f"No encontré información específica en mi base de conocimientos sobre '{query}', pero he buscado información actualizada en internet:\n\n{websearch_context}",
                        "confidence_score": 0.6,  # Lower confidence since it's from websearch
                        "context_used": [websearch_context],
                        "source": "websearch"
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Websearch fallback failed: {e}")
                    response_data = {
                        "generated_response": f"Lo siento, no encontré información específica sobre '{query}' ni en mi base de conocimientos ni en la búsqueda web. ¿Podrías reformular tu pregunta o ser más específico?",
                        "confidence_score": 0.0,
                        "context_used": []
                    }
            
            # Compile complete response
            return {
                "query": query,
                "retrieved_documents": retrieved_docs,
                "generated_response": response_data["generated_response"],
                "context_used": response_data["context_used"],
                "confidence_score": response_data["confidence_score"],
                "timestamp": datetime.now().isoformat(),
                "context_type": context_type
            }
            
        except Exception as e:
            logger.error(f"❌ Error in RAG query: {e}")
            # Return error response in expected format
            return {
                "query": query,
                "retrieved_documents": [],
                "generated_response": f"Disculpa, ocurrió un error al procesar tu consulta: {str(e)}",
                "context_used": [],
                "confidence_score": 0.0,
                "timestamp": datetime.now().isoformat(),
                "context_type": context_type,
                "error": str(e)
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get real statistics about the RAG system"""
        try:
            total_docs = 0
            doc_types = {}
            
            for context_type, vectordb in self.vector_stores.items():
                try:
                    # Get collection info
                    collection = vectordb._collection
                    count = collection.count()
                    total_docs += count
                    doc_types[context_type] = count
                except Exception as e:
                    logger.warning(f"Could not get stats for {context_type}: {e}")
                    doc_types[context_type] = 0
            
            return {
                "total_documents": total_docs,
                "document_types": doc_types,
                "vector_stores_loaded": len(self.vector_stores),
                "embedding_model": "text-embedding-ada-002",
                "vector_dimension": 1536,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {
                "total_documents": 0,
                "document_types": {},
                "error": str(e)
            }

    async def query_by_grade(
        self, 
        query: str, 
        db_url: str,
        grade: Optional[str] = None,
        context_type: str = "academic"
    ) -> Dict[str, Any]:
        """Complete RAG query pipeline with grade-aware search"""
        try:
            logger.info(f"🎓 Starting grade-aware RAG query: '{query}' for grade: {grade}")
            
            # Step 1: Retrieve documents from grade-specific database
            retrieved_docs = await self.retrieve_documents_by_grade(
                query=query,
                grade=grade,
                db_url=db_url,
                context_type=context_type,
                max_results=5,
                similarity_threshold=0.7
            )
            
            
            search_prompt = f"{query}. Tu única función es regresar en tu respuesta enlaces relevantes como videos o documentos, evita sitios como wikipedia, foros, etc. Además debes dar información relevante sobre el tema solicitado. Es para un estudiante de {grade}."
            websearch_context = buscar_con_websearch(search_prompt)
            
            # Step 2: Generate response based on retrieved documents
            response_data = await self.generate_response(query, retrieved_docs,websearch_context)
            
            #response_data["generated_response"] += f"\n\nAdemás, he buscado información actualizada en internet:\n\n{websearch_context}"
            # Step 3: Compile complete response with grade context
            return {
                "query": query,
                "retrieved_documents": retrieved_docs,
                "generated_response": response_data["generated_response"],
                "context_used": response_data["context_used"],
                "confidence_score": response_data["confidence_score"],
                "grade_context": self._normalize_grade_name(grade) if grade else "academic",
                "context_type": context_type,
                "timestamp": datetime.now().isoformat(),
                "sources": [
                    {
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "similarity": doc["similarity_score"],
                        "source": doc["metadata"].get("source", "unknown"),
                        "grade": doc["metadata"].get("grade_context", "academic")
                    } for doc in retrieved_docs
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Error in grade-aware RAG query: {e}")
            return {
                "query": query,
                "retrieved_documents": [],
                "generated_response": f"Disculpa, ocurrió un error al procesar tu consulta: {str(e)}",
                "context_used": [],
                "confidence_score": 0.0,
                "grade_context": self._normalize_grade_name(grade) if grade else "academic",
                "context_type": context_type,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

# Global RAG service instance
rag_service = RAGService()
