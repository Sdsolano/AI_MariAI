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
    
    async def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate response using OpenAI with retrieved context"""
        try:
            # Build context from retrieved documents
            context = "\n\n".join([
                f"Documento {i+1} (Fuente: {doc['metadata'].get('source', 'desconocida')}):\n{doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Create prompt with context
            prompt = f"""Eres Mari AI, un asistente académico inteligente. Responde la pregunta del estudiante basándote únicamente en el contexto proporcionado.

Contexto disponible:
{context}

Pregunta del estudiante: {query}

Instrucciones:
- Responde de manera clara y concisa
- Basa tu respuesta únicamente en la información del contexto
- Si no encuentras información relevante en el contexto, menciona que no tienes esa información específica
- Usa un tono amigable y profesional
- Si es apropiado, menciona las fuentes de información

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
                # No relevant documents found, provide a default response
                response_data = {
                    "generated_response": f"Lo siento, no pude encontrar información específica sobre '{query}' en la base de conocimientos académica. ¿Podrías reformular tu pregunta o ser más específico?",
                    "confidence_score": 0.1,
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

# Global RAG service instance
rag_service = RAGService()
