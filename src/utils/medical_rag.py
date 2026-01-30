"""
Medical RAG System - Knowledge Base for AI Agents
Grounds AI responses in real medical literature
"""

import os
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import ollama
from chromadb.api.types import Embeddings, EmbeddingFunction


class CustomOllamaEmbeddingFunction(EmbeddingFunction):
    """
    Custom Ollama embedding function that uses the correct API (embeddings() not embed())
    Fixes compatibility issue between ChromaDB 1.4.1 and Ollama client 0.1.6
    """
    def __init__(self, url: str = "http://localhost:11434", model_name: str = "llama3.2:3b"):
        self.client = ollama.Client(host=url)
        self.model_name = model_name
    
    def __call__(self, input: List[str]) -> Embeddings:
        """Generate embeddings using Ollama's embeddings() method"""
        embeddings = []
        for text in input:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response['embedding'])
        return embeddings


class MedicalKnowledgeBase:
    """
    RAG system for medical knowledge
    Prevents AI from making up medical facts
    """
    
    def __init__(self, knowledge_dir: str = "knowledge_base"):
        self.knowledge_dir = knowledge_dir
        
        # Initialize ChromaDB (local vector store)
        self.client = chromadb.Client()
        
        # Use Ollama for embeddings (runs locally)
        # Use custom embedding function to fix API compatibility issue
        # ChromaDB's OllamaEmbeddingFunction uses client.embed() but Ollama client has embeddings()
        self.embedding_function = CustomOllamaEmbeddingFunction(
            url="http://localhost:11434",
            model_name="llama3.2:3b"
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            embedding_function=self.embedding_function
        )
        
        print("✅ Medical Knowledge Base initialized")
    
    def load_documents(self) -> None:
        """
        Load all medical documents into vector database
        """
        print("\n📚 Loading medical knowledge base...")
        
        documents = []
        metadatas = []
        ids = []
        
        doc_id = 0
        
        # Walk through knowledge base directory
        for root, dirs, files in os.walk(self.knowledge_dir):
            for file in files:
                if file.endswith('.txt'):
                    filepath = os.path.join(root, file)
                    
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Split into chunks (for better retrieval)
                    chunks = self._chunk_document(content, chunk_size=500)
                    
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadatas.append({
                            'source': file,
                            'category': os.path.basename(root),
                            'chunk': i
                        })
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1
                    
                    print(f"   ✓ Loaded: {file} ({len(chunks)} chunks)")
        
        # Add to vector store
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"\n✅ Loaded {len(documents)} document chunks into knowledge base")
        else:
            print("⚠️  No documents found in knowledge_base/")
    
    def _chunk_document(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split document into chunks for better retrieval
        """
        # Simple chunking by paragraphs
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def search(self, query: str, n_results: int = 3) -> List[Dict]:
        """
        Search knowledge base for relevant medical information
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        relevant_docs = []
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                relevant_docs.append({
                    'content': doc,
                    'source': results['metadatas'][0][i]['source'],
                    'category': results['metadatas'][0][i]['category']
                })
        
        return relevant_docs
    
    def get_context_for_query(self, query: str, n_results: int = 3) -> str:
        """
        Get formatted context from knowledge base
        """
        docs = self.search(query, n_results)
        
        if not docs:
            return "No relevant medical knowledge found."
        
        context = "MEDICAL KNOWLEDGE BASE:\n\n"
        
        for i, doc in enumerate(docs, 1):
            context += f"Document {i} (Source: {doc['source']}):\n"
            context += f"{doc['content']}\n\n"
        
        return context


# Test if run directly
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Medical RAG System")
    print("=" * 60)
    
    # Initialize knowledge base
    kb = MedicalKnowledgeBase()
    
    # Load documents
    kb.load_documents()
    
    # Test queries
    print("\n" + "=" * 60)
    print("TEST 1: Query about cholesterol")
    print("=" * 60)
    
    query1 = "Should I keep or remove cholesterol outliers above 400?"
    docs1 = kb.search(query1, n_results=2)
    
    print(f"\nQuery: {query1}")
    print(f"Found {len(docs1)} relevant documents:\n")
    
    for doc in docs1:
        print(f"Source: {doc['source']}")
        print(f"Content preview: {doc['content'][:200]}...")
        print()
    
    # Test 2
    print("=" * 60)
    print("TEST 2: Query about gender bias")
    print("=" * 60)
    
    query2 = "Why does gender imbalance matter for heart disease models?"
    docs2 = kb.search(query2, n_results=2)
    
    print(f"\nQuery: {query2}")
    print(f"Found {len(docs2)} relevant documents:\n")
    
    for doc in docs2:
        print(f"Source: {doc['source']}")
        print(f"Content preview: {doc['content'][:200]}...")
        print()
    
    print("\n✅ Medical RAG system test complete!")
