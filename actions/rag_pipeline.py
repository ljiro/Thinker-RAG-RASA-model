import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
import torch
from typing import List, Dict, Any
import pickle
import json

class FAISSRAGPipeline:
    def __init__(self, knowledge_base_path: str = "knowledge_base/documents/"):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = None
        self.llm = None
        self.tokenizer = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        # FAISS configuration
        self.index_path = "vector_db/faiss_index.index"
        self.metadata_path = "vector_db/faiss_metadata.pkl"
        
        self._setup_models()
        self._setup_faiss_index()
    
    def _setup_models(self):
        """Initialize models optimized for low VRAM usage"""
        print("Loading embedding model...")
        
        # Lightweight embedding model (~400MB) - compatible with FAISS
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        print("Loading LLM...")
        
        # Quantized model configuration for low VRAM
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Using a small, efficient model that fits in 3-4GB VRAM
        model_name = "microsoft/DialoGPT-medium"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        # Text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def _setup_faiss_index(self):
        """Initialize or load FAISS index"""
        print("Setting up FAISS index...")
        
        os.makedirs("vector_db", exist_ok=True)
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            # Load existing index
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            print(f"Loaded FAISS index with {len(self.documents)} documents")
        else:
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
            self.documents = []
            self.metadata = []
            print("Created new FAISS index")
    
    def _save_faiss_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        print(f"Saved FAISS index with {len(self.documents)} documents")
    
    def chunk_text(self, text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def add_documents(self, file_path: str):
        """Add documents to the FAISS knowledge base"""
        import PyPDF2
        import docx
        
        print(f"Processing document: {file_path}")
        
        text = ""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Generate embeddings and add to FAISS
        new_documents = []
        new_metadata = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:  # Only add meaningful chunks
                # Generate embedding
                embedding = self.embedding_model.encode([chunk])
                embedding = embedding.astype('float32')
                
                # Normalize for cosine similarity
                faiss.normalize_L2(embedding)
                
                # Add to FAISS index
                self.index.add(embedding)
                
                # Store document and metadata
                new_documents.append(chunk)
                new_metadata.append({
                    "source": file_path,
                    "chunk_id": i,
                    "original_length": len(text)
                })
        
        if new_documents:
            self.documents.extend(new_documents)
            self.metadata.extend(new_metadata)
            self._save_faiss_index()
            print(f"Added {len(new_documents)} chunks from {file_path}")
    
    def search_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for similar documents using FAISS"""
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, min(n_results, len(self.documents)))
        
        formatted_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure valid index
                formatted_results.append({
                    'content': self.documents[idx],
                    'source': self.metadata[idx]['source'],
                    'chunk_id': self.metadata[idx]['chunk_id'],
                    'similarity_score': float(distances[0][i])  # Cosine similarity
                })
        
        # Sort by similarity score (highest first)
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return formatted_results
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using the LLM with RAG context"""
        if not context:
            return "I couldn't find relevant information in my knowledge base to answer your question."
        
        # Prepare context from retrieved documents
        context_text = "\n\n".join([
            f"Document {i+1} (Source: {os.path.basename(item['source'])}, Similarity: {item['similarity_score']:.3f}):\n{item['content']}" 
            for i, item in enumerate(context)
        ])
        
        # Create prompt with context
        prompt = f"""Based on the following context information, please provide a helpful and accurate answer to the user's question. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information you can.

Context Information:
{context_text}

User Question: {query}

Please provide a comprehensive answer based on the context:"""
        
        # Generate response
        try:
            outputs = self.generator(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0]['generated_text']
            
            # Extract only the new generated part (after the prompt)
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            # Clean up the response
            response = response.split('\n')[0] if '\n' in response else response
            
            return response
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            # Fallback: return the most relevant context
            most_relevant = context[0]['content'][:500] + "..." if len(context[0]['content']) > 500 else context[0]['content']
            return f"Based on my knowledge base: {most_relevant}"

# Global instance
rag_pipeline = FAISSRAGPipeline()