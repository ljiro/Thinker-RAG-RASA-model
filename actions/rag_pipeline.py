import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline
)
import torch
from typing import List, Dict, Any
import pickle

# FORCE CPU USAGE - Guaranteed stable
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("🚀 Initializing Enhanced RAG Pipeline...")

class EnhancedRAGPipeline:
    def __init__(self, knowledge_base_path: str = "knowledge_base/documents/"):
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model = None
        self.llm = None
        self.tokenizer = None
        self.generator = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        # FAISS configuration
        self.index_path = "vector_db/faiss_index.index"
        self.metadata_path = "vector_db/faiss_metadata.pkl"
        
        self._setup_models()
        self._setup_faiss_index()
    
    def _setup_models(self):
        """Initialize models with better performance"""
        print("📥 Loading embedding model...")
        
        # Lightweight embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.embedding_dim = 384
        
        print("📥 Loading LLM...")
        
        # Try a different, more capable model
        model_name = "microsoft/DialoGPT-medium"  # Upgraded to medium for better responses
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Fix tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create pipeline with better parameters
            self.generator = pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=self.tokenizer,
                device=-1,
                max_new_tokens=200,  # Increased for better responses
                do_sample=True,
                temperature=0.8,     # Slightly higher for more creative responses
                top_p=0.9,           # Nucleus sampling
                repetition_penalty=1.1,  # Reduce repetition
                pad_token_id=self.tokenizer.eos_token_id,
                torch_dtype=torch.float32
            )
            
            print("✅ All models loaded successfully")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}, falling back to small model")
            # Fallback to small model
            model_name = "microsoft/DialoGPT-small"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
            self.generator = pipeline(
                "text-generation",
                model=self.llm,
                tokenizer=self.tokenizer,
                device=-1,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print("✅ Fallback to DialoGPT-small loaded")
    
    def _setup_faiss_index(self):
        """Initialize or load FAISS index"""
        print("🔍 Setting up FAISS index...")
        
        os.makedirs("vector_db", exist_ok=True)
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            print(f"✅ Loaded FAISS index with {len(self.documents)} documents")
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.metadata = []
            print("✅ Created new FAISS index")
    
    def _save_faiss_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
    
    def chunk_text(self, text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            if i + chunk_size >= len(words):
                break
                
        return chunks
    
    def add_documents(self, file_path: str):
        """Add documents to the knowledge base"""
        import PyPDF2
        import docx
        
        print(f"📄 Processing document: {file_path}")
        
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
        
        if not text.strip():
            print(f"❌ No text extracted from {file_path}")
            return
        
        chunks = self.chunk_text(text)
        new_documents = []
        new_metadata = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 50:
                try:
                    embedding = self.embedding_model.encode([chunk])
                    embedding = embedding.astype('float32')
                    faiss.normalize_L2(embedding)
                    
                    self.index.add(embedding)
                    new_documents.append(chunk)
                    new_metadata.append({
                        "source": file_path,
                        "chunk_id": i,
                        "original_length": len(text)
                    })
                except Exception as e:
                    continue
        
        if new_documents:
            self.documents.extend(new_documents)
            self.metadata.extend(new_metadata)
            self._save_faiss_index()
            print(f"✅ Added {len(new_documents)} chunks from {file_path}")
    
    def search_similar(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.index.search(query_embedding, min(n_results, len(self.documents)))
            
            formatted_results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.documents):
                    formatted_results.append({
                        'content': self.documents[idx],
                        'source': self.metadata[idx]['source'],
                        'chunk_id': self.metadata[idx]['chunk_id'],
                        'similarity_score': float(distances[0][i])
                    })
            
            formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error in search_similar: {e}")
            return []
    
    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using the LLM with RAG context"""
        if not context:
            return "I couldn't find relevant information in my knowledge base to answer your question."
        
        try:
            # Prepare context from retrieved documents
            context_text = ""
            for i, item in enumerate(context[:2]):  # Use top 2 documents
                content = item['content']
                # Clean up the content
                content = ' '.join(content.split())  # Remove extra whitespace
                # Truncate if too long
                if len(content) > 400:
                    content = content[:400] + "..."
                context_text += f"Source {i+1}: {content}\n\n"
            
            # Create a more conversational prompt that encourages response
            prompt = f"""Here is some information from documents:

{context_text}
Based on this information, answer the following question: {query}

Answer:"""
            
            print(f"📝 PROMPT LENGTH: {len(prompt)} characters")
            print(f"📝 CONTEXT DOCUMENTS: {len(context)}")
            
            # Generate response with different parameters
            outputs = self.generator(
                prompt,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            response = outputs[0]['generated_text']
            
            print(f"🔍 RAW GENERATED TEXT: '{response}'")
            
            # Try different extraction methods
            if prompt in response:
                extracted = response.split(prompt)[-1].strip()
                print(f"✅ EXTRACTED VIA PROMPT SPLIT: '{extracted}'")
                response = extracted
            elif "Answer:" in response:
                extracted = response.split("Answer:")[-1].strip()
                print(f"✅ EXTRACTED VIA ANSWER SPLIT: '{extracted}'")
                response = extracted
            else:
                print(f"⚠️ USING FULL RESPONSE: '{response}'")
            
            # Clean up response
            response = response.split('\n')[0] if '\n' in response else response
            
            # If response is still empty or too short, use fallback
            if not response or len(response.strip()) < 20:
                print("🔄 USING FALLBACK RESPONSE")
                # Create a summary from the context
                key_points = []
                for i, item in enumerate(context[:2]):
                    content = item['content']
                    # Extract first sentence or first 100 chars
                    first_part = content.split('.')[0] if '.' in content else content[:150]
                    key_points.append(first_part.strip())
                
                response = f"Based on the Baguio City ordinances, here are some key traffic rules: {' '.join(key_points)}"
                if len(response) > 300:
                    response = response[:300] + "..."
            
            print(f"🎯 FINAL RESPONSE: '{response}'")
            
            return response
            
        except Exception as e:
            print(f"❌ GENERATION ERROR: {str(e)}")
            # Fallback: create a summary from context
            if context:
                summary_parts = []
                for item in context[:2]:
                    content = item['content']
                    # Take the first meaningful part
                    sentences = content.split('.')
                    if len(sentences) > 1:
                        summary_parts.append(sentences[0].strip() + ".")
                    else:
                        summary_parts.append(content[:200].strip() + "...")
                
                return f"Based on Baguio City ordinances: {' '.join(summary_parts)}"
            return "I found traffic regulation information but couldn't generate a detailed response."

# Initialize the pipeline
print("🚀 Starting enhanced RAG pipeline...")
rag_pipeline = EnhancedRAGPipeline()
print("✅ Enhanced RAG pipeline ready!")