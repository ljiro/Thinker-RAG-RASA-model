# rag_pipeline.py - PRESERVE NUMBERING VERSION
import os
import json
import logging
import numpy as np
import faiss
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, vector_db_path: str = "vector_db"):
        logger.info("🚀 Initializing RAG Pipeline...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vector_db_path = vector_db_path
        
        # Core RAG components
        self.embedder = None
        self.llm_client = None
        self.index = None
        self._documents = []
        self._metadata = []
        
        self._setup_models()
        self._load_vector_db()

    def _setup_models(self):
        """Setup embedding model and LLM client"""
        try:
            # Embedding model for retrieval
            logger.info("📥 Loading embedding model...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            
            # GPT-4o-mini for generation
            logger.info("🔗 Setting up GPT-4o-mini client...")
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                logger.error("❌ OPENAI_API_KEY not found.")
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            self.llm_client = OpenAI(api_key=api_key)
            logger.info("✅ RAG models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error setting up RAG models: {e}")
            raise

    def _load_vector_db(self):
        """Load existing vector database"""
        try:
            index_path = os.path.join(self.vector_db_path, "ordinance.index")
            metadata_path = os.path.join(self.vector_db_path, "chunks_metadata.json")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.warning("📚 No vector database found. Please run setup_ordinances.py first.")
                self._create_empty_index()
                return
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata and create documents
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)
            
            self._documents = []
            for meta in self._metadata:
                enhanced_content = (
                    f"BOOK {meta['book_number']}: {meta['book_title']}. "
                    f"ARTICLE {meta['article_number']}: {meta['article_title']}. "
                    f"SECTION {meta['section_number']}: {meta['section_title']}. "
                    f"CONTENT: {meta['content']}"
                )
                
                doc = {
                    "content": enhanced_content,
                    "source": f"Book {meta['book_number']}, Article {meta['article_number']}, Section {meta['section_number']}",
                    "metadata": meta
                }
                self._documents.append(doc)
            
            logger.info(f"📖 RAG system loaded with {len(self._documents)} ordinance sections")
            
        except Exception as e:
            logger.error(f"❌ Error loading vector DB: {e}")
            self._create_empty_index()

    def _create_empty_index(self):
        """Create empty FAISS index as fallback"""
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self._documents = []
        self._metadata = []

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for query"""
        if not self._documents or self.index is None:
            logger.warning("No documents in knowledge base")
            return []
        
        try:
            # Encode query
            query_emb = self.embedder.encode(query)
            
            # Search FAISS index
            D, I = self.index.search(np.array([query_emb]).astype("float32"), k)
            
            results = []
            for i, idx in enumerate(I[0]):
                if idx < len(self._documents):
                    doc = self._documents[idx]
                    similarity_score = float(1 - D[0][i] / 100.0)  # Normalize score
                    
                    results.append({
                        "content": doc["content"],
                        "source": doc["source"],
                        "metadata": doc["metadata"],
                        "similarity_score": similarity_score
                    })
            
            # Sort by similarity and filter
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            filtered_results = [r for r in results if r["similarity_score"] > 0.1]
            
            logger.info(f"🔍 Retrieved {len(filtered_results)} results")
            return filtered_results if filtered_results else results[:2]
            
        except Exception as e:
            logger.error(f"❌ Error in retrieval: {e}")
            return []

    def generate(self, query: str, context_docs: List[Dict[str, Any]] = None) -> str:
        """Generate response using retrieved context - preserves numbering and bullet points in ONE message"""
        if context_docs is None:
            context_docs = self.retrieve(query)
        
        if not context_docs:
            return "I couldn't find specific information about this in the Baguio City ordinances. Please try rephrasing your question or contact the relevant City Government office."
        
        # Prepare context for LLM
        context_text = "\n\n".join([
            f"SOURCE: {doc['source']}\nCONTENT: {doc['content']}"
            for doc in context_docs[:3]  # Use top 3 results
        ])
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert legal assistant specializing in Baguio City ordinances. Your role is to provide comprehensive, accurate, and well-structured information from the official Baguio City Code of Ordinances.

ORDINANCE CONTEXT:
{context_text}

USER QUESTION: {query}

CRITICAL RESPONSE GUIDELINES:

**STRUCTURE REQUIREMENTS:**
- Provide ONE SINGLE, COHESIVE RESPONSE - never split into multiple messages
- Use clear hierarchical organization with numbered lists (1., 2., 3., etc.) for main points
- **MAKE MAIN POINTS BOLD** by wrapping them in **asterisks** like **1. Main Point Title**
- Use bullet points (•) for supporting details and sub-points
- Maintain logical flow: Introduction → Main Points → Supporting Details → Conclusion
- Use line breaks between sections for readability

**CONTENT REQUIREMENTS:**
- Be comprehensive but concise - cover all relevant aspects of the question
- Include specific details like penalty amounts, requirements, procedures when available
- Focus on practical application - how the ordinances affect residents and visitors
- Highlight important safety information, deadlines, or legal requirements
- Provide complete information without needing follow-up questions

**FORMATTING PROHIBITIONS:**
- ❌ NO markdown headers (##, ###)
- ❌ NO technical references (Book X, Article Y, Section Z)
- ❌ NO source citations ("According to Book 3, Article 5...")
- ❌ NO fragmented responses - everything must be in one message

**RESPONSE STRUCTURE TEMPLATE:**
[Brief introduction establishing context and relevance]

**1. [First major regulation or requirement]**
   • [Supporting detail or specific rule]
   • [Additional relevant information]

**2. [Second major regulation or requirement]** 
   • [Supporting detail or specific rule]
   • [Additional relevant information]

**3. [Third major regulation or requirement]**
   • [Supporting detail or specific rule]
   • [Additional relevant information]

[Additional numbered points as needed...]

• [Important general guidelines]
• [Safety considerations]
• [Practical advice for compliance]

[Closing summary with key takeaways and any important reminders]

**BOLD FORMATTING EXAMPLES:**
- **1. Public Utility Jeepneys (PUJs) Routes**
- **2. Traffic Enforcement Procedures** 
- **3. Parking Regulations and Restrictions**

Ensure the response is authoritative yet accessible, comprehensive yet organized, and most importantly - entirely self-contained in one perfectly formatted message with bold main points."""
                    },
                    {
                        "role": "user", 
                        "content": query
                    }
                ],
                max_tokens=1200,
                temperature=0.2,
                top_p=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ GPT-4o-mini API error: {e}")
            # Fallback to best matching document as one response
            best_doc = context_docs[0]
            content = best_doc['content']
            # Remove book/article references from fallback
            content = content.split('CONTENT:')[-1] if 'CONTENT:' in content else content
            return f"Based on Baguio City ordinances:\n\n{content}"

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "total_documents": len(self._documents),
            "has_data": len(self._documents) > 0,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gpt-4o-mini"
        }

    # Compatibility methods for old code
    def search_similar(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Compatibility method for old code"""
        return self.retrieve(query, k=n_results)

    def generate_answer(self, query: str) -> str:
        """Compatibility method for old code"""
        return self.generate(query)

    # Properties for compatibility
    @property
    def documents(self):
        return self._documents


# Global instance
rag_pipeline = RAGPipeline()

# Export functions for easy access
def generate_answer(query: str) -> str:
    """Generate answer for a query"""
    return rag_pipeline.generate(query)

def retrieve_context(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve context for a query"""
    return rag_pipeline.retrieve(query, k)

# Test the RAG system
if __name__ == "__main__":
    try:
        print("\n🧪 Testing RAG Pipeline")
        stats = rag_pipeline.get_stats()
        print(f"📊 System Stats: {stats}")
        
        test_queries = [
            "traffic regulations in Baguio",
            "business permit requirements", 
            "What are the fines for traffic violations in Baguio?",
            "parking rules in Baguio City"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            answer = rag_pipeline.generate(query)
            print(f"✅ Answer: {answer}")
            print("---" * 20)
            
    except Exception as e:
        print(f"❌ RAG test failed: {e}")