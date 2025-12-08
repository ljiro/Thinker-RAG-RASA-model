# rag_pipeline.py - UPDATED FOR CORPORATE DOCUMENTS (Preserve your formatting)
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

class CorporateRAGPipeline:
    def __init__(self, vector_db_path: str = "knowledge_base/vector_db"):
        logger.info("🏢 Initializing Corporate RAG Pipeline for Gift of Grace...")
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
            # Embedding model - using smaller model for memory efficiency
            logger.info("📥 Loading embedding model...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
            
            # GPT-4o-mini for generation
            logger.info("🔗 Setting up GPT-4o-mini client...")
            api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                logger.error("❌ OPENAI_API_KEY not found in .env file.")
                logger.warning("⚠️ OpenAI features will be disabled. Add OPENAI_API_KEY to .env for better responses.")
                self.llm_client = None
            else:
                self.llm_client = OpenAI(api_key=api_key)
                logger.info("✅ OpenAI client initialized successfully")
            
            logger.info("✅ RAG models initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up RAG models: {e}")
            # Don't raise, allow fallback mode
            self.llm_client = None

    def _load_vector_db(self):
        """Load existing vector database for corporate documents"""
        try:
            index_path = os.path.join(self.vector_db_path, "corporate.index")
            documents_path = os.path.join(self.vector_db_path, "corporate_documents.json")
            
            if not os.path.exists(index_path) or not os.path.exists(documents_path):
                logger.warning("📚 No corporate database found. Please run setup_ordinances.py first.")
                self._create_empty_index()
                return
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents
            with open(documents_path, 'r', encoding='utf-8') as f:
                self._documents = json.load(f)
            
            logger.info(f"📖 Corporate RAG system loaded with {len(self._documents)} document chunks")
            logger.info(f"🏭 Company: Gift of Grace Food Manufacturing Corporation")
            
        except Exception as e:
            logger.error(f"❌ Error loading corporate vector DB: {e}")
            self._create_empty_index()

    def _create_empty_index(self):
        """Create empty FAISS index as fallback"""
        if self.embedder:
            dim = self.embedder.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
        self._documents = []
        self._metadata = []

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant corporate document chunks for query"""
        if not self._documents or self.index is None:
            logger.warning("No corporate documents in knowledge base")
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
            
            logger.info(f"🔍 Retrieved {len(filtered_results)} corporate document results")
            return filtered_results if filtered_results else results[:2]
            
        except Exception as e:
            logger.error(f"❌ Error in corporate retrieval: {e}")
            return []

    def generate_with_openai(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using OpenAI"""
        if not self.llm_client:
            return self._generate_fallback(query, context_docs)
        
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
                        "content": f"""You are a corporate information assistant specializing in Gift of Grace Food Manufacturing Corporation (GoGFMC). Your role is to provide comprehensive, accurate, and well-structured information from their official corporate report.

CORPORATE REPORT CONTEXT:
{context_text}

USER QUESTION: {query}

CRITICAL RESPONSE GUIDELINES:

**ABOUT THE COMPANY:**
- Gift of Grace Food Manufacturing Corporation (GoGFMC) is a Filipino food manufacturing company
- Headquartered in Baguio City, Cordillera Administrative Region
- Founded by Satur Cadsi (CEO) and Janice Osenio Cadsi (COO)
- Specializes in healthy food products like kimchi, tofu, and rice coffee
- Award-winning MSME with Halal certification

**RESPONSE STRUCTURE REQUIREMENTS:**
- Provide ONE SINGLE, COHESIVE RESPONSE - never split into multiple messages
- Use clear hierarchical organization with numbered lists (1., 2., 3., etc.) for main points
- **MAKE MAIN POINTS BOLD** by wrapping them in **asterisks** like **1. Company Background**
- Use bullet points (•) for supporting details and sub-points
- Maintain logical flow: Introduction → Main Points → Details → Conclusion
- Use line breaks between sections for readability

**CONTENT REQUIREMENTS:**
- Be comprehensive but concise - cover all relevant aspects of the question
- Include specific details like product information, awards, contact details when available
- Focus on practical information - what the company does, makes, and offers
- Highlight important achievements, certifications, and social programs
- Provide complete information without needing follow-up questions

**FORMATTING PROHIBITIONS:**
- ❌ NO markdown headers (##, ###)
- ❌ NO technical references ("Page X, Chunk Y")
- ❌ NO source citations ("According to page 3...")
- ❌ NO fragmented responses - everything must be in one message

**RESPONSE STRUCTURE TEMPLATE:**
[Brief introduction about Gift of Grace Food Manufacturing Corporation]

**1. [First major topic or answer component]**
   • [Supporting detail or specific information]
   • [Additional relevant information]

**2. [Second major topic or answer component]** 
   • [Supporting detail or specific information]
   • [Additional relevant information]

**3. [Third major topic or answer component]**
   • [Supporting detail or specific information]
   • [Additional relevant information]

[Additional numbered points as needed...]

• [Important general information]
• [Key achievements or certifications]
• [Practical details for consumers or partners]

[Closing summary with key takeaways about the company]

**BOLD FORMATTING EXAMPLES:**
- **1. Company Origins and History**
- **2. Product Portfolio Overview** 
- **3. Awards and Recognition**
- **4. Contact Information**

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
            return self._generate_fallback(query, context_docs)

    def _generate_fallback(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Fallback response when OpenAI is not available"""
        if not context_docs:
            return "I couldn't find specific information about Gift of Grace Food Manufacturing Corporation in my knowledge base."
        
        # Simple keyword-based response
        best_doc = context_docs[0]
        content = best_doc['content']
        
        # Extract key sentences based on query
        query_lower = query.lower()
        sentences = content.split('. ')
        
        relevant_sentences = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in query_lower.split() if len(word) > 3):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            response = "**Based on Gift of Grace Food Manufacturing Corporation's corporate report:**\n\n"
            for i, sentence in enumerate(relevant_sentences[:5], 1):
                response += f"**{i}. {sentence}**\n"
            return response
        else:
            return f"**Based on Gift of Grace Food Manufacturing Corporation's corporate report:**\n\n{content[:500]}..."

    def generate(self, query: str, context_docs: List[Dict[str, Any]] = None) -> str:
        """Generate response using retrieved corporate context"""
        if context_docs is None:
            context_docs = self.retrieve(query)
        
        if not context_docs:
            return "I couldn't find specific information about that in the Gift of Grace corporate report. Please try asking about their products, history, awards, or contact information."
        
        if self.llm_client:
            return self.generate_with_openai(query, context_docs)
        else:
            return self._generate_fallback(query, context_docs)

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "total_documents": len(self._documents),
            "has_data": len(self._documents) > 0,
            "company": "Gift of Grace Food Manufacturing Corporation",
            "document_type": "Corporate Report (2025)",
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gpt-4o-mini" if self.llm_client else "Fallback mode",
            "openai_available": self.llm_client is not None
        }

    def generate_answer(self, query: str) -> str:
        """Generate answer for a query"""
        return self.generate(query)

    def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve context for a query"""
        return self.retrieve(query, k)


# Global instance
rag_pipeline = CorporateRAGPipeline()

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
        print("\n🏢 Testing Corporate RAG Pipeline - Gift of Grace")
        stats = rag_pipeline.get_stats()
        print(f"📊 System Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        test_queries = [
            "Tell me about Gift of Grace Food Manufacturing Corporation",
            "What products does Gift of Grace make?",
            "Who are the founders of Gift of Grace?",
            "What awards has Gift of Grace won?",
            "Where is Gift of Grace located?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            answer = rag_pipeline.generate(query)
            print(f"✅ Answer length: {len(answer)} characters")
            print(f"Preview: {answer[:200]}...")
            print("---" * 20)
            
    except Exception as e:
        print(f"❌ Corporate RAG test failed: {e}")
        import traceback
        traceback.print_exc()