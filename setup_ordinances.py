# setup_ordinances.py - OPTIMIZED FOR CORPORATE DOCUMENTS
import os
import json
import re
import logging
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    try:
        import faiss_cpu as faiss
        FAISS_AVAILABLE = True
    except ImportError:
        FAISS_AVAILABLE = False
        logger.error("❌ FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.error("❌ Sentence Transformers not available")

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.error("❌ pypdf not available. Install with: pip install pypdf")

class CorporateDocumentProcessor:
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        logger.info("🏢 Initializing Corporate Document Processor...")
        
        # Check dependencies
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("Sentence Transformers not available")
        if not PYPDF_AVAILABLE:
            raise ImportError("pypdf not available")
        
        self.knowledge_base_path = knowledge_base_path
        self.documents_path = os.path.join(knowledge_base_path, "documents")
        self.vector_db_path = os.path.join(knowledge_base_path, "vector_db")
        
        # Create directories
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # Initialize embedding model with memory optimization
        logger.info("🧠 Loading embedding model...")
        try:
            # Try a smaller model first for memory efficiency
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Using all-MiniLM-L6-v2 model")
        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            raise
        
        # Initialize variables
        self.index = None
        self.chunks = []
        self.metadata = []

    def extract_pdf_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF"""
        logger.info(f"📄 Extracting content from: {pdf_path}")
        
        try:
            reader = PdfReader(pdf_path)
            pages_content = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    # Clean text
                    text = re.sub(r'\s+', ' ', text)
                    text = text.replace('\n', ' ').replace('\r', ' ')
                    
                    pages_content.append({
                        "content": text.strip(),
                        "page_number": page_num,
                        "source_file": os.path.basename(pdf_path),
                        "document_type": "corporate_report",
                        "company": "Gift of Grace Food Manufacturing Corporation"
                    })
            
            logger.info(f"✅ Extracted {len(pages_content)} pages")
            return pages_content
            
        except Exception as e:
            logger.error(f"❌ Error extracting PDF: {e}")
            return []

    def create_chunks(self, pages: List[Dict[str, Any]], chunk_size: int = 400) -> List[Dict[str, Any]]:
        """Create text chunks"""
        if not pages:
            return []
        
        logger.info(f"✂️ Creating chunks from {len(pages)} pages...")
        all_chunks = []
        
        for page in pages:
            content = page["content"]
            
            # Simple chunking
            start = 0
            chunk_num = 1
            
            while start < len(content):
                end = start + chunk_size
                if end < len(content):
                    # Try to end at sentence boundary
                    sentence_end = content.rfind('. ', start, end)
                    if sentence_end > start + chunk_size * 0.5:
                        end = sentence_end + 1
                
                chunk_text = content[start:end].strip()
                if chunk_text:
                    chunk = page.copy()
                    chunk["content"] = chunk_text
                    chunk["chunk_number"] = chunk_num
                    all_chunks.append(chunk)
                    chunk_num += 1
                
                start = end
        
        logger.info(f"✅ Created {len(all_chunks)} chunks")
        return all_chunks

    def create_vector_database(self):
        """Create vector database"""
        if not self.chunks:
            logger.error("❌ No chunks to process")
            return False
        
        logger.info(f"🔢 Creating vector database from {len(self.chunks)} chunks...")
        
        try:
            # Extract content
            contents = [chunk["content"] for chunk in self.chunks]
            
            # Generate embeddings in smaller batches
            logger.info("🧠 Generating embeddings (please wait)...")
            embeddings_list = []
            batch_size = 8  # Small batch for memory
            
            for i in range(0, len(contents), batch_size):
                batch = contents[i:i+batch_size]
                logger.info(f"  Processing batch {i//batch_size + 1}/{(len(contents)-1)//batch_size + 1}")
                batch_embeddings = self.embedder.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings_list.append(batch_embeddings)
            
            # Combine embeddings
            embeddings = np.vstack(embeddings_list)
            
            # Normalize
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            
            # Save FAISS index
            index_path = os.path.join(self.vector_db_path, "corporate.index")
            faiss.write_index(self.index, index_path)
            logger.info(f"💾 Saved FAISS index to {index_path}")
            
            # Save metadata
            metadata_path = os.path.join(self.vector_db_path, "corporate_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved metadata to {metadata_path}")
            
            # Create enhanced documents
            enhanced_docs = []
            for i, chunk in enumerate(self.chunks):
                enhanced_docs.append({
                    "content": chunk["content"],
                    "source": f"Page {chunk['page_number']}, Chunk {chunk.get('chunk_number', i+1)}",
                    "metadata": {
                        "page": chunk["page_number"],
                        "chunk": chunk.get("chunk_number", i+1),
                        "file": chunk["source_file"],
                        "type": chunk["document_type"],
                        "company": chunk["company"]
                    }
                })
            
            # Save documents
            docs_path = os.path.join(self.vector_db_path, "corporate_documents.json")
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_docs, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 Saved {len(enhanced_docs)} document chunks")
            
            # Create summary
            summary = {
                "company": "Gift of Grace Food Manufacturing Corporation (GoGFMC)",
                "total_chunks": len(enhanced_docs),
                "document_type": "corporate_report",
                "processed_date": datetime.now().isoformat(),
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": dimension
            }
            
            summary_path = os.path.join(self.vector_db_path, "summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Vector database created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating vector database: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("🏢 GIFT OF GRACE FOOD MANUFACTURING CORP")
    print("        KNOWLEDGE BASE SETUP")
    print("="*60)
    
    try:
        # Check dependencies
        print("\n🔍 Checking dependencies...")
        
        if not FAISS_AVAILABLE:
            print("❌ FAISS not available. Install with: pip install faiss-cpu")
            return False
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ Sentence Transformers not available. Install with: pip install sentence-transformers")
            return False
        if not PYPDF_AVAILABLE:
            print("❌ pypdf not available. Install with: pip install pypdf")
            return False
        
        print("✅ All dependencies available")
        
        # Initialize processor
        processor = CorporateDocumentProcessor()
        
        # Find PDF files
        pdf_files = [f for f in os.listdir(processor.documents_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"\n⚠️ No PDF files found in {processor.documents_path}")
            print(f"📥 Please place your PDF document in that folder")
            return False
        
        print(f"\n📑 Found {len(pdf_files)} PDF file(s):")
        for pdf_file in pdf_files:
            print(f"   • {pdf_file}")
        
        # Process PDFs
        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(processor.documents_path, pdf_file)
            print(f"\n🔄 Processing: {pdf_file}")
            
            pages = processor.extract_pdf_content(pdf_path)
            if pages:
                chunks = processor.create_chunks(pages)
                all_chunks.extend(chunks)
                print(f"✅ Processed: {len(chunks)} chunks")
            else:
                print(f"❌ Failed to extract content")
        
        if not all_chunks:
            print("❌ No content extracted")
            return False
        
        # Set chunks and create vector database
        processor.chunks = all_chunks
        print(f"\n📊 Total chunks: {len(all_chunks)}")
        
        print("\n🔄 Creating vector database...")
        if processor.create_vector_database():
            print("\n" + "="*60)
            print("🎉 SETUP COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("\n💡 Next Steps:")
            print("   1. Make sure OPENAI_API_KEY is in .env file")
            print("   2. Restart Rasa actions server")
            print("   3. Run: rasa shell")
            return True
        else:
            print("\n❌ SETUP FAILED!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)