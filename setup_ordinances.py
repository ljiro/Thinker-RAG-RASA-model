# setup_ordinances.py - Handles PDF processing and vector DB creation
import os
import json
import logging
import PyPDF2
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrdinanceProcessor:
    def __init__(self):
        logger.info("📄 Ordinance Processor initialized")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def process_pdf_to_vector_db(self, pdf_path: str, vector_db_path: str = "vector_db") -> bool:
        """Process PDF and create vector database"""
        try:
            logger.info(f"📖 Processing PDF: {pdf_path}")
            
            # Extract text from PDF
            text = self._extract_pdf_text(pdf_path)
            if not text:
                logger.error("❌ Failed to extract text from PDF")
                return False
            
            # Process text into chunks
            chunks = self._process_ordinance_text(text)
            if not chunks:
                logger.error("❌ Failed to process ordinance text")
                return False
            
            # Create vector database
            self._create_vector_db(chunks, vector_db_path)
            
            logger.info(f"✅ Successfully created vector DB with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing PDF: {e}")
            return False

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    text_content.append(f"===== Page {page_num + 1} =====\n{text}")
                    logger.info(f"📄 Processed page {page_num + 1}/{len(pdf_reader.pages)}")
                
                full_text = "\n".join(text_content)
                logger.info(f"✅ Extracted {len(full_text)} characters from PDF")
                return full_text
                
        except Exception as e:
            logger.error(f"❌ Error extracting PDF text: {e}")
            return ""

    def _process_ordinance_text(self, text: str) -> List[Dict[str, Any]]:
        """Process ordinance text into structured chunks"""
        logger.info("🔨 Processing ordinance text into chunks...")
        
        lines = text.split('\n')
        chunks = []
        
        current_book = {"number": "Unknown", "title": "Unknown"}
        current_article = {"number": "Unknown", "title": "General Provisions"}
        current_section = {"number": "Unknown", "title": ""}
        current_content = []
        page_num = 1
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Update page number
            if "===== Page" in line:
                page_match = re.search(r"Page (\d+)", line)
                if page_match:
                    page_num = int(page_match.group(1))
                i += 1
                continue
            
            if not line or self._is_page_header(line):
                i += 1
                continue
            
            # Detect BOOK
            if line.startswith("BOOK") and not line.startswith("BOOKMARK"):
                if current_content and current_section["number"] != "Unknown":
                    chunks.extend(self._finalize_chunk(
                        current_content, current_book, current_article, current_section, page_num
                    ))
                    current_content = []
                
                current_book = self._extract_book_info(line)
                current_content = [line]
            
            # Detect ARTICLE
            elif line.startswith("ARTICLE"):
                if current_content and current_section["number"] != "Unknown":
                    chunks.extend(self._finalize_chunk(
                        current_content, current_book, current_article, current_section, page_num
                    ))
                    current_content = []
                
                current_article = self._extract_article_info(line)
                current_content = [line]
            
            # Detect SECTION
            elif line.startswith("SECTION"):
                if current_content and current_section["number"] != "Unknown":
                    chunks.extend(self._finalize_chunk(
                        current_content, current_book, current_article, current_section, page_num
                    ))
                    current_content = []
                
                current_section = self._extract_section_info(line)
                current_content = [line]
            
            # Regular content
            else:
                current_content.append(line)
                
                if len(' '.join(current_content)) > 500 and current_section["number"] != "Unknown":
                    chunks.extend(self._finalize_chunk(
                        current_content, current_book, current_article, current_section, page_num
                    ))
                    current_content = [f"CONTINUED: {current_section['title']}"]
            
            i += 1
        
        # Final chunk
        if current_content and current_section["number"] != "Unknown":
            chunks.extend(self._finalize_chunk(
                current_content, current_book, current_article, current_section, page_num
            ))
        
        logger.info(f"✅ Processed {len(chunks)} ordinance chunks")
        return chunks

    def _extract_book_info(self, line: str) -> Dict[str, str]:
        try:
            parts = line.split(None, 1)
            return {"number": parts[1] if len(parts) > 1 else "I", "title": parts[1] if len(parts) > 1 else "General"}
        except:
            return {"number": "Unknown", "title": "Unknown"}

    def _extract_article_info(self, line: str) -> Dict[str, str]:
        try:
            parts = line.split(None, 1)
            return {"number": parts[1] if len(parts) > 1 else "I", "title": parts[1] if len(parts) > 1 else "General Provisions"}
        except:
            return {"number": "Unknown", "title": "General Provisions"}

    def _extract_section_info(self, line: str) -> Dict[str, str]:
        section_match = re.match(r"SECTION\s+(\d+(?:\.\d+)?)\.?\s*(.*)", line)
        if section_match:
            return {
                "number": section_match.group(1),
                "title": section_match.group(2).strip() if section_match.group(2) else ""
            }
        return {"number": "Unknown", "title": ""}

    def _finalize_chunk(self, content_lines: List[str], book: Dict, article: Dict, 
                       section: Dict, page_num: int) -> List[Dict[str, Any]]:
        if not content_lines:
            return []
        
        content = ' '.join(content_lines)
        
        chunk = {
            "content": content,
            "book_number": book["number"],
            "book_title": book["title"],
            "article_number": article["number"],
            "article_title": article["title"],
            "section_number": section["number"],
            "section_title": section["title"],
            "page_number": page_num,
            "content_type": self._determine_content_type(content),
            "chunk_id": f"{book['number']}_{article['number']}_{section['number']}_{page_num}"
        }
        
        return [chunk]

    def _determine_content_type(self, content: str) -> str:
        content_upper = content.upper()
        
        if any(word in content_upper for word in ["PENALTY", "FINE", "VIOLATION"]):
            return "penalty"
        elif any(word in content_upper for term in ["DEFINITION", "MEANING"] for word in [term]):
            return "definition"
        elif any(word in content_upper for term in ["REQUIREMENT", "MUST", "SHALL"] for word in [term]):
            return "requirement"
        elif any(word in content_upper for term in ["PROHIBITED", "PROHIBITION"] for word in [term]):
            return "prohibition"
        
        return "general"

    def _is_page_header(self, line: str) -> bool:
        return any(indicator in line for indicator in [
            "===== Page", "Ordinance Numbered", "Baguio City Code"
        ])

    def _create_vector_db(self, chunks: List[Dict[str, Any]], vector_db_path: str):
        """Create vector database from chunks"""
        logger.info("🔨 Creating vector database...")
        
        dim = self.embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        
        # Process chunks and create embeddings
        for i, chunk in enumerate(chunks):
            enhanced_content = (
                f"BOOK {chunk['book_number']}: {chunk['book_title']}. "
                f"ARTICLE {chunk['article_number']}: {chunk['article_title']}. "
                f"SECTION {chunk['section_number']}: {chunk['section_title']}. "
                f"CONTENT: {chunk['content']}"
            )
            
            embedding = self.embedder.encode(enhanced_content)
            index.add(np.array([embedding]).astype("float32"))
            
            if (i + 1) % 500 == 0:
                logger.info(f"📄 Embedded {i + 1}/{len(chunks)} chunks")
        
        # Save to disk
        self._save_vector_db(index, chunks, vector_db_path)

    def _save_vector_db(self, index, chunks: List[Dict[str, Any]], vector_db_path: str):
        """Save vector database to disk"""
        try:
            os.makedirs(vector_db_path, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(index, os.path.join(vector_db_path, "ordinance.index"))
            
            # Save metadata
            with open(os.path.join(vector_db_path, "chunks_metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            
            # Save config
            config = {
                "embedding_model": "all-MiniLM-L6-v2",
                "total_chunks": len(chunks),
                "created_at": str(np.datetime64('now'))
            }
            with open(os.path.join(vector_db_path, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 Vector DB saved to {vector_db_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving vector DB: {e}")
            raise

def main():
    """Main function to process PDF and create vector database"""
    processor = OrdinanceProcessor()
    
    # PDF paths to check
    pdf_paths = [
        "knowledge_base/documents/Baguio-City-Code-of-Ordinances.pdf",
        "Baguio-City-Code-of-Ordinances.pdf"
    ]
    
    # Find PDF file
    pdf_path = None
    for path in pdf_paths:
        if os.path.exists(path):
            pdf_path = path
            break
    
    if not pdf_path:
        print("❌ No PDF file found. Please ensure the Baguio City ordinances PDF is in the correct location.")
        return
    
    print(f"🚀 Processing PDF: {pdf_path}")
    
    # Process PDF and create vector database
    success = processor.process_pdf_to_vector_db(pdf_path, "vector_db")
    
    if success:
        print("✅ Ordinance system setup complete!")
        print("🎯 You can now start your Rasa actions server")
    else:
        print("❌ Failed to setup ordinance system")

if __name__ == "__main__":
    main()