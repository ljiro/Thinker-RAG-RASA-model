# actions/pdf_processor.py
import os
import PyPDF2
import pdfplumber
import json
from typing import List, Dict, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len
        )
        self.documents = []
        self.load_existing_documents()
    
    def load_existing_documents(self):
        """Load existing processed documents"""
        try:
            with open('documents/processed_documents.json', 'r') as f:
                self.documents = json.load(f)
            print(f"📄 Loaded {len(self.documents)} existing C++ documents from cache")
        except FileNotFoundError:
            self.documents = []
            print("📄 No existing C++ document cache found")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file with better formatting for C++ content"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up the text
                        page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                        page_text = page_text.replace('-\n', '')    # Remove hyphenated line breaks
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying PyPDF2")
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            page_text = re.sub(r'\s+', ' ', page_text)
                            page_text = page_text.replace('-\n', '')
                            text += page_text + "\n"
            except Exception as e2:
                print(f"PyPDF2 also failed: {e2}")
                raise
        
        return text.strip()
    
    def parse_cpp_content(self, text: str, file_name: str) -> Dict[str, Any]:
        """Parse C++ programming content from text with focus on C++ concepts"""
        content = {
            "title": "",
            "description": "",
            "category": "C++ Programming",
            "topics": [],
            "code_examples": [],
            "key_points": [],
            "concepts": [],
            "source": "PDF",
            "file_name": file_name,
            "content": text,
            "cpp_standard": "",
            "difficulty": "Intermediate"
        }
        
        lines = text.split('\n')
        
        # Extract title from first meaningful line
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                if not content["title"]:
                    content["title"] = line
                break
        
        # Auto-detect C++ specific content
        self._detect_cpp_content(content, text)
        
        # If no title found, create one
        if not content["title"]:
            content["title"] = f"C++ Concept from {file_name}"
        
        # Use beginning as description
        content["description"] = text[:800] + "..." if len(text) > 800 else text
        
        return content
    
    def _detect_cpp_content(self, content: Dict, text: str):
        """Detect C++ specific content and topics"""
        text_lower = text.lower()
        
        # C++ Specific Topics
        cpp_topics = {
            "classes": ["class", "object", "constructor", "destructor", "member function"],
            "templates": ["template", "typename", "template class", "template function"],
            "inheritance": ["inheritance", "derive", "base class", "virtual", "polymorphism"],
            "pointers": ["pointer", "reference", "dereference", "addressof", "smart pointer"],
            "memory": ["memory", "heap", "stack", "new", "delete", "malloc", "free"],
            "stl": ["vector", "map", "set", "algorithm", "iterator", "stl"],
            "functions": ["function", "parameter", "return", "overload", "lambda"],
            "operators": ["operator", "overload", "friend function"],
            "exceptions": ["exception", "try", "catch", "throw"],
            "cpp11+": ["auto", "lambda", "smart pointer", "move semantics", "constexpr"]
        }
        
        # C++ Standards detection
        if "c++11" in text_lower or "c++0x" in text_lower:
            content["cpp_standard"] = "C++11"
        elif "c++14" in text_lower:
            content["cpp_standard"] = "C++14"
        elif "c++17" in text_lower:
            content["cpp_standard"] = "C++17"
        elif "c++20" in text_lower:
            content["cpp_standard"] = "C++20"
        else:
            content["cpp_standard"] = "C++ Standard"
        
        # Detect topics
        detected_topics = []
        for topic, keywords in cpp_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
                content["concepts"].append(topic)
        
        content["topics"] = detected_topics
        
        # Extract code examples
        self._extract_cpp_code_examples(content, text)
        
        # Extract key points
        self._extract_key_points(content, text)
    
    def _extract_cpp_code_examples(self, content: Dict, text: str):
        """Extract C++ code examples from text"""
        # Look for code blocks or function definitions
        code_patterns = [
            r'```cpp(.*?)```',
            r'```c\+\+(.*?)```',
            r'#include.*?\{.*?\}',
            r'class\s+\w+.*?\{.*?\}',
            r'void\s+\w+.*?\{.*?\}',
            r'int\s+main.*?\{.*?\}'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if len(match.strip()) > 10:
                    content["code_examples"].append(match.strip())
        
        # If no structured code found, look for lines that look like C++
        if not content["code_examples"]:
            lines = text.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if any(keyword in line for keyword in ['#include', 'using namespace', 'class ', 'void ', 'int ', 'cout', 'cin', 'std::']):
                    in_code = True
                    code_lines.append(line.strip())
                elif in_code and line.strip() and not line.strip().startswith('//'):
                    code_lines.append(line.strip())
                elif in_code and not line.strip():
                    if code_lines:
                        content["code_examples"].append('\n'.join(code_lines))
                        code_lines = []
                    in_code = False
            
            if code_lines:
                content["code_examples"].append('\n'.join(code_lines))
    
    def _extract_key_points(self, content: Dict, text: str):
        """Extract key points from C++ content"""
        lines = text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            # Look for important points (often marked with *, -, •, or numbered)
            if (line.startswith('*') or line.startswith('-') or 
                line.startswith('•') or re.match(r'^\d+\.', line)):
                point = re.sub(r'^[*\-•\d\.\s]+', '', line)
                if len(point) > 10:
                    key_points.append(point)
            
            # Look for lines with key C++ terms
            elif any(term in line.lower() for term in ['important', 'note', 'remember', 'key point', 'essential']):
                if len(line) > 15:
                    key_points.append(line)
        
        content["key_points"] = key_points[:10]  # Limit to 10 key points
    
    def process_pdf_directory(self, directory: str = "documents/uploaded_pdfs") -> List[Dict[str, Any]]:
        """Process all PDFs in directory focusing on C++ content"""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created PDF directory: {directory}")
            return []
        
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"📭 No PDF files found in {directory}")
            print(f"💡 Please add your C++ PDF files to: {directory}")
            return []
        
        print(f"📥 Found {len(pdf_files)} PDF files. Processing for C++ content...")
        
        all_contents = []
        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            try:
                # Check if this PDF is already processed
                if not self.is_pdf_processed(file_path):
                    contents = self.process_single_pdf(file_path)
                    all_contents.extend(contents)
                    print(f"✅ Processed {pdf_file}: {len(contents)} C++ concepts")
                    for content in contents[:2]:  # Show first 2 concepts
                        print(f"   📖 {content['title']}")
                else:
                    print(f"📋 PDF already processed: {pdf_file}")
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
        
        return all_contents
    
    def is_pdf_processed(self, file_path: str) -> bool:
        """Check if a PDF has already been processed"""
        for doc in self.documents:
            if doc.get('metadata', {}).get('source') == file_path:
                return True
        return False
    
    def process_single_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process single PDF file for C++ content"""
        text = self.extract_text_from_pdf(file_path)
        
        # Split text into chunks that might represent individual concepts
        chunks = self.text_splitter.split_text(text)
        
        contents = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 100:  # Skip very short chunks
                continue
                
            content = self.parse_cpp_content(chunk, os.path.basename(file_path))
            content["chunk_id"] = f"{os.path.basename(file_path)}_{i}"
            contents.append(content)
        
        # Add to documents
        for content in contents:
            self.documents.append({
                "id": content["chunk_id"],
                "title": content["title"],
                "content": content["content"],
                "metadata": {
                    "category": content["category"],
                    "topics": content["topics"],
                    "concepts": content["concepts"],
                    "source": file_path,
                    "file_name": content["file_name"],
                    "cpp_standard": content["cpp_standard"],
                    "difficulty": content["difficulty"],
                    "code_examples": content["code_examples"],
                    "key_points": content["key_points"]
                }
            })
        
        # Save documents
        self.save_documents()
        
        return contents
    
    def save_documents(self):
        """Save processed documents to file"""
        os.makedirs('documents', exist_ok=True)
        with open('documents/processed_documents.json', 'w') as f:
            json.dump(self.documents, f, indent=2)
    
    def get_cpp_contents(self) -> List[Dict[str, Any]]:
        """Get all processed C++ contents"""
        contents = []
        for doc in self.documents:
            contents.append({
                "title": doc["title"],
                "description": doc["content"][:600] + "..." if len(doc["content"]) > 600 else doc["content"],
                "category": doc["metadata"]["category"],
                "topics": doc["metadata"]["topics"],
                "concepts": doc["metadata"]["concepts"],
                "cpp_standard": doc["metadata"]["cpp_standard"],
                "difficulty": doc["metadata"]["difficulty"],
                "code_examples": doc["metadata"]["code_examples"],
                "key_points": doc["metadata"]["key_points"],
                "source": "PDF",
                "file_name": doc["metadata"]["file_name"],
                "content": doc["content"]
            })
        return contents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about C++ documents"""
        stats = {
            "total_documents": len(self.documents),
            "unique_files": len(set(doc['metadata']['source'] for doc in self.documents)),
            "topics_count": {},
            "concepts_count": {},
            "cpp_standards": {}
        }
        
        # Count topics and concepts
        for doc in self.documents:
            for topic in doc['metadata']['topics']:
                stats["topics_count"][topic] = stats["topics_count"].get(topic, 0) + 1
            
            for concept in doc['metadata']['concepts']:
                stats["concepts_count"][concept] = stats["concepts_count"].get(concept, 0) + 1
            
            standard = doc['metadata']['cpp_standard']
            stats["cpp_standards"][standard] = stats["cpp_standards"].get(standard, 0) + 1
        
        return stats
    
    def clear_documents(self):
        """Clear all PDF documents"""
        self.documents = []
        try:
            os.remove('documents/processed_documents.json')
        except FileNotFoundError:
            pass
        print("🧹 Cleared all C++ documents")