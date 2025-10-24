# actions/actions.py
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json
import os
import re

try:
    from .pdf_processor import PDFProcessor
except ImportError:
    from pdf_processor import PDFProcessor

class CppRAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.pdf_processor = PDFProcessor()
        self.index = None
        self.cpp_contents = []
        self.all_documents = []
        
        self.load_cpp_contents()
        self.build_vector_index()
        self.print_welcome_message()
    
    def print_welcome_message(self):
        """Print welcome message with system status"""
        stats = self.pdf_processor.get_stats()
        
        print("\n" + "="*60)
        print("🤖 C++ PROGRAMMING RAG ASSISTANT")
        print("="*60)
        print(f"📚 C++ Concepts: {len(self.cpp_contents)}")
        print(f"📄 PDF Files: {stats['unique_files']}")
        print(f"🔍 Vector Index: {self.index.ntotal if self.index else 0} embeddings")
        print(f"🏷️ Topics: {len(stats['topics_count'])}")
        print(f"💡 Concepts: {len(stats['concepts_count'])}")
        print("="*60)
        print("💡 Ready to help with C++ programming!")
        print("="*60 + "\n")
    
    def load_cpp_contents(self):
        """Load C++ contents from PDFs"""
        print("📥 Scanning for C++ PDF files...")
        
        # Process PDFs if not already processed
        pdf_contents = self.pdf_processor.process_pdf_directory()
        if pdf_contents:
            print(f"✅ Processed {len(pdf_contents)} C++ concepts from PDFs")
        
        # Get all C++ contents
        self.cpp_contents = self.pdf_processor.get_cpp_contents()
        
        if not self.cpp_contents:
            print("❌ No C++ content found. Please add C++ PDF files to 'documents/uploaded_pdfs/'")
        else:
            print(f"✅ Loaded {len(self.cpp_contents)} C++ concepts")
    
    def build_vector_index(self):
        """Build FAISS vector index for C++ content"""
        self.all_documents = []
        
        for content in self.cpp_contents:
            # Create rich text representation for semantic search
            text_representation = f"""
            Title: {content['title']}
            Description: {content['description']}
            Category: {content['category']}
            Topics: {', '.join(content['topics'])}
            Concepts: {', '.join(content['concepts'])}
            C++ Standard: {content['cpp_standard']}
            Difficulty: {content['difficulty']}
            """
            
            self.all_documents.append({
                "content": text_representation,
                "cpp_content": content
            })
        
        if not self.all_documents:
            self.index = None
            return
        
        # Generate embeddings
        texts = [doc["content"] for doc in self.all_documents]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        print(f"🔍 C++ Vector database built with {len(self.all_documents)} concepts")
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search C++ content using semantic similarity"""
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.all_documents):
                results.append({
                    'cpp_content': self.all_documents[idx]['cpp_content'],
                    'score': float(score)
                })
        
        return results
    
    def process_query(self, user_input: str) -> str:
        """Main method to process any user query about C++"""
        input_lower = user_input.lower().strip()
        
        if not input_lower:
            return self._get_welcome_message()
        
        # Handle greetings
        if any(word in input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            return self._get_welcome_message()
        
        # Handle goodbye
        if any(word in input_lower for word in ['bye', 'goodbye', 'see you', 'exit', 'quit']):
            return "👋 Goodbye! Happy C++ coding! 🚀"
        
        # Handle thanks
        if any(word in input_lower for word in ['thanks', 'thank you', 'thank']):
            return "You're welcome! 😊 Let me know if you need more help with C++."
        
        # Handle help requests
        if any(word in input_lower for word in ['help', 'what can you do', 'abilities']):
            return self._get_help_message()
        
        # Handle specific C++ concept searches
        if any(word in input_lower for word in ['class', 'template', 'inheritance', 'pointer', 'memory', 'stl', 'function', 'operator']):
            return self._handle_cpp_concept_query(user_input)
        
        # Handle code examples requests
        if any(word in input_lower for word in ['code', 'example', 'implement', 'syntax', 'program']):
            return self._handle_code_example_query(user_input)
        
        # Handle explanation requests
        if any(word in input_lower for word in ['explain', 'how', 'why', 'work', 'what is']):
            return self._handle_explanation_query(user_input)
        
        # Handle list requests
        if any(word in input_lower for word in ['list', 'show', 'all concepts', 'available', 'topics']):
            return self._list_cpp_concepts()
        
        # Handle system status
        if any(word in input_lower for word in ['status', 'how many', 'statistics']):
            return self._get_system_status()
        
        # Handle refresh
        if any(word in input_lower for word in ['refresh', 'reload', 'update']):
            return self._refresh_pdfs()
        
        # Default: semantic search
        return self._handle_semantic_search(user_input)
    
    def _get_welcome_message(self) -> str:
        """Generate welcome message"""
        stats = self.pdf_processor.get_stats()
        
        return f"""🤖 **Welcome to your C++ Programming Assistant!**

I have analyzed your C++ PDF documents and found **{len(self.cpp_contents)} C++ concepts** across **{stats['unique_files']} PDF files**.

**I can help you with:**
• C++ concepts and explanations
• Code examples and syntax
• Object-oriented programming
• Templates and STL
• Memory management
• And much more!

**Try asking me:**
• "Explain classes in C++"
• "Show me template examples"
• "What is inheritance?"
• "Give me pointer code examples"
• "List all C++ topics"

What C++ concept would you like to explore?"""
    
    def _get_help_message(self) -> str:
        """Generate help message"""
        return """🆘 **C++ Programming Help**

🔍 **Find C++ Concepts:**
   - "Explain classes and objects"
   - "What are templates?"
   - "How does inheritance work?"
   - "Tell me about smart pointers"

💻 **Get Code Examples:**
   - "Show me class examples"
   - "Template function code"
   - "STL vector usage"
   - "Pointer arithmetic examples"

📚 **Learn C++ Features:**
   - "C++11 features"
   - "Memory management techniques"
   - "Operator overloading"
   - "Exception handling"

📋 **Explore:**
   - "List all C++ topics"
   - "What concepts are available?"
   - "System status"

I specialize in C++ programming from your PDF documents!"""
    
    def _handle_cpp_concept_query(self, query: str) -> str:
        """Handle C++ concept queries"""
        results = self.semantic_search(query, top_k=1)
        if not results:
            return self._handle_semantic_search(query)
        
        content = results[0]['cpp_content']
        similarity = results[0]['score']
        
        response = f"📚 **{content['title']}**\n"
        response += f"🏷️ Category: {content['category']}\n"
        response += f"⚡ C++ Standard: {content['cpp_standard']}\n"
        response += f"📊 Relevance: {similarity:.3f}\n\n"
        
        response += f"📝 **Description:**\n{content['description']}\n\n"
        
        if content['topics']:
            response += f"🔖 **Topics:** {', '.join(content['topics'])}\n\n"
        
        if content['key_points']:
            response += "💡 **Key Points:**\n"
            for point in content['key_points'][:5]:
                response += f"• {point}\n"
            response += "\n"
        
        response += "💻 **What would you like to see?**\n"
        response += "• Code examples\n• More detailed explanation\n• Related concepts\n• Or ask another question"
        
        return response
    
    def _handle_code_example_query(self, query: str) -> str:
        """Handle code example requests"""
        results = self.semantic_search(query, top_k=1)
        if not results:
            return "🤔 I couldn't find specific C++ code examples for that. Try:\n• Being more specific about the concept\n• Asking about common C++ features\n• Using 'List all C++ topics' to see available concepts"
        
        content = results[0]['cpp_content']
        
        response = f"💻 **Code Examples for: {content['title']}**\n\n"
        
        if content['code_examples']:
            for i, example in enumerate(content['code_examples'][:3], 1):
                response += f"**Example {i}:**\n```cpp\n{example.strip()}\n```\n\n"
        else:
            response += "📝 No specific code examples found in the PDF for this concept.\n\n"
            response += "💡 **Common C++ patterns for this topic:**\n"
            
            # Generate generic examples based on topics
            if any(topic in content['topics'] for topic in ['classes', 'objects']):
                response += """```cpp
class MyClass {
private:
    int data;
public:
    MyClass(int value) : data(value) {}
    void display() { std::cout << data; }
};
```\n"""
            elif 'templates' in content['topics']:
                response += """```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
```\n"""
            elif 'pointers' in content['topics']:
                response += """```cpp
int main() {
    int value = 42;
    int* ptr = &value;
    std::cout << *ptr; // Output: 42
    return 0;
}
```\n"""
        
        response += f"📄 **Source:** {content.get('file_name', 'C++ PDF')}"
        return response
    
    def _handle_explanation_query(self, query: str) -> str:
        """Handle explanation requests"""
        results = self.semantic_search(query, top_k=1)
        if not results:
            return "🤔 I couldn't find a specific C++ concept to explain. Try:\n• Being more specific about what you want explained\n• Using 'List all C++ topics' to see available concepts\n• Asking about common C++ features"
        
        content = results[0]['cpp_content']
        
        response = f"📚 **Detailed Explanation: {content['title']}**\n\n"
        response += f"{content['content'][:1500]}"
        if len(content['content']) > 1500:
            response += "...\n\n💡 *Content truncated. Ask for specific aspects for more details.*"
        response += "\n\n"
        
        if content['concepts']:
            response += f"🔍 **Related Concepts:** {', '.join(content['concepts'])}\n\n"
        
        response += f"📖 **C++ Standard:** {content['cpp_standard']}\n"
        response += f"📊 **Difficulty:** {content['difficulty']}"
        
        return response
    
    def _list_cpp_concepts(self) -> str:
        """List available C++ concepts"""
        if not self.cpp_contents:
            return "📭 No C++ concepts available. Please add C++ PDF files to 'documents/uploaded_pdfs/' folder."
        
        # Group by category/topic
        topics_dict = {}
        for content in self.cpp_contents:
            primary_topic = content['topics'][0] if content['topics'] else 'General C++'
            if primary_topic not in topics_dict:
                topics_dict[primary_topic] = []
            topics_dict[primary_topic].append(content)
        
        response = "📚 **Available C++ Concepts:**\n\n"
        
        for topic, contents in list(topics_dict.items())[:8]:  # Show first 8 topics
            response += f"🔹 **{topic}:**\n"
            for content in contents[:4]:  # Show first 4 concepts per topic
                response += f"• {content['title']}"
                if content['cpp_standard']:
                    response += f" ({content['cpp_standard']})"
                response += "\n"
            response += "\n"
        
        response += f"📖 Total: {len(self.cpp_contents)} concepts across {len(topics_dict)} topics\n\n"
        response += "💬 **Ask about any concept for details!**\n"
        response += "Examples:\n• 'Explain [concept name]'\n• 'Show code for [topic]'\n• 'What is [C++ feature]?'"
        
        return response
    
    def _get_system_status(self) -> str:
        """Get system status"""
        stats = self.pdf_processor.get_stats()
        
        response = "📊 **C++ RAG System Status**\n\n"
        response += f"📚 C++ Concepts: {len(self.cpp_contents)}\n"
        response += f"📄 PDF Files: {stats['unique_files']}\n"
        response += f"🔍 Vector Index: {self.index.ntotal if self.index else 0} embeddings\n"
        response += f"🏷️ Topics: {len(stats['topics_count'])}\n"
        response += f"💡 Concepts: {len(stats['concepts_count'])}\n\n"
        
        # Show most common topics
        if stats['topics_count']:
            response += "📈 **Most Common Topics:**\n"
            for topic, count in list(stats['topics_count'].items())[:6]:
                response += f"• {topic}: {count} concepts\n"
            response += "\n"
        
        # Show C++ standards distribution
        if stats['cpp_standards']:
            response += "⚡ **C++ Standards:**\n"
            for standard, count in stats['cpp_standards'].items():
                response += f"• {standard}: {count} concepts\n"
        
        return response
    
    def _refresh_pdfs(self) -> str:
        """Refresh PDF documents"""
        self.pdf_processor.clear_documents()
        self.cpp_contents = []
        self.load_cpp_contents()
        self.build_vector_index()
        
        stats = self.pdf_processor.get_stats()
        return f"🔄 **PDFs Refreshed!**\n\n📚 C++ Concepts: {len(self.cpp_contents)}\n📄 PDF Files: {stats['unique_files']}\n🔍 Vector Index: {self.index.ntotal if self.index else 0}"
    
    def _handle_semantic_search(self, query: str) -> str:
        """Handle general semantic search"""
        results = self.semantic_search(query, top_k=3)
        if not results:
            return "🤔 I couldn't find any relevant C++ concepts. Try:\n• Using specific C++ terms (classes, templates, pointers)\n• Asking about C++ features or syntax\n• Using 'List all C++ topics' to see available concepts\n• Or ask for help with what I can do"
        
        response = f"🔍 **I found these C++ concepts related to '{query}':**\n\n"
        
        for i, result in enumerate(results, 1):
            content = result['cpp_content']
            response += f"{i}. **{content['title']}** - Relevance: {result['score']:.3f}\n"
            if content['topics']:
                response += f"   Topics: {', '.join(content['topics'][:3])}\n"
            response += "\n"
        
        response += "💡 **Ask about any concept for:**\n• Detailed explanations\n• Code examples\n• Related topics\n• Or ask a follow-up question!"
        
        return response

# Initialize the C++ RAG system
cpp_rag_system = CppRAGSystem()

class ActionConversationalQuery(Action):
    """Single action that handles all user queries about C++"""
    
    def name(self) -> Text:
        return "action_conversational_query"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get('text', '').strip()
        
        if not user_input:
            dispatcher.utter_message(text="👋 Hello! I'm your C++ programming assistant. I can help you understand C++ concepts from your PDF documents!")
            return []
        
        # Process the query using our C++ RAG system
        response = cpp_rag_system.process_query(user_input)
        dispatcher.utter_message(text=response)
        
        return []