# actions.py - COMPLETE FILE
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType
from rasa_sdk.types import DomainDict

import sys
import os
import logging
import time
from datetime import datetime
import re

# Add the actions directory to the path so we can import our rag_pipeline
sys.path.append(os.path.dirname(__file__))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("actions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from rag_pipeline import rag_pipeline
    RAG_AVAILABLE = True
    logger.info("✅ RAG pipeline imported successfully")
    logger.info(f"📊 Knowledge base contains {len(rag_pipeline.documents)} documents")
except ImportError as e:
    logger.error(f"❌ Failed to import RAG pipeline: {e}")
    RAG_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Error initializing RAG pipeline: {e}")
    RAG_AVAILABLE = False


class ActionSessionStart(Action):
    """Action triggered when a new session starts."""
    
    def name(self) -> Text:
        return "action_session_start"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[EventType]:
        
        # Simple welcome message
        welcome_message = "Hello! How can I help you today?"

        dispatcher.utter_message(text=welcome_message)
        
        # Check if RAG system is available
        if not RAG_AVAILABLE:
            dispatcher.utter_message(
                text="⚠️ Note: My knowledge base system is currently unavailable. " \
                     "I'll only be able to answer basic questions."
            )
        else:
            # Show knowledge base status (optional)
            doc_count = len(rag_pipeline.documents)
            if doc_count == 0:
                dispatcher.utter_message(
                    text="💡 Use 'add documents' to learn how to add content to my knowledge base."
                )

        return [SessionStarted(), ActionExecuted("action_listen")]


class ActionSearchKnowledge(Action):
    """Enhanced action for searching the knowledge base using RAG with detailed responses."""
    
    def name(self) -> Text:
        return "action_search_knowledge"

    def _clean_response(self, text):
        """Remove document references from the response text."""
        # Pattern to match "Document X content about..." 
        pattern1 = r'Document\s+\d+\s+content\s+about[^.]*\.\s*'
        # Pattern to match "Document X about..."
        pattern2 = r'Document\s+\d+\s+about[^.]*\.\s*'
        
        # Remove all patterns
        text = re.sub(pattern1, '', text, flags=re.IGNORECASE)
        text = re.sub(pattern2, '', text, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.\s*\.', '.', text)  # Remove duplicate periods
        
        return text

    def run(
        self, 
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        start_time = time.time()
        
        # Check if RAG system is available
        if not RAG_AVAILABLE:
            dispatcher.utter_message(
                text="❌ I'm sorry, but my knowledge search system is currently unavailable. " \
                     "Please make sure the action server is running properly and check the logs for errors."
            )
            return []
        
        # Get user message
        user_message = tracker.latest_message.get('text', '').strip()
        
        if not user_message:
            dispatcher.utter_message(
                text="❌ I didn't receive your question. Please try asking again."
            )
            return []

        # Extract entities or use full message
        question_entity = next(tracker.get_latest_entity_values("question"), None)
        search_query = question_entity or user_message
        
        logger.info(f"🔍 Processing search query: '{search_query}'")
        
        try:
            # Show searching message
            dispatcher.utter_message(text="🔍 Searching my knowledge base for relevant information...")
            
            # Search for relevant information in the knowledge base
            similar_docs = rag_pipeline.search_similar(search_query, n_results=3)
            
            if similar_docs:
                logger.info(f"✅ Found {len(similar_docs)} relevant documents")
                
                # Generate response using RAG
                logger.info("🤖 Generating response...")
                response = rag_pipeline.generate_response(search_query, similar_docs)
                
                # CLEAN THE RESPONSE - Remove document references
                cleaned_response = self._clean_response(response)
                
                # Add source information
                sources = list(set([doc['source'] for doc in similar_docs]))
                source_files = [os.path.basename(src) for src in sources]
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Create the FULL response with CLEANED answer AND sources
                full_response = f"{cleaned_response}\n\n"
                full_response += f"📚 **Sources**: {', '.join(source_files)}\n"
                full_response += f"⏱️ **Processing time**: {processing_time:.2f}s"
                
                # Send the complete response
                dispatcher.utter_message(text=full_response)
                
                logger.info(f"✅ Successfully generated response in {processing_time:.2f}s")
                
            else:
                logger.info(f"❌ No relevant documents found for: '{search_query}'")
                dispatcher.utter_message(
                    text=f"❌ I couldn't find relevant information about '{search_query}' in my knowledge base."
                )
                
        except Exception as e:
            logger.error(f"❌ Error in action_search_knowledge: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text=f"❌ I encountered an error while searching for information about '{search_query}'. Please try again."
            )
        
        return [SlotSet("search_query", search_query), SlotSet("last_search_time", datetime.now().isoformat())]


class ActionAddDocument(Action):
    """Enhanced action to provide detailed information about adding documents."""
    
    def name(self) -> Text:
        return "action_add_document"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        instructions = """
📥 **How to Add Documents to My Knowledge Base**

**Step-by-Step Guide:**

1. **Prepare Your Documents**
   • Supported formats: PDF, TXT, DOCX
   • Place files in: `knowledge_base/documents/` folder

2. **Add Documents**
   • Copy your files to the documents folder
   • Run: `python setup_knowledge_base.py`
   • Restart the action server: `rasa run actions`

3. **Verification**
   • Use: `check knowledge base` to confirm documents were added
   • Test by asking questions about the new content

**Best Practices:**
• Use clear, well-structured documents for best results
• Documents should be text-heavy (not image-based PDFs)
• Ideal document size: 1-50 pages
• Remove sensitive information before adding

**Current Knowledge Base Status:**
"""
        
        # Add current status
        if RAG_AVAILABLE:
            doc_count = len(rag_pipeline.documents)
            instructions += f"• Documents indexed: {doc_count}\n"
            if doc_count == 0:
                instructions += "• ⚠️ No documents currently in knowledge base\n"
        else:
            instructions += "• ❌ Knowledge base unavailable\n"
        
        instructions += "\nReady to expand my knowledge! 🚀"

        dispatcher.utter_message(text=instructions)
        return []


class ActionCheckKnowledgeBase(Action):
    """Enhanced action to check the detailed status of the knowledge base."""
    
    def name(self) -> Text:
        return "action_check_knowledge_base"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(
                text="❌ My knowledge base system is currently unavailable. Please check the action server logs."
            )
            return []
        
        try:
            total_documents = len(rag_pipeline.documents)
            unique_sources = len(set([doc['source'] for doc in rag_pipeline.metadata]))
            
            status_message = "📊 **Knowledge Base Detailed Status**\n\n"
            status_message += f"• **Documents indexed**: {total_documents} chunks\n"
            status_message += f"• **Unique source files**: {unique_sources}\n"
            status_message += f"• **Search system**: ✅ Operational\n"
            status_message += f"• **Response generation**: ✅ Active\n"
            status_message += f"• **Running on**: CPU (Stable)\n"
            status_message += f"• **Last update**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if total_documents > 0:
                # Show some sample sources
                sample_sources = list(set([os.path.basename(doc['source']) for doc in rag_pipeline.metadata[:5]]))
                status_message += f"• **Sample documents**: {', '.join(sample_sources)}\n"
                
                if total_documents > 5:
                    status_message += f"• **And {total_documents - 5} more chunks...**\n"
            else:
                status_message += "\n⚠️ **No documents in knowledge base**\n"
                status_message += "Use 'add documents' to get started and expand my knowledge!"
            
            dispatcher.utter_message(text=status_message)
            
        except Exception as e:
            logger.error(f"❌ Error checking knowledge base: {e}")
            dispatcher.utter_message(
                text="❌ Unable to check knowledge base status at the moment. Please try again later."
            )
        
        return []


class ActionProvideHelp(Action):
    """Enhanced action to provide comprehensive help information."""
    
    def name(self) -> Text:
        return "action_provide_help"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        help_text = """
🤖 **Comprehensive Help Guide**

**How to Use Me Effectively:**

🎯 **Ask Detailed Questions**
• "Explain machine learning algorithms in detail"
• "What are the key principles of project management?"
• "Describe the process of neural network training"
• "Compare and contrast different AI approaches"

📚 **Knowledge Base Management**
• "Check knowledge base" - See detailed status
• "Add documents" - Learn how to expand my knowledge
• "Search for [topic]" - Direct knowledge base search

🔍 **Advanced Usage**
• I can handle complex, multi-part questions
• I provide detailed answers with source references
• I can explain concepts from my knowledge base thoroughly
• I include processing metadata in responses

💡 **Example Questions:**
• "What are the main types of artificial intelligence and their applications?"
• "Explain how deep learning differs from traditional machine learning"
• "Describe the key features of effective leadership according to my documents"

📊 **System Information:**
"""
        
        # Add system status
        if RAG_AVAILABLE:
            doc_count = len(rag_pipeline.documents)
            help_text += f"• Knowledge base: {doc_count} document chunks ready\n"
            help_text += "• Response style: Detailed and comprehensive\n"
            help_text += "• Source citation: Enabled\n"
        else:
            help_text += "• Knowledge base: ❌ Unavailable\n"
        
        help_text += "\nI'm ready to provide detailed, well-sourced answers! 🚀"

        dispatcher.utter_message(text=help_text)
        return []


class ActionFallback(Action):
    """Enhanced fallback action with helpful guidance."""
    
    def name(self) -> Text:
        return "action_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        fallback_text = """
❓ I'm not quite sure what you're asking.

💡 **Here's how I can help you:**

• Ask detailed questions about topics in my knowledge base
• Request explanations of complex concepts
• Search for specific information across my documents
• Check what documents I have available
• Learn how to add more content to my knowledge base

🔍 **Try asking something like:**
• "Explain artificial intelligence in detail"
• "What do you know about machine learning?"
• "Search for information about neural networks"
• "Check knowledge base status"

Or simply tell me what topic you're interested in!"""

        dispatcher.utter_message(text=fallback_text)
        return []


class ActionShowCapabilities(Action):
    """Action to showcase system capabilities."""
    
    def name(self) -> Text:
        return "action_show_capabilities"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        capabilities = """
🚀 **My Enhanced Capabilities**

**Advanced RAG System:**
• 📚 Document understanding and retrieval
• 🤖 AI-powered response generation
• 🔍 Semantic search across knowledge base
• 📊 Source citation and relevance scoring

**What I Can Do:**
• Answer complex, detailed questions
• Provide comprehensive explanations
• Search through multiple documents simultaneously
• Generate well-structured, informative responses
• Handle technical and conceptual questions

**Knowledge Features:**
• Multi-document comprehension
• Context-aware responses
• Detailed source referencing
• Processing time optimization

Ready to tackle your challenging questions! 💪"""

        dispatcher.utter_message(text=capabilities)
        return []


class ActionShowSearching(Action):
    """Action to show that the system is searching."""
    
    def name(self) -> Text:
        return "action_show_searching"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        # This action can be used to show typing indicators in UI integrations
        # For text interface, we handle this in the main search action
        return []
    

class ActionDebugIntent(Action):
    """Debug action to see what intent is being detected"""
    
    def name(self) -> Text:
        return "action_debug_intent"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        latest_intent = tracker.latest_message.get('intent', {}).get('name')
        entities = tracker.latest_message.get('entities', [])
        text = tracker.latest_message.get('text', '')
        
        debug_msg = f"""
🔍 **Debug Information:**
• **User said**: "{text}"
• **Detected intent**: "{latest_intent}"
• **Entities**: {entities}
"""
        
        dispatcher.utter_message(text=debug_msg)
        return []