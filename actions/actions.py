from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType
from rasa_sdk.types import DomainDict

import sys
import os
import logging

# Add the actions directory to the path so we can import our rag_pipeline
sys.path.append(os.path.dirname(__file__))

# Set up logging
logger = logging.getLogger(__name__)

try:
    from rag_pipeline import rag_pipeline
    RAG_AVAILABLE = True
    logger.info("RAG pipeline imported successfully")
except ImportError as e:
    logger.error(f"Failed to import RAG pipeline: {e}")
    RAG_AVAILABLE = False
except Exception as e:
    logger.error(f"Error initializing RAG pipeline: {e}")
    RAG_AVAILABLE = False


class ActionSessionStart(Action):
    """Action triggered when a new session starts."""
    
    def name(self) -> Text:
        return "action_session_start"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: DomainDict
    ) -> List[EventType]:
        
        # Send welcome message
        dispatcher.utter_message(
            text="Hello! I'm your AI assistant with access to a knowledge base. " \
                 "You can ask me questions about the documents in my knowledge base."
        )
        
        # Check if RAG system is available
        if not RAG_AVAILABLE:
            dispatcher.utter_message(
                text="Note: My knowledge base system is currently unavailable. " \
                     "I'll only be able to answer basic questions."
            )

        return [SessionStarted(), ActionExecuted("action_listen")]


class ActionSearchKnowledge(Action):
    """Main action for searching the knowledge base using RAG."""
    
    def name(self) -> Text:
        return "action_search_knowledge"

    def run(
        self, 
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        # Check if RAG system is available
        if not RAG_AVAILABLE:
            dispatcher.utter_message(
                text="I'm sorry, but my knowledge search system is currently unavailable. " \
                     "Please make sure the action server is running properly."
            )
            return []
        
        # Get user message
        user_message = tracker.latest_message.get('text', '').strip()
        
        if not user_message:
            dispatcher.utter_message(
                text="I didn't receive your question. Please try asking again."
            )
            return []

        # Extract entities or use full message
        question_entity = next(tracker.get_latest_entity_values("question"), None)
        search_query = question_entity or user_message
        
        logger.info(f"Processing search query: {search_query}")
        
        try:
            # Show typing indicator (simulated)
            dispatcher.utter_message(response="utter_searching")
            
            # Search for relevant information in the knowledge base
            similar_docs = rag_pipeline.search_similar(search_query, n_results=3)
            
            if similar_docs:
                # Generate response using RAG
                logger.info(f"Found {len(similar_docs)} relevant documents, generating response...")
                response = rag_pipeline.generate_response(search_query, similar_docs)
                
                # Add source information
                sources = list(set([doc['source'] for doc in similar_docs]))
                source_info = f"\n\n📚 Sources: {', '.join([os.path.basename(src) for src in sources])}"
                
                full_response = response + source_info
                dispatcher.utter_message(text=full_response)
                
            else:
                logger.info("No relevant documents found")
                dispatcher.utter_message(
                    text="I couldn't find relevant information in my knowledge base to answer your question. " \
                         "Please try:\n" \
                         "• Rephrasing your question\n" \
                         "• Using different keywords\n" \
                         "• Adding more relevant documents to the knowledge base"
                )
                
        except Exception as e:
            logger.error(f"Error in action_search_knowledge: {str(e)}", exc_info=True)
            dispatcher.utter_message(
                text="I encountered an error while searching for information. " \
                     "Please try again in a moment."
            )
        
        return [SlotSet("search_query", search_query)]


class ActionAddDocument(Action):
    """Action to provide information about adding documents to the knowledge base."""
    
    def name(self) -> Text:
        return "action_add_document"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        instructions = """
To add documents to my knowledge base:

1. Place your documents (PDF, TXT, or DOCX files) in the 'knowledge_base/documents/' folder
2. Run the setup script: `python setup_knowledge_base.py`
3. Restart the action server

Supported formats:
• PDF files (.pdf)
• Text files (.txt) 
• Word documents (.docx)

After adding documents, I'll be able to answer questions based on their content!
"""

        dispatcher.utter_message(text=instructions)
        return []


class ActionCheckKnowledgeBase(Action):
    """Action to check the status of the knowledge base."""
    
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
                text="My knowledge base system is currently unavailable."
            )
            return []
        
        try:
            total_documents = len(rag_pipeline.documents)
            status_message = f"📊 Knowledge Base Status:\n"
            status_message += f"• Documents indexed: {total_documents}\n"
            status_message += f"• Search system: ✅ Operational\n"
            status_message += f"• LLM: ✅ Ready\n"
            
            if total_documents == 0:
                status_message += "\n⚠️ No documents in knowledge base. Use 'add documents' to get started."
            
            dispatcher.utter_message(text=status_message)
            
        except Exception as e:
            logger.error(f"Error checking knowledge base: {e}")
            dispatcher.utter_message(
                text="Unable to check knowledge base status at the moment."
            )
        
        return []


class ActionFallback(Action):
    """Fallback action when the user's intent isn't clear."""
    
    def name(self) -> Text:
        return "action_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(
            text="I'm not sure I understand. You can ask me questions about the documents in my knowledge base, " \
                 "or try rephrasing your question."
        )
        return []


class ActionProvideHelp(Action):
    """Action to provide help information to the user."""
    
    def name(self) -> Text:
        return "action_provide_help"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        help_text = """
🤖 How to use me:

**Ask Questions**: I can answer questions based on the documents in my knowledge base
• "What is machine learning?"
• "Explain neural networks"
• "Tell me about artificial intelligence"

**Knowledge Base Management**:
• "Check knowledge base" - See current status
• "Add documents" - Learn how to add more documents

**Examples**:
• "What are the main types of AI?"
• "Can you explain deep learning?"
• "Search for information about transformers"

Just ask me anything about the topics in my knowledge base!
"""

        dispatcher.utter_message(text=help_text)
        return []


class ActionSearchingIndicator(Action):
    """Action to show that the system is searching (typing indicator)."""
    
    def name(self) -> Text:
        return "action_show_searching"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        # This would typically trigger a typing indicator in UI integrations
        # For text interface, we can send a temporary message or just wait
        return []