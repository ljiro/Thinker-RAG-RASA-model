# actions.py - COMPLETE CORPORATE-ONLY VERSION
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType
from datetime import datetime
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

# Import RAG pipeline - CORPORATE VERSION
try:
    # Try relative import first (when running as package)
    from .rag_pipeline import rag_pipeline, generate_answer, retrieve_context
    RAG_AVAILABLE = True
    logger.info("✅ Corporate RAG Pipeline imported successfully via relative import")
except ImportError:
    try:
        # Fallback to direct import (when running standalone)
        from rag_pipeline import rag_pipeline, generate_answer, retrieve_context
        RAG_AVAILABLE = True
        logger.info("✅ Corporate RAG Pipeline imported successfully via direct import")
    except ImportError as e:
        RAG_AVAILABLE = False
        logger.error(f"❌ Failed to import corporate RAG pipeline: {e}")

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[EventType]:
        
        events = [SessionStarted()]
        
        if len(tracker.events) <= 3:
            # Don't auto-greet, let conversation start naturally
            events.append(ActionExecuted("action_listen"))
        
        return events

class ActionCorporateQuery(Action):
    def name(self) -> Text:
        return "action_corporate_query"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        search_query = tracker.latest_message.get('text', '')
        logger.info(f"🏢 Corporate Query: '{search_query}'")
        
        if not search_query:
            dispatcher.utter_message(text="Please ask a question about Gift of Grace Food Manufacturing Corporation.")
            return []
        
        logger.info(f"🔍 Processing corporate query: {search_query}")
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(text="The corporate knowledge base is currently unavailable.")
            return []
        
        try:
            # Generate answer about the company
            answer = generate_answer(search_query)
            
            if answer:
                logger.info(f"📝 Corporate Response: {len(answer)} characters")
                
                # Add company introduction if needed
                if "Gift of Grace" not in answer and "GoGFMC" not in answer:
                    intro = "Based on the corporate report of Gift of Grace Food Manufacturing Corporation:\n\n"
                    answer = intro + answer
                
                # Send as single message
                dispatcher.utter_message(text=answer)
                
                logger.info("✅ Corporate response sent")
            else:
                dispatcher.utter_message(
                    text="I couldn't find specific information about that in the Gift of Grace corporate report. "
                         "Please try asking about:\n\n"
                         "• Their products (kimchi, tofu, rice coffee)\n"
                         "• Company history and founders\n"
                         "• Awards and certifications\n"
                         "• Contact information\n"
                         "• Manufacturing standards and CSR programs"
                )
                
        except Exception as e:
            logger.error(f"❌ Error in corporate query: {e}")
            dispatcher.utter_message(
                text="I encountered an error while searching the corporate database. Please try again with a different question."
            )
        
        return [SlotSet("last_search_time", datetime.now().isoformat())]

class ActionSearchKnowledge(Action):
    def name(self) -> Text:
        return "action_search_knowledge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        search_query = tracker.latest_message.get('text', '').lower()
        logger.info(f"🔍 ActionSearchKnowledge processing: '{search_query}'")
        
        # ALL queries go to corporate query now
        logger.info("🏢 Redirecting ALL queries to corporate query...")
        return ActionCorporateQuery().run(dispatcher, tracker, domain)

class ActionHandleProducts(Action):
    def name(self) -> Text:
        return "action_handle_products"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("🛒 Handling products query")
        return ActionCorporateQuery().run(dispatcher, tracker, domain)

class ActionHandleAwards(Action):
    def name(self) -> Text:
        return "action_handle_awards"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("🏆 Handling awards query")
        return ActionCorporateQuery().run(dispatcher, tracker, domain)

class ActionHandleContact(Action):
    def name(self) -> Text:
        return "action_handle_contact"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("📞 Handling contact query")
        return ActionCorporateQuery().run(dispatcher, tracker, domain)

class ActionHandleFounders(Action):
    def name(self) -> Text:
        return "action_handle_founders"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("👥 Handling founders query")
        return ActionCorporateQuery().run(dispatcher, tracker, domain)

class ActionCorporateIntro(Action):
    def name(self) -> Text:
        return "action_corporate_intro"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        intro_text = (
            "🏢 **Welcome to Gift of Grace Food Manufacturing Corporation Assistant!**\n\n"
            "I'm here to provide you with comprehensive information about Gift of Grace Food Manufacturing Corporation (GoGFMC).\n\n"
            "**About the Company:**\n"
            "• Filipino food manufacturing company based in Baguio City\n"
            "• Founded by Satur Cadsi (CEO) and Janice Osenio Cadsi (COO)\n"
            "• Specializes in healthy, innovative food products\n"
            "• Award-winning MSME with Halal certification\n\n"
            "💡 **You can ask me about:**\n"
            "- Company history and founders\n"
            "- Products (kimchi, tofu, rice coffee)\n"
            "- Awards and achievements\n"
            "- Manufacturing standards\n"
            "- Contact information\n"
            "- And much more!\n\n"
            "Try asking: \"Tell me about Gift of Grace Food Manufacturing\" or say 'help' for guidance."
        )
        
        dispatcher.utter_message(text=intro_text)
        return []

class ActionProvideHelp(Action):
    def name(self) -> Text:
        return "action_provide_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        help_text = (
            "🏢 **Gift of Grace Food Manufacturing Corporation Assistant**\n\n"
            "I'm your dedicated assistant for information about Gift of Grace Food Manufacturing Corporation (GoGFMC).\n\n"
            "📋 **What I Can Help You With:**\n\n"
            "**Company Information:**\n"
            "• Company history and origins\n"
            "• Founders: Satur Cadsi (CEO) and Janice Osenio Cadsi (COO)\n"
            "• Mission, vision, and core values\n"
            "• Awards and industry recognition\n"
            "• Corporate social responsibility programs\n\n"
            "**Products:**\n"
            "• Kimchi Gift (flagship product)\n"
            "• Tofu Gift (non-GMO soybean curd)\n"
            "• Rice Coffee with Moringa\n"
            "• Partner products and diversification\n\n"
            "**Operations & Standards:**\n"
            "• Manufacturing standards\n"
            "• Halal certification\n"
            "• Technology adoption (DOST-SETUP)\n"
            "• Digital transformation\n\n"
            "**Market Presence:**\n"
            "• Retail network and distribution\n"
            "• Contact information\n"
            "• Digital footprint\n\n"
            "💡 **Try Asking:**\n"
            "- \"Tell me about Gift of Grace Food Manufacturing\"\n"
            "- \"What products do they make?\"\n"
            "- \"Who are the founders?\"\n"
            "- \"What awards have they won?\"\n"
            "- \"Where is Gift of Grace located?\"\n"
            "- \"What is their Halal certification?\"\n"
            "- \"How can I contact them?\"\n"
            "- \"Tell me about their kimchi product\""
        )
        
        dispatcher.utter_message(text=help_text)
        return []

class ActionFallback(Action):
    def name(self) -> Text:
        return "action_fallback"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        message = tracker.latest_message.get('text', '').lower()
        
        # All questions go to corporate query
        if RAG_AVAILABLE:
            return ActionCorporateQuery().run(dispatcher, tracker, domain)
        
        # Default fallback response
        dispatcher.utter_message(
            text="I'm your assistant for Gift of Grace Food Manufacturing Corporation information.\n\n"
                 "🏢 **I can help you with:**\n"
                 "• Company history and information\n"
                 "• Product details (kimchi, tofu, rice coffee)\n"
                 "• Awards and certifications\n"
                 "• Contact information\n"
                 "• Manufacturing standards\n\n"
                 "Try asking specific questions about Gift of Grace or say 'help' for guidance."
        )
        return []

class ActionShowCapabilities(Action):
    def name(self) -> Text:
        return "action_show_capabilities"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        capabilities = (
            "🏢 **Gift of Grace Food Manufacturing Corporation Assistant**\n\n"
            "🚀 **My Capabilities:**\n\n"
            "📊 **Comprehensive Corporate Knowledge:**\n"
            "- Access to full 2025 corporate report\n"
            "- Company history and origin story\n"
            "- Leadership team information\n"
            "- Strategic goals and vision\n\n"
            "🛒 **Product Information:**\n"
            "- Kimchi Gift (K-Fil Fusion product)\n"
            "- Tofu Gift (non-GMO, multiple textures)\n"
            "- Rice Coffee with Moringa (caffeine-free)\n"
            "- Partner products and diversification\n\n"
            "🏆 **Achievements & Standards:**\n"
            "- Awards: Presidential Award finalist, Inspiring Filipina Entrepreneur\n"
            "- Halal certification details\n"
            "- DOST-SETUP technology adoption\n"
            "- Manufacturing and quality standards\n\n"
            "🌱 **Social Responsibility:**\n"
            "- Community livelihood programs\n"
            "- Sustainability and circular economy\n"
            "- Educational scholarships\n"
            "- Social inclusion initiatives\n\n"
            "📞 **Contact & Distribution:**\n"
            "- Company location and address\n"
            "- Retail network information\n"
            "- Digital presence and contact channels\n\n"
            "💡 **Example Questions:**\n"
            "- \"Tell me about Gift of Grace Food Manufacturing\"\n"
            "- \"What products do they make?\"\n"
            "- \"Who are the founders?\"\n"
            "- \"What awards have they won?\"\n"
            "- \"Where is Gift of Grace located?\"\n"
            "- \"What is their Halal certification?\"\n"
            "- \"How can I contact them?\"\n"
            "- \"Tell me about their kimchi product\""
        )
        
        dispatcher.utter_message(text=capabilities)
        return []

class ActionCheckKnowledgeBase(Action):
    def name(self) -> Text:
        return "action_check_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        status_parts = ["🏢 **Corporate Knowledge Base Status:**"]
        
        if RAG_AVAILABLE:
            try:
                stats = rag_pipeline.get_stats()
                status_parts.append(f"• Documents: {stats.get('total_documents', 0)}")
                status_parts.append(f"• Company: {stats.get('company', 'Unknown')}")
                status_parts.append(f"• Document Type: {stats.get('document_type', 'N/A')}")
                status_parts.append(f"• Data Available: {'✅ Yes' if stats.get('has_data') else '❌ No'}")
                status_parts.append(f"• Embedding Model: {stats.get('embedding_model', 'N/A')}")
                status_parts.append(f"• LLM Model: {stats.get('llm_model', 'N/A')}")
            except Exception as e:
                status_parts.append(f"• Status: ⚠️ Error: {e}")
        else:
            status_parts.append("• System: ❌ Not available")
        
        status_parts.extend([
            "",
            "📋 **Document Information:**",
            "• Gift of Grace Food Manufacturing Corporation Report 2025",
            "• Comprehensive corporate profile",
            "• 8-page detailed document",
            "",
            "💡 **To update knowledge base:**",
            "• Add PDF files to 'knowledge_base/documents/'",
            "• Run: python setup_ordinances.py", 
            "• Restart actions server"
        ])
        
        dispatcher.utter_message(text="\n".join(status_parts))
        return []

class ActionAddDocument(Action):
    def name(self) -> Text:
        return "action_add_document"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        help_text = (
            "📁 **Adding Corporate Documents to Knowledge Base:**\n\n"
            "To add corporate documents about Gift of Grace Food Manufacturing Corporation:\n\n"
            "1. **Place PDF files** in the 'knowledge_base/documents/' folder\n"
            "2. **Run the setup script**: python setup_ordinances.py\n"
            "3. **Restart** the Rasa actions server\n\n"
            "The system will automatically process the documents and make them searchable.\n\n"
            "🏢 **Current Document:**\n"
            "- Aragona et al. Gift of Grace MOA gemini ai.pdf\n"
            "- Gift of Grace Food Manufacturing Corporation Report 2025\n"
            "- Comprehensive 8-page corporate profile"
        )
        
        dispatcher.utter_message(text=help_text)
        return []

class ActionDebugCorporateResponse(Action):
    def name(self) -> Text:
        return "action_debug_corporate_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(text="Corporate RAG system is not available.")
            return []
        
        try:
            # Test corporate-specific queries
            test_queries = [
                "Tell me about Gift of Grace Food Manufacturing Corporation",
                "What products does Gift of Grace make?",
                "Who are the founders of Gift of Grace?",
                "What awards has Gift of Grace won?",
                "Where is Gift of Grace located?",
                "What is their Halal certification?",
                "How can I contact Gift of Grace?"
            ]
            
            for query in test_queries:
                response = generate_answer(query)
                logger.info(f"🔍 Corporate RAG Response for '{query}': {len(response)} characters")
                
                dispatcher.utter_message(text=f"**Test Query:** {query}")
                dispatcher.utter_message(text=f"**Response:**\n{response}")
                dispatcher.utter_message(text="---" * 10)
                
        except Exception as e:
            logger.error(f"❌ Corporate debug error: {e}")
            dispatcher.utter_message(text=f"Debug error: {e}")
        
        return []

class ActionDebugSystemStatus(Action):
    def name(self) -> Text:
        return "action_debug_system_status"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        status_parts = ["🔧 **Corporate System Debug Status:**"]
        
        if RAG_AVAILABLE:
            try:
                stats = rag_pipeline.get_stats()
                status_parts.append(f"• System: ✅ Operational")
                status_parts.append(f"• Company: {stats.get('company', 'Unknown')}")
                status_parts.append(f"• Total Chunks: {stats.get('total_documents', 0)}")
                status_parts.append(f"• Has Data: {'✅ Yes' if stats.get('has_data') else '❌ No'}")
                status_parts.append(f"• Embedding Model: {stats.get('embedding_model', 'N/A')}")
                status_parts.append(f"• LLM Model: {stats.get('llm_model', 'N/A')}")
            except Exception as e:
                status_parts.append(f"• System: ⚠️ Error: {e}")
        else:
            status_parts.append("• System: ❌ RAG not available")
        
        # Test a query
        if RAG_AVAILABLE:
            try:
                test_query = "What is Gift of Grace Food Manufacturing Corporation?"
                response = generate_answer(test_query)
                status_parts.append(f"\n• Test Query Response: ✅ {len(response)} characters")
                status_parts.append(f"• Response Preview: {response[:100]}...")
            except Exception as e:
                status_parts.append(f"\n• Test Query: ❌ Failed: {e}")
        
        dispatcher.utter_message(text="\n".join(status_parts))
        return []

# Standard response actions
class ActionUtterGoodbye(Action):
    def name(self) -> Text:
        return "utter_goodbye"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="Goodbye! Feel free to ask if you have more questions about Gift of Grace Food Manufacturing Corporation.")
        return []

class ActionUtterThankYou(Action):
    def name(self) -> Text:
        return "utter_thankyou"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="You're welcome! Happy to help with any questions about Gift of Grace Food Manufacturing Corporation.")
        return []

class ActionUtterIamABot(Action):
    def name(self) -> Text:
        return "utter_iamabot"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="I am an AI assistant specialized in providing information about Gift of Grace Food Manufacturing Corporation.")
        return []

class ActionUtterOutOfScope(Action):
    def name(self) -> Text:
        return "utter_out_of_scope"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(
            text="I'm specifically designed to answer questions about Gift of Grace Food Manufacturing Corporation. "
                 "Please ask about the company, its products, history, or related topics."
        )
        return []

class ActionUtterHelp(Action):
    def name(self) -> Text:
        return "utter_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(
            text="I can answer questions about Gift of Grace Food Manufacturing Corporation. "
                 "Try asking about their products, history, awards, or contact information!"
        )
        return []

class ActionUtterDefault(Action):
    def name(self) -> Text:
        return "utter_default"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(
            text="I'm not sure I understand. You can ask me questions about Gift of Grace Food Manufacturing Corporation or say 'help' to see what I can do."
        )
        return []

class ActionUtterSearching(Action):
    def name(self) -> Text:
        return "utter_searching"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(text="🔍 Searching my knowledge base for information about Gift of Grace...")
        return []

# Additional utility actions
class ActionProcessGreeting(Action):
    def name(self) -> Text:
        return "action_process_greeting"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        greeting = (
            "Hello! 👋\n\n"
            "I'm your assistant for Gift of Grace Food Manufacturing Corporation.\n\n"
            "I can help you with information about:\n"
            "• Company history and founders\n"
            "• Products and manufacturing\n"
            "• Awards and certifications\n"
            "• Contact information\n\n"
            "How can I help you today?"
        )
        
        dispatcher.utter_message(text=greeting)
        return []

class ActionRestartConversation(Action):
    def name(self) -> Text:
        return "action_restart_conversation"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        restart_message = (
            "🔄 Conversation restarted!\n\n"
            "I'm your assistant for Gift of Grace Food Manufacturing Corporation.\n\n"
            "What would you like to know about the company?"
        )
        
        dispatcher.utter_message(text=restart_message)
        
        # Clear any slots if needed
        return [
            SlotSet("search_query", None),
            SlotSet("last_search_time", None)
        ]

class ActionTestRAGConnection(Action):
    def name(self) -> Text:
        return "action_test_rag_connection"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(text="❌ RAG system is not available.")
            return []
        
        try:
            # Simple test query
            test_query = "What is Gift of Grace Food Manufacturing Corporation?"
            response = generate_answer(test_query)
            
            if response:
                dispatcher.utter_message(
                    text=f"✅ RAG System Test Successful!\n\n"
                         f"**Query:** {test_query}\n\n"
                         f"**Response Preview:**\n{response[:200]}...\n\n"
                         f"**Response Length:** {len(response)} characters"
                )
            else:
                dispatcher.utter_message(text="⚠️ RAG system responded but with empty response.")
                
        except Exception as e:
            dispatcher.utter_message(text=f"❌ RAG Test Failed: {e}")
        
        return []

class ActionShowCompanySummary(Action):
    def name(self) -> Text:
        return "action_show_company_summary"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        summary = (
            "🏢 **Gift of Grace Food Manufacturing Corporation (GoGFMC) - Quick Summary**\n\n"
            "**Company Overview:**\n"
            "• Filipino food manufacturing company based in Baguio City\n"
            "• Founded by Satur Cadsi (CEO) and Janice Osenio Cadsi (COO)\n"
            "• Started as a home-based kimchi business in 2015/2017\n"
            "• Now a thriving corporation with Halal certification\n\n"
            "**Core Products:**\n"
            "• Kimchi Gift - K-Fil Fusion kimchi\n"
            "• Tofu Gift - Non-GMO soybean curd\n"
            "• Rice Coffee with Moringa - Caffeine-free beverage\n\n"
            "**Key Achievements:**\n"
            "• Presidential Award for Outstanding MSMEs Finalist (2025)\n"
            "• Inspiring Filipina Entrepreneur Award (Janice Cadsi, 2025)\n"
            "• Regional Best SETUP Adoptor (DOST-CAR, 2025)\n\n"
            "**Location:**\n"
            "#5 Purok 6, Pinsao Pilot Project, Baguio City 2600, Benguet\n\n"
            "Ask me for more details about any of these topics!"
        )
        
        dispatcher.utter_message(text=summary)
        return []