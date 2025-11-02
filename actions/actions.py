# actions.py
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType
from datetime import datetime
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

# Import RAG pipeline - FIXED IMPORTS
try:
    # Try relative import first (when running as package)
    from .rag_pipeline import rag_pipeline, generate_answer, retrieve_context
    RAG_AVAILABLE = True
    logger.info("✅ RAG Pipeline imported successfully via relative import")
except ImportError:
    try:
        # Fallback to direct import (when running standalone)
        from rag_pipeline import rag_pipeline, generate_answer, retrieve_context
        RAG_AVAILABLE = True
        logger.info("✅ RAG Pipeline imported successfully via direct import")
    except ImportError as e:
        RAG_AVAILABLE = False
        logger.error(f"❌ Failed to import RAG pipeline: {e}")

class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[EventType]:
        
        events = [SessionStarted()]
        
        if len(tracker.events) <= 3:
            events.append(ActionExecuted("action_listen"))
        
        return events

class ActionSearchKnowledge(Action):
    def name(self) -> Text:
        return "action_search_knowledge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # DEBUG: Check if multiple actions are being called
        recent_events = [e for e in tracker.events if e.get('event') == 'action']
        recent_action_names = [e.get('name') for e in recent_events[-5:]]  # Last 5 actions
        logger.info(f"🔍 Recent actions in tracker: {recent_action_names}")
        
        search_query = tracker.latest_message.get('text', '')
        logger.info(f"🔍 ActionSearchKnowledge processing: '{search_query}'")
        
        if not search_query:
            dispatcher.utter_message(text="Please ask a question about Baguio City ordinances.")
            return []
        
        logger.info(f"🔍 Processing search query: {search_query}")
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(text="The knowledge base is currently unavailable. Please try again later.")
            return []
        
        try:
            # Generate answer - this returns the complete formatted response
            answer = generate_answer(search_query)
            
            if answer:
                # DEBUG: Log the response to see what we're getting
                logger.info(f"📝 RAG Response received, length: {len(answer)} characters")
                
                # Send as ONE SINGLE MESSAGE - preserve all formatting including bullet points
                dispatcher.utter_message(text=answer)
                
                # DEBUG: Confirm message was sent
                logger.info("✅ Response sent as single message")
            else:
                dispatcher.utter_message(
                    text="I couldn't find specific information about that in the Baguio City ordinances. "
                         "Please try rephrasing your question."
                )
                
        except Exception as e:
            logger.error(f"❌ Error in action_search_knowledge: {e}")
            dispatcher.utter_message(
                text="I encountered an error while searching. Please try again with a different question."
            )
        
        return [SlotSet("last_search_time", datetime.now().isoformat())]

class ActionSearchOrdinances(Action):
    def name(self) -> Text:
        return "action_search_ordinances"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        question = tracker.latest_message.get('text', '')
        
        if not question:
            dispatcher.utter_message(text="Please ask a question about Baguio City ordinances.")
            return []
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(text="The ordinance system is currently unavailable.")
            return []
        
        try:
            # This should return ONE complete response with formatting
            answer = generate_answer(question)
            
            if answer:
                logger.info(f"📝 Ordinance search response: {len(answer)} characters")
                dispatcher.utter_message(text=answer)
            else:
                dispatcher.utter_message(
                    text="I couldn't find specific information about that in the ordinances database."
                )
            
        except Exception as e:
            logger.error(f"❌ Error searching ordinances: {e}")
            dispatcher.utter_message(
                text="I encountered an error while searching ordinances. Please try again."
            )
        
        return [SlotSet("last_search_time", datetime.now().isoformat())]

class ActionOrdinanceHelp(Action):
    def name(self) -> Text:
        return "action_ordinance_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        help_text = (
            "🔍 **Ordinance Search Help:**\n\n"
            "I can help you search through Baguio City ordinances for:\n"
            "• Traffic rules and regulations\n"
            "• Business permit information\n"
            "• Local laws and guidelines\n"
            "• City procedures and requirements\n\n"
            "**Examples:**\n"
            "- \"What are the traffic rules in Baguio?\"\n"
            "- \"How do business permits work?\"\n"
            "- \"Information about local regulations\"\n"
            "- \"Search for city guidelines\""
        )
        
        dispatcher.utter_message(text=help_text)
        return []

class ActionCheckOrdinanceSystem(Action):
    def name(self) -> Text:
        return "action_check_ordinance_system"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        status_parts = ["🔧 **System Status:**"]
        
        if RAG_AVAILABLE:
            try:
                stats = rag_pipeline.get_stats()
                status_parts.append(f"• RAG System: ✅ Available ({stats.get('total_documents', 0)} documents)")
                status_parts.append(f"• Has Data: {'✅ Yes' if stats.get('has_data') else '❌ No'}")
            except:
                status_parts.append("• RAG System: ⚠️ Available but stats unavailable")
        else:
            status_parts.append("• RAG System: ❌ Unavailable")
        
        status_parts.extend([
            "",
            "💡 **What I can help with:**",
            "• Search Baguio City ordinances and laws",
            "• Answer questions from knowledge base", 
            "• Provide general guidance",
            "• Traffic and regulation information"
        ])
        
        dispatcher.utter_message(text="\n".join(status_parts))
        return []

class ActionProvideHelp(Action):
    def name(self) -> Text:
        return "action_provide_help"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        help_text = (
            "🤖 **How I Can Help You:**\n\n"
            "I'm your assistant for Baguio City information with access to:\n\n"
            "🔍 **Ordinance Knowledge Base:**\n"
            "- Answer questions about Baguio City ordinances\n"
            "- Provide information from legal documents\n"
            "- Help with general inquiries\n"
            "- Traffic and local information\n\n"
            "**Try asking:**\n"
            "- \"What are the traffic rules in Baguio?\"\n"
            "- \"Information about local businesses\"\n" 
            "- \"Baguio city guidelines\"\n"
            "- \"Search for regulations\"\n"
            "- \"Check system status\""
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
        
        # Check if it's a general question that RAG can handle
        question_words = ['what', 'how', 'when', 'where', 'why', 'can you', 'tell me', 'search']
        if any(word in message for word in question_words) and RAG_AVAILABLE:
            return ActionSearchKnowledge().run(dispatcher, tracker, domain)
        
        # Default fallback response
        dispatcher.utter_message(
            text="I'm not quite sure what you're asking. I can help you with:\n"
                 "• Baguio City ordinances and laws\n"
                 "• General questions about Baguio\n"
                 "• Local guidelines and information\n\n"
                 "Try asking about specific topics or say 'help' for more guidance."
        )
        return []

class ActionShowCapabilities(Action):
    def name(self) -> Text:
        return "action_show_capabilities"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        capabilities = (
            "🚀 **My Capabilities:**\n\n"
            "🔍 **Ordinance Knowledge Base:**\n"
            "- Search through Baguio City Code of Ordinances\n"
            "- Answer questions based on legal documents\n"
            "- Provide specific regulation information\n"
            "- Reference official ordinance sections\n\n"
            "💡 **Try these examples:**\n"
            "- \"What are the traffic rules in Baguio?\"\n"
            "- \"How do I get a business permit?\"\n"
            "- \"What are the penalties for littering?\"\n"
            "- \"Search ordinances about noise control\""
        )
        
        dispatcher.utter_message(text=capabilities)
        return []

class ActionAddDocument(Action):
    def name(self) -> Text:
        return "action_add_document"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        help_text = (
            "📁 **Adding Documents to Knowledge Base:**\n\n"
            "To add documents to the system:\n\n"
            "1. **Place PDF files** in the 'knowledge_base/documents/' folder\n"
            "2. **Run the setup script**: python setup_ordinances.py\n"
            "3. **Restart** the Rasa actions server\n\n"
            "The system will automatically process the documents and make them searchable."
        )
        
        dispatcher.utter_message(text=help_text)
        return []

class ActionCheckKnowledgeBase(Action):
    def name(self) -> Text:
        return "action_check_knowledge_base"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        status_parts = ["📚 **Knowledge Base Status:**"]
        
        if RAG_AVAILABLE:
            try:
                stats = rag_pipeline.get_stats()
                status_parts.append(f"• Documents: {stats.get('total_documents', 0)}")
                status_parts.append(f"• Data Available: {'✅ Yes' if stats.get('has_data') else '❌ No'}")
                status_parts.append(f"• Embedding Model: {stats.get('embedding_model', 'N/A')}")
                status_parts.append(f"• LLM Model: {stats.get('llm_model', 'N/A')}")
            except Exception as e:
                status_parts.append(f"• Status: ⚠️ Error: {e}")
        else:
            status_parts.append("• System: ❌ Not available")
        
        status_parts.extend([
            "",
            "💡 **To update knowledge base:**",
            "• Add PDF files to 'knowledge_base/documents/'",
            "• Run: python setup_ordinances.py", 
            "• Restart actions server"
        ])
        
        dispatcher.utter_message(text="\n".join(status_parts))
        return []

class ActionDebugIntent(Action):
    def name(self) -> Text:
        return "action_debug_intent"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        latest_message = tracker.latest_message
        intent = latest_message.get('intent', {}).get('name', 'None')
        entities = latest_message.get('entities', [])
        text = latest_message.get('text', '')
        
        debug_info = (
            f"🔍 **Debug Information:**\n"
            f"• Intent: {intent}\n"
            f"• Text: '{text}'\n"
            f"• Entities: {entities}\n"
            f"• RAG System: {'✅ Available' if RAG_AVAILABLE else '❌ Unavailable'}"
        )
        
        dispatcher.utter_message(text=debug_info)
        return []

class ActionDebugResponse(Action):
    def name(self) -> Text:
        return "action_debug_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Test the RAG response directly
        test_query = "What are the fines for traffic violations in Baguio?"
        
        if RAG_AVAILABLE:
            try:
                response = generate_answer(test_query)
                logger.info(f"🔍 DEBUG RAG Response Type: {type(response)}")
                logger.info(f"🔍 DEBUG RAG Response Length: {len(response)} characters")
                logger.info(f"🔍 DEBUG RAG Response Preview: {response[:200]}...")
                
                # Count newlines to see structure
                newline_count = response.count('\n')
                logger.info(f"🔍 DEBUG Newlines in response: {newline_count}")
                
                # Count bullet points
                bullet_count = response.count('•') + response.count('-') + response.count('*')
                logger.info(f"🔍 DEBUG Bullet points in response: {bullet_count}")
                
                dispatcher.utter_message(
                    text=f"🔍 **Debug Response Analysis:**\n"
                         f"• Response length: {len(response)} characters\n"
                         f"• Newlines: {newline_count}\n"
                         f"• Bullet points: {bullet_count}\n"
                         f"• RAG Available: ✅ Yes"
                )
                
                # Send the actual response
                dispatcher.utter_message(text=response)
                
            except Exception as e:
                logger.error(f"❌ Debug error: {e}")
                dispatcher.utter_message(text=f"Debug error: {e}")
        else:
            dispatcher.utter_message(text="RAG not available for debugging")
        
        return []

class ActionTestRAGResponse(Action):
    def name(self) -> Text:
        return "action_test_rag_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(text="RAG system is not available for testing.")
            return []
        
        try:
            # Test with a known query that should return formatted response
            test_queries = [
                "What are the traffic rules in Baguio?",
                "Tell me about business permits in Baguio City",
                "What are the fines for violations?"
            ]
            
            for query in test_queries:
                response = generate_answer(query)
                dispatcher.utter_message(text=f"**Query:** {query}")
                dispatcher.utter_message(text=f"**Response:** {response}")
                dispatcher.utter_message(text="---")
                
        except Exception as e:
            logger.error(f"❌ RAG test error: {e}")
            dispatcher.utter_message(text=f"Test error: {e}")
        
        return []

class ActionDebugMessageFlow(Action):
    def name(self) -> Text:
        return "action_debug_message_flow"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Test the exact flow
        test_query = "What are the traffic rules in Baguio?"
        
        if RAG_AVAILABLE:
            try:
                answer = generate_answer(test_query)
                logger.info(f"🔍 RAG response type: {type(answer)}")
                logger.info(f"🔍 RAG response length: {len(answer)}")
                
                # Send as ONE message with clear markers
                debug_response = f"🚨 DEBUG SINGLE MESSAGE 🚨\n\n{answer}\n\n🚨 END SINGLE MESSAGE 🚨"
                dispatcher.utter_message(text=debug_response)
                
                logger.info("✅ Sent as single debug message")
                
            except Exception as e:
                logger.error(f"❌ Debug error: {e}")
                dispatcher.utter_message(text=f"Debug error: {e}")
        else:
            dispatcher.utter_message(text="RAG not available")
        
        return []

class ActionDebugSingleResponse(Action):
    def name(self) -> Text:
        return "action_debug_single_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        if not RAG_AVAILABLE:
            dispatcher.utter_message(text="RAG system is not available.")
            return []
        
        try:
            # Test with a simple query that should return one response
            test_query = "What are the traffic rules in Baguio?"
            
            # Get the RAG response
            answer = generate_answer(test_query)
            
            # Log the response details
            logger.info(f"🔍 RAG Response Type: {type(answer)}")
            logger.info(f"🔍 RAG Response Length: {len(answer)} characters")
            logger.info(f"🔍 RAG Response Preview: {answer[:200]}...")
            
            # Count how many dispatcher calls we make
            logger.info("📤 Sending SINGLE message via dispatcher...")
            
            # Send as ONE message with clear boundaries
            bounded_response = f"🚨 START SINGLE MESSAGE 🚨\n\n{answer}\n\n🚨 END SINGLE MESSAGE 🚨"
            dispatcher.utter_message(text=bounded_response)
            
            logger.info("✅ Single message sent via dispatcher")
            
        except Exception as e:
            logger.error(f"❌ Debug error: {e}")
            dispatcher.utter_message(text=f"Debug error: {e}")
        
        return []

class ActionQueryOrdinances(Action):
    def name(self) -> Text:
        return "action_query_ordinances"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("🔍 ActionQueryOrdinances triggered")
        return ActionSearchKnowledge().run(dispatcher, tracker, domain)

class ActionHandleOrdinancePenalty(Action):
    def name(self) -> Text:
        return "action_handle_ordinance_penalty"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("🔍 ActionHandleOrdinancePenalty triggered")
        return ActionSearchKnowledge().run(dispatcher, tracker, domain)

class ActionHandleBusinessPermit(Action):
    def name(self) -> Text:
        return "action_handle_business_permit"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("🔍 ActionHandleBusinessPermit triggered")
        return ActionSearchKnowledge().run(dispatcher, tracker, domain)

class ActionShowOrdinanceStats(Action):
    def name(self) -> Text:
        return "action_show_ordinance_stats"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("🔍 ActionShowOrdinanceStats triggered")
        return ActionCheckKnowledgeBase().run(dispatcher, tracker, domain)

# Add any other actions from your domain.yml that might be triggering

class ActionFindMultipleMessages(Action):
    def name(self) -> Text:
        return "action_find_multiple_messages"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        logger.info("🕵️‍♂️ TRACING MULTIPLE MESSAGE SOURCE")
        
        # Check recent events
        recent_events = tracker.events[-20:]  # Last 20 events
        for i, event in enumerate(recent_events):
            logger.info(f"Event {i}: {event.get('event')} - {event.get('name', 'N/A')}")
        
        # Test single message
        test_response = """Baguio City has established various traffic rules to ensure safety.

1. Enforcement of Traffic Rules
- The Baguio City Police Office implements regulations
- Various personnel monitor compliance

2. Non-Motorized Vehicle Regulations  
- Designated lanes for safety
- Concessionaires must follow rules

3. Safety Measures for Pedestrians
- Clear pedestrian lanes
- Yield to pedestrians

• Follow speed limits and signs
• Be vigilant in crowded areas

Crucial for safety awareness."""

        logger.info("📤 Sending SINGLE test message")
        dispatcher.utter_message(text=test_response)
        logger.info("✅ Single test message sent")
        
        return []