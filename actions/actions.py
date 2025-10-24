from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

import sys
import os
sys.path.append(os.path.dirname(__file__))

from rag_pipeline import rag_pipeline

class ActionSearchKnowledge(Action):
    def name(self) -> Text:
        return "action_search_knowledge"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get user message
        user_message = tracker.latest_message.get('text')
        
        # Extract entities or use full message
        question_entity = next(tracker.get_latest_entity_values("question"), None)
        search_query = question_entity or user_message
        
        try:
            # Search for relevant information
            similar_docs = rag_pipeline.search_similar(search_query, n_results=3)
            
            if similar_docs:
                # Generate response using RAG
                response = rag_pipeline.generate_response(search_query, similar_docs)
                
                # Add source information
                sources = list(set([doc['source'] for doc in similar_docs]))
                source_info = f"\n\nSources: {', '.join([os.path.basename(src) for src in sources])}"
                
                dispatcher.utter_message(text=response + source_info)
            else:
                dispatcher.utter_message(
                    text="I couldn't find relevant information in my knowledge base. " \
                         "Please try rephrasing your question or adding more documents to the knowledge base."
                )
                
        except Exception as e:
            dispatcher.utter_message(
                text="Sorry, I encountered an error while searching for information. " \
                     "Please try again later."
            )
            print(f"Error in action_search_knowledge: {str(e)}")
        
        return [SlotSet("search_query", search_query)]

class ActionAddDocument(Action):
    def name(self) -> Text:
        return "action_add_document"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # This would typically be called via an API to add new documents
        # For now, we'll just provide information
        dispatcher.utter_message(
            text="To add documents to my knowledge base, please place PDF, TXT, or DOCX files " \
                 "in the 'knowledge_base/documents/' folder and restart the application."
        )
        
        return []