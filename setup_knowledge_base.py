#!/usr/bin/env python3
import os
import glob
import sys

# Add actions directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'actions'))

try:
    from rag_pipeline import rag_pipeline
    
    def setup_knowledge_base():
        """Initialize the knowledge base with documents"""
        knowledge_path = "knowledge_base/documents/"
        
        if not os.path.exists(knowledge_path):
            os.makedirs(knowledge_path)
            print(f"📁 Created directory: {knowledge_path}")
            print("💡 Please add your Baguio City PDF documents to this folder and run again.")
            return
        
        documents = []
        for ext in ['*.pdf', '*.txt', '*.docx']:
            documents.extend(glob.glob(os.path.join(knowledge_path, ext)))
        
        if not documents:
            print("❌ No documents found in knowledge_base/documents/")
            print("💡 Add Baguio City PDF files to the folder and run again.")
            return
        
        print(f"📄 Found {len(documents)} documents to process...")
        
        for doc_path in documents:
            rag_pipeline.add_documents(doc_path)
        
        print(f"✅ Knowledge base setup complete!")
        print(f"📊 Total documents chunks: {len(rag_pipeline.documents)}")

    if __name__ == "__main__":
        setup_knowledge_base()

except ImportError as e:
    print(f"❌ Failed to import RAG pipeline: {e}")
    print("💡 Make sure all dependencies are installed:")
    print("   pip install -r actions/requirements_actions.txt")
except Exception as e:
    print(f"❌ Error: {e}")