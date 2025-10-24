#!/usr/bin/env python3
import os
import glob
from actions.rag_pipeline import rag_pipeline  # ✅ import the instance, not the module

def setup_knowledge_base():
    """Initialize the knowledge base with documents using FAISS"""
    knowledge_path = "knowledge_base/documents/"
    
    if not os.path.exists(knowledge_path):
        os.makedirs(knowledge_path)
        print(f"Created knowledge base directory: {knowledge_path}")
        print("Please add your PDF, TXT, or DOCX files to this directory and run this script again.")
        return
    
    # Find all supported documents
    documents = []
    for ext in ['*.pdf', '*.txt', '*.docx']:
        documents.extend(glob.glob(os.path.join(knowledge_path, ext)))
    
    if not documents:
        print("No documents found in knowledge base directory.")
        print("Supported formats: PDF, TXT, DOCX")
        print("Please add documents and run this script again.")
        return
    
    print(f"Found {len(documents)} documents to process...")
    
    for doc_path in documents:
        rag_pipeline.add_documents(doc_path)
    
    print("FAISS knowledge base setup complete!")
    print(f"Total documents in index: {len(rag_pipeline.documents)}")

if __name__ == "__main__":
    setup_knowledge_base()
