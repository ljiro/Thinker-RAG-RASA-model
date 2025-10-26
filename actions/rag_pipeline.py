import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TinyLlamaRAGPipeline:
    def __init__(self):
        logger.info("🧠 Checking device setup...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✅ Using device: {self.device.upper()}")
        self._setup_models()

    # ----------------------------------------------------
    # Model setup
    # ----------------------------------------------------
    def _setup_models(self):
        try:
            logger.info("🚀 Initializing RAG Pipeline with TinyLlama GPTQ (optimized for GTX 1650)...")

            # Embedding model
            logger.info("📥 Loading embedding model (MiniLM)...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

            # ---------------------
            # LLM (AutoGPTQ)
            # ---------------------
            model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ"
            logger.info(f"📦 Attempting to load AutoGPTQ model: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.llm = AutoGPTQForCausalLM.from_quantized(
                model_name,
                device_map="auto",
                use_safetensors=True,
                trust_remote_code=True,
                max_memory={0: "3.0GiB", "cpu": "8GiB"},
                offload_buffers=True,  # 👈 Prevents CUDA OOM
            )
            self.llm.eval()

            logger.info(f"✅ Successfully loaded {model_name} (4-bit quantized, GPU-safe)")

        except Exception as e:
            logger.error(f"❌ AutoGPTQ model load failed: {e}")
            self._setup_fallback_model()

        # Setup FAISS after model init
        self._setup_faiss()

    # ----------------------------------------------------
    # Fallback model (DistilGPT-2)
    # ----------------------------------------------------
    def _setup_fallback_model(self):
        try:
            fallback_name = "distilgpt2"
            logger.info("🔄 Falling back to DistilGPT-2 (CPU safe)...")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            self.llm = AutoModelForCausalLM.from_pretrained(fallback_name)
            self.device = "cpu"
            self.llm.to(self.device)
            logger.info("✅ DistilGPT-2 fallback loaded")
        except Exception as e:
            logger.error(f"❌ Fallback model load failed: {e}")
            raise e

    # ----------------------------------------------------
    # FAISS setup
    # ----------------------------------------------------
    def _setup_faiss(self):
        logger.info("🔍 Setting up FAISS index...")
        dim = self.embedder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

        # Simulated documents (replace with real corpus)
        for i in range(417):
            doc = {
                "source": f"mock_source_{i}.pdf",
                "content": f"Document {i} content about Baguio tourism, traffic, and culture."
            }
            self.docs.append(doc)
            embedding = self.embedder.encode(doc["content"])
            self.index.add(np.array([embedding]).astype("float32"))

        logger.info(f"✅ Loaded FAISS index with {len(self.docs)} documents")
        logger.info("✅🎉 RAG pipeline ready for Baguio City Q&A (TinyLlama GPTQ optimized)!")

    # ----------------------------------------------------
    # Retrieval
    # ----------------------------------------------------
    def retrieve_context(self, query, k=3):
        query_emb = self.embedder.encode(query)
        D, I = self.index.search(np.array([query_emb]).astype("float32"), k)
        return [self.docs[i] for i in I[0]]  # ✅ returns list of dicts

    # ✅ Alias for backward compatibility
    def search_similar(self, query, n_results=3):
        """Alias for backward compatibility with pdf_processor.py"""
        return self.retrieve_context(query, k=n_results)

    # ----------------------------------------------------
    # Text Generation (cleaned up and improved)
    # ----------------------------------------------------
    def _generate(self, prompt):
        """Unified generation helper with cleanup and anti-repetition."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🧹 Clean up repeated lines and prompt echoes
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            if not line.strip().lower().startswith(("question:", "context:")) and line.strip():
                cleaned.append(line)
        cleaned_text = " ".join(cleaned).replace("Answer:", "").strip()
        return cleaned_text or "I'm sorry, I couldn't find relevant information."

    # ----------------------------------------------------
    # Public methods for RASA and testing
    # ----------------------------------------------------
    def generate_answer(self, query):
        """Simple query-based generation for console testing."""
        context_docs = self.retrieve_context(query)
        context_text = "\n".join([doc["content"] for doc in context_docs])
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer (concise and factual):"
        return self._generate(prompt)

    def generate_response(self, query, context_docs=None):
        """Alias for backward compatibility with pdf_processor.py"""
        if context_docs is None:
            context_docs = self.retrieve_context(query)
        context_text = "\n".join([doc["content"] for doc in context_docs])
        prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer (concise and factual):"
        return self._generate(prompt)


# ----------------------------------------------------
# Instantiate pipeline
# ----------------------------------------------------
if __name__ == "__main__":
    rag_pipeline = TinyLlamaRAGPipeline()
    test_query = "What makes Burnham Park in Baguio City famous?"
    answer = rag_pipeline.generate_answer(test_query)
    print("\n--- ANSWER ---")
    print(answer)

# For import by RASA or other systems
rag_pipeline = TinyLlamaRAGPipeline()
