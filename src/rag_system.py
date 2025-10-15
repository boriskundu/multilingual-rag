"""
RAG system with multilingual support - UPDATED for GPT-4 and chunk saving
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Embeddings & Vector stores
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document

import pandas as pd

# Use the sync translation wrapper
from src.utils import translate_text, translation_backend

logger = logging.getLogger(__name__)

def _call_openai_chat(system_prompt: str, context: str, query: str, model: str = "gpt-4o") -> str:
    """
    Call OpenAI API with GPT-4 (or specified model).
    Works with both old and new OpenAI SDK versions.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        # Try new SDK first (openai >= 1.0)
        import openai
        if hasattr(openai, 'OpenAI'):
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,  # Now using GPT-4
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return resp.choices[0].message.content
        else:
            # Old SDK (openai < 1.0)
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

class MultilingualRAG:
    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        vector_store_type: str = "chroma",
        persist_directory: str = "./data/embeddings",
        use_multilingual_embeddings: bool = True,
        llm_model: str = "gpt-4o"  # Default to GPT-4
    ):
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.use_multilingual_embeddings = use_multilingual_embeddings
        self.llm_model = llm_model  # Store LLM model choice

        # Initialize embeddings
        if ("ada" in embedding_model) or ("openai" in embedding_model.lower()):
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"}
            )

        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create and persist vector store from documents"""
        if self.vector_store_type.lower() == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory / "chroma")
            )
            self.vector_store.persist()
        elif self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            self.vector_store.save_local(str(self.persist_directory / "faiss"))

        logger.info(
            f"Created {self.vector_store_type} vector store with {len(documents)} documents"
        )

    def load_vector_store(self) -> None:
        """Load existing vector store"""
        if self.vector_store_type.lower() == "chroma":
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory / "chroma"),
                embedding_function=self.embeddings
            )
        elif self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.load_local(
                str(self.persist_directory / "faiss"),
                self.embeddings
            )
        logger.info(f"Loaded {self.vector_store_type} vector store")

    def multilingual_retrieval(
        self,
        query: str,
        target_language: str = "en",
        k: int = 5
    ) -> Tuple[List[Document], str, float]:
        """
        Retrieve documents using multilingual approach.
        Returns: (documents, effective_query_used, retrieval_time)
        """
        start_time = time.time()

        if self.use_multilingual_embeddings:
            # Approach 1: Direct multilingual embeddings
            effective_query = query
        else:
            # Approach 2: Translate then retrieve (to English)
            if target_language and target_language.lower() != "en":
                effective_query = translate_text(query, src=target_language, dest="en")
            else:
                effective_query = query

        retrieved_docs = self.vector_store.similarity_search(effective_query, k=k)
        retrieval_time = time.time() - start_time
        return retrieved_docs, effective_query, retrieval_time

    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Document],
        source_language: str = "en",
        system_prompt: Optional[str] = None
    ) -> Tuple[str, float, str]:
        """
        Generate response using LLM.
        
        Returns: (answer_in_source_language, gen_time_seconds, combined_chunks_text)
        """
        start_time = time.time()

        if system_prompt is None:
            system_prompt = (
                "You are a helpful healthcare information assistant. "
                "Based on the provided context, answer the question accurately and concisely. "
                "If the context doesn't contain relevant information, say so."
            )

        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Call OpenAI with GPT-4
        try:
            answer = _call_openai_chat(system_prompt, context, query, model=self.llm_model)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"[Error generating response]"

        # Translate back to source language if needed
        if source_language and source_language.lower() != "en":
            answer = translate_text(answer, src="en", dest=source_language)

        generation_time = time.time() - start_time
        
        # Return chunks for LLM-as-judge evaluation
        return answer, generation_time, context

    def run_experiment(
        self,
        questions: List[Dict[str, str]],
        output_path: str = "results/rag_experiments.csv"
    ) -> pd.DataFrame:
        """Run experiments and save results with retrieved chunks"""
        results: List[Dict[str, Any]] = []

        for q_data in questions:
            question = q_data["question"]
            language = q_data.get("language", "en")

            logger.info(f"Processing question: {question}")

            # Approach 1: Multilingual Embeddings
            self.use_multilingual_embeddings = True
            docs_multi, query_multi, t_ret_multi = self.multilingual_retrieval(
                question, language
            )
            resp_multi, t_gen_multi, chunks_multi = self.generate_response(
                question, docs_multi, language
            )

            # Approach 2: Translation Pipeline
            self.use_multilingual_embeddings = False
            docs_trans, query_trans, t_ret_trans = self.multilingual_retrieval(
                question, language
            )
            resp_trans, t_gen_trans, chunks_trans = self.generate_response(
                query_trans, docs_trans, language
            )

            results.append({
                "question": question,
                "language": language,
                "translated_query": query_trans,
                
                # Save retrieved chunks for LLM-as-judge
                "multilingual_chunks": chunks_multi,
                "translation_chunks": chunks_trans,
                
                # Shortened version for preview
                "multilingual_content": "\n---\n".join([d.page_content[:200] for d in docs_multi]),
                "translation_content": "\n---\n".join([d.page_content[:200] for d in docs_trans]),
                
                "system_prompt": "Healthcare assistant prompt",
                "multilingual_response": resp_multi,
                "translation_response": resp_trans,
                "multilingual_time": t_ret_multi + t_gen_multi,
                "translation_time": t_ret_trans + t_gen_trans,
                "retrieval_time_multi": t_ret_multi,
                "retrieval_time_trans": t_ret_trans,
                "generation_time_multi": t_gen_multi,
                "generation_time_trans": t_gen_trans
            })

        df = pd.DataFrame(results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved experiment results to {output_path}")
        return df