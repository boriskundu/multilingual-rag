"""
RAG system with multilingual support - UPDATED for GPT-4 and chunk saving
IMPORTANT: Instructs LLM to ignore Spanish content in retrieved chunks
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Suppress progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Updated imports for newer langchain
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

import pandas as pd

# Use the sync translation wrapper
from src.utils import translate_text, translation_backend

logger = logging.getLogger(__name__)


# ============================================================
# BASE OPENAI CALL (kept exactly, used when use_claude=False)
# ============================================================
def _call_openai_chat(
    system_prompt: str, context: str, query: str, model: str = "gpt-4o"
) -> str:
    """
    Call OpenAI API with GPT-4 (or specified model).
    Works with both old and new OpenAI SDK versions.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        import openai

        if hasattr(openai, "OpenAI"):
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,  # Now using GPT-4
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        else:
            # Old SDK (openai < 1.0)
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ],
                temperature=0.3,
                max_tokens=500,
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
        llm_model: str = "gpt-4o",
        use_claude: bool = False,  # 👈 NEW
    ):
        """
        Args:
            embedding_model: HF or OpenAI embed model
            vector_store_type: "chroma" or "faiss"
            persist_directory: where to save vector store
            use_multilingual_embeddings: if True, query is used as-is
            llm_model: name of LLM ("gpt-4o" or "claude-3-...")
            use_claude: if True, call Anthropic instead of OpenAI
        """
        self.embedding_model = embedding_model
        self.vector_store_type = vector_store_type
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.use_multilingual_embeddings = use_multilingual_embeddings
        self.llm_model = llm_model
        self.use_claude = use_claude  # 👈 store it

        # Initialize embeddings
        if ("ada" in embedding_model) or ("openai" in embedding_model.lower()):
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cpu"},  # REMOVED show_progress_bar
            )

        self.vector_store = None

    # ======================================================
    # CLAUDE CALLER (in-class, so it sees self.llm_model)
    # ======================================================
    def _call_claude_chat(
        self, system_prompt: str, context: str, query: str, model: Optional[str] = None
    ) -> str:
        """
        Anthropic Claude call, same signature as _call_openai_chat.
        We do it here so notebooks can just pass use_claude=True.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. pip install anthropic"
            ) from e

        client = anthropic.Anthropic(api_key=api_key)
        model = model or self.llm_model

        user_content = f"Context:\n{context}\n\nQuestion: {query}"

        resp = client.messages.create(
            model=model,
            max_tokens=500,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        # For Claude, text is in resp.content[0].text
        return resp.content[0].text

    # ======================================================
    # VECTOR STORE OPS
    # ======================================================
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create and persist vector store from documents"""
        if self.vector_store_type.lower() == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory / "chroma"),
            )
            self.vector_store.persist()
        elif self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents, embedding=self.embeddings
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
                embedding_function=self.embeddings,
            )
        elif self.vector_store_type.lower() == "faiss":
            self.vector_store = FAISS.load_local(
                str(self.persist_directory / "faiss"), self.embeddings
            )
        logger.info(f"Loaded {self.vector_store_type} vector store")

    # ======================================================
    # RETRIEVAL
    # ======================================================
    def multilingual_retrieval(
        self, query: str, target_language: str = "en", k: int = 5
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

    # ======================================================
    # GENERATION
    # ======================================================
    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Document],
        source_language: str = "en",
        system_prompt: Optional[str] = None,
        return_english_for_eval: bool = False,
    ) -> Tuple[str, float, str, Optional[str]]:
        """
        Generate response using LLM.

        Returns: (answer, generation_time, context, english_answer_for_eval)
        - For multilingual embeddings: generates directly in target language
        - For translation pipeline: generates in English, then translates
        - english_answer_for_eval: English version for LLM judge evaluation
        """
        start_time = time.time()

        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Determine the generation language based on approach
        if self.use_multilingual_embeddings:
            # MULTILINGUAL EMBEDDINGS: Generate directly in target language
            if source_language and source_language.lower() != "en":
                if system_prompt is None:
                    system_prompt = (
                        "You are a helpful healthcare information assistant. "
                        "Based on the provided context from FDA and NIH medical sources, answer the question accurately and concisely. "
                        "CRITICAL INSTRUCTIONS:\n"
                        "1. IGNORE any Spanish text in the context - only use English information\n"
                        f"2. Respond in {source_language} language to match the user's question\n"
                        f"3. If the English content doesn't contain relevant information, say so clearly in {source_language}"
                    )

                try:
                    # Generate directly in target language
                    if self.use_claude:
                        answer = self._call_claude_chat(
                            system_prompt,
                            context,
                            query,  # original query in target language
                            model=self.llm_model,
                        )
                    else:
                        answer = _call_openai_chat(
                            system_prompt,
                            context,
                            query,  # original query in target language
                            model=self.llm_model,
                        )

                    # For evaluation purposes, translate to English
                    english_answer_for_eval = None
                    if return_english_for_eval:
                        english_answer_for_eval = translate_text(
                            answer, src=source_language, dest="en"
                        )

                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    answer = "[Error generating response]"
                    english_answer_for_eval = None
            else:
                # English query with multilingual embeddings
                if system_prompt is None:
                    system_prompt = (
                        "You are a helpful healthcare information assistant. "
                        "Based on the provided context from FDA and NIH medical sources, answer the question accurately and concisely. "
                        "IMPORTANT: Ignore any Spanish text in the context - only use English information. "
                        "If the English content doesn't contain relevant information, say so clearly."
                    )

                try:
                    if self.use_claude:
                        answer = self._call_claude_chat(
                            system_prompt, context, query, model=self.llm_model
                        )
                    else:
                        answer = _call_openai_chat(
                            system_prompt, context, query, model=self.llm_model
                        )
                    english_answer_for_eval = (
                        answer if return_english_for_eval else None
                    )
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    answer = "[Error generating response]"
                    english_answer_for_eval = None

        else:
            # TRANSLATION PIPELINE: Always generate in English, then translate
            if system_prompt is None:
                system_prompt = (
                    "You are a helpful healthcare information assistant. "
                    "Based on the provided context from FDA and NIH medical sources, answer the question accurately and concisely. "
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. IGNORE any Spanish text in the context - only use English information\n"
                    "2. Respond ONLY in English\n"
                    "3. If the English content doesn't contain relevant information, say so clearly"
                )

            # Translate query to English if needed
            english_query = query
            if source_language and source_language.lower() != "en":
                english_query = translate_text(query, src=source_language, dest="en")

            try:
                # Generate in English
                if self.use_claude:
                    answer_en = self._call_claude_chat(
                        system_prompt, context, english_query, model=self.llm_model
                    )
                else:
                    answer_en = _call_openai_chat(
                        system_prompt, context, english_query, model=self.llm_model
                    )

                # Store English answer for evaluation
                english_answer_for_eval = (
                    answer_en if return_english_for_eval else None
                )

                # Translate to target language
                if source_language and source_language.lower() != "en":
                    answer = translate_text(answer_en, src="en", dest=source_language)
                else:
                    answer = answer_en

            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                answer = "[Error generating response]"
                english_answer_for_eval = None

        generation_time = time.time() - start_time
        return answer, generation_time, context, english_answer_for_eval

    # ======================================================
    # EXPERIMENT LOOP
    # ======================================================
    def run_experiment(
        self, questions: List[Dict[str, str]], output_path: str = "results/rag_experiments.csv"
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
            # No translation steps in multilingual embeddings
            (
                resp_multi,
                t_gen_multi,
                chunks_multi,
                english_multi,
            ) = self.generate_response(
                question,
                docs_multi,
                language,
                return_english_for_eval=True,
            )

            # Approach 2: Translation Pipeline
            self.use_multilingual_embeddings = False
            docs_trans, query_trans, t_ret_trans = self.multilingual_retrieval(
                question, language
            )
            # Translation pipeline includes query translation + back translation
            (
                resp_trans,
                t_gen_trans,
                chunks_trans,
                english_trans,
            ) = self.generate_response(
                query_trans,
                docs_trans,
                language,
                return_english_for_eval=True,
            )

            results.append(
                {
                    "question": question,
                    "language": language,
                    "translated_query": query_trans,
                    # Save retrieved chunks for LLM-as-judge
                    "multilingual_chunks": chunks_multi,
                    "translation_chunks": chunks_trans,
                    # Save English versions for evaluation
                    "multilingual_english": english_multi,
                    "translation_english": english_trans,
                    # Shortened version for preview
                    "multilingual_content": "\n---\n".join(
                        [d.page_content[:200] for d in docs_multi]
                    ),
                    "translation_content": "\n---\n".join(
                        [d.page_content[:200] for d in docs_trans]
                    ),
                    "system_prompt": "Healthcare assistant prompt",
                    "multilingual_response": resp_multi,
                    "translation_response": resp_trans,
                    "multilingual_time": t_ret_multi + t_gen_multi,
                    "translation_time": t_ret_trans + t_gen_trans,
                    "retrieval_time_multi": t_ret_multi,
                    "retrieval_time_trans": t_ret_trans,
                    "generation_time_multi": t_gen_multi,
                    "generation_time_trans": t_gen_trans,
                    # extra: which LLM was used
                    "llm_model": self.llm_model,
                    "use_claude": self.use_claude,
                }
            )

        df = pd.DataFrame(results)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved experiment results to {output_path}")
        return df
