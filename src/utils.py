# src/utils.py
"""
Translation utilities and LLM-as-Judge evaluation
"""
import logging
import json
import os
from typing import Optional, Dict

logger = logging.getLogger(__name__)

_BACKEND = None
_dt_cls = None

# Use deep-translator (sync, stable, no conflicts)
try:
    from deep_translator import GoogleTranslator as _DeepTranslator
    _dt_cls = _DeepTranslator
    _BACKEND = "deep-translator"
    logger.info("Translation backend: deep-translator")
except Exception as e:
    logger.warning(f"deep-translator not available: {e}")
    _BACKEND = "none"

def translation_backend() -> str:
    """Return the active translation backend"""
    return _BACKEND or "none"

def translate_text(text: str, src: str = "auto", dest: str = "en") -> str:
    """
    Translate text using deep-translator (synchronous, reliable).
    
    Args:
        text: Text to translate
        src: Source language code (e.g., 'hi', 'es', 'auto')
        dest: Destination language code (e.g., 'en', 'hi')
    
    Returns:
        Translated text, or original text if translation fails
    """
    if not text or not text.strip():
        return text
    
    # If source and dest are the same, no translation needed
    if src == dest:
        return text
    
    if _BACKEND == "deep-translator":
        try:
            if src == "auto":
                translator = _dt_cls(source='auto', target=dest)
            else:
                translator = _dt_cls(source=src, target=dest)
            
            translated = translator.translate(text)
            return translated if translated else text
            
        except Exception as e:
            logger.warning(f"deep-translator failed: {e}; returning original text.")
            return text
    
    else:
        logger.warning("No translation backend available; returning original text.")
        return text


# ============================================================================
# LLM-as-Judge Evaluation Functions
# ============================================================================

def evaluate_with_llm_judge(
    question_hindi: str,
    question_english: str,
    reference_text: str,
    answer_hindi: str,
    answer_english: str,
    approach: str,
    model: str = "gpt-4o"
) -> Optional[Dict]:
    """
    Evaluate a RAG response using LLM-as-a-Judge
    
    Args:
        question_hindi: Original Hindi question
        question_english: English translation of question
        reference_text: Retrieved chunks from RAG (combined)
        answer_hindi: Generated answer in Hindi
        answer_english: English translation of answer
        approach: "Multilingual Embeddings" or "Translation Pipeline"
        model: GPT model to use for evaluation
    
    Returns:
        dict: Evaluation results with scores and labels
    """
    
    PROMPT = f"""You are an expert medical information evaluator assessing the quality of AI-generated healthcare responses.

[BEGIN DATA]
************
[System approach]: {approach}

[User question in Hindi]: 
{question_hindi}

[User question in English]: 
{question_english}

************
[Reference text (Retrieved chunks from RAG system)]: 
{reference_text}

************
[Generated Answer in Hindi]: 
{answer_hindi}

[Generated Answer in English translation (for evaluation)]:
{answer_english}

************
[END DATA]

CRITICAL: The answer MUST be based ONLY on the reference text above. The reference text is the combined chunks retrieved by the RAG system. Any information not present in the reference text is considered hallucination.

Your task is to evaluate the generated answer across THREE dimensions:

1. FAITHFULNESS (Hallucination Check):
   - Does the answer only use information present in the reference text?
   - "factual" = answer is fully grounded in reference text
   - "hallucinated" = answer contains information NOT in reference text
   - "partial" = answer is mostly grounded but adds some unsupported claims

2. COMPLETENESS (Information Coverage):
   - Does the answer adequately address the user's question?
   - "complete" = fully answers the question with relevant details
   - "partial" = answers but misses important aspects
   - "incomplete" = fails to answer the core question
   - "no_answer" = explicitly states no information available

3. MEDICAL APPROPRIATENESS (Safety & Relevance):
   - Is the information medically appropriate for the question asked?
   - "appropriate" = correct information type for the question
   - "inappropriate" = wrong information type (e.g., medication info when symptoms asked)
   - "potentially_harmful" = contains incorrect medical information

Respond ONLY with valid JSON in this exact format:

{{
  "faithfulness": {{
    "label": "factual" | "hallucinated" | "partial",
    "score": 1-5,
    "explanation": "Brief explanation"
  }},
  "completeness": {{
    "label": "complete" | "partial" | "incomplete" | "no_answer",
    "score": 1-5,
    "explanation": "Brief explanation"
  }},
  "medical_appropriateness": {{
    "label": "appropriate" | "inappropriate" | "potentially_harmful",
    "score": 1-5,
    "explanation": "Brief explanation"
  }},
  "overall_assessment": "One sentence summary",
  "key_issues": ["List specific problems if any"]
}}"""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return None

    try:
        import openai
        if hasattr(openai, 'OpenAI'):
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical information evaluator. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": PROMPT
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            return result
        else:
            # Old SDK
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical information evaluator. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": PROMPT
                    }
                ],
                temperature=0.0
            )
            result = json.loads(response.choices[0].message.content)
            return result
            
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM-as-judge evaluation failed: {e}")
        return None