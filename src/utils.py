# src/utils.py
"""
Translation utilities and LLM-as-Judge evaluation
"""
import logging
import json
import os
from typing import Optional, Dict, Any

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
    Translate text using deep-translator with explicit language control.
    ONLY supports English, Hindi, Chinese.
    """
    if not text or not text.strip():
        return text

    # EXPLICIT language mapping
    def map_language_code(lang_code: str) -> str:
        """Map our custom codes to deep-translator codes"""
        lang_map = {
            # Our project languages ONLY
            "hindi": "hi",
            "chinese": "zh-CN",  # Force Chinese Simplified
            "english": "en",
            # Standard codes
            "hi": "hi",
            "zh": "zh-CN",
            "zh-cn": "zh-CN",
            "zh-CN": "zh-CN",
            "en": "en",
            "auto": "auto",
        }
        mapped = lang_map.get(lang_code.lower(), None)
        if mapped is None:
            logger.warning(f"Unsupported language code: {lang_code}, defaulting to 'en'")
            return "en"
        return mapped

    # Map language codes with strict validation
    src_mapped = map_language_code(src)
    dest_mapped = map_language_code(dest)

    # If source and dest are the same, no translation needed
    if src_mapped == dest_mapped:
        return text

    # CRITICAL: Validate target language
    allowed_targets = {"hi", "zh-CN", "en"}
    if dest_mapped not in allowed_targets:
        logger.error(
            f"BLOCKED: Target language '{dest_mapped}' not in allowed set: {allowed_targets}"
        )
        return text

    logger.info(f"Translating: {src} -> {dest} (mapped: {src_mapped} -> {dest_mapped})")
    logger.info(f"Text preview: {text[:50]}...")

    if _BACKEND == "deep-translator":
        try:
            # Translation with explicit source and target
            if src_mapped == "auto":
                translator = _dt_cls(source="auto", target=dest_mapped)
            else:
                translator = _dt_cls(source=src_mapped, target=dest_mapped)

            translated = translator.translate(text)

            # NOTE: Spanish validation DISABLED
            # System prompts in rag_system.py already handle Spanish filtering

            if translated:
                logger.info(f"Translation successful: {translated[:50]}...")
                return translated
            else:
                logger.warning("Translation returned empty result")
                return text

        except Exception as e:
            logger.error(f"deep-translator failed: {src_mapped} -> {dest_mapped} -> {e}")
            return text

    else:
        logger.warning("No translation backend available; returning original text.")
        return text


# ============================================================================
# LLM-as-Judge Evaluation Functions
# ============================================================================
def evaluate_with_llm_judge(
    question_native: str,
    question_english: str,
    reference_text: Any,
    answer_native: str,
    answer_english: str,
    approach: str,
    model: str = "gpt-4o",
    use_claude: bool = False,
) -> Optional[Dict]:
    """
    Evaluate a RAG response using LLM-as-a-Judge

    Supports:
    - OpenAI (default)
    - Anthropic Claude (if model contains 'claude' OR use_claude=True)
    """

    # --- normalize reference_text so notebook CSVs don't break us ---
    if reference_text is None:
        reference_text_str = ""
    elif isinstance(reference_text, str):
        reference_text_str = reference_text
    else:
        # list / dict / whatever → stringify
        reference_text_str = json.dumps(reference_text, ensure_ascii=False)

    PROMPT = f"""You are an expert medical information evaluator assessing the quality of AI-generated healthcare responses.

[BEGIN DATA]
************
[System approach]: {approach}

[User question in native language]: 
{question_native}

[User question in English]: 
{question_english}

************
[Reference text (Retrieved chunks from RAG system)]: 
{reference_text_str}

************
[Generated Answer in native language]: 
{answer_native}

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

    # ------------------------------------------------------------------
    # decide: Claude or OpenAI
    # ------------------------------------------------------------------
    want_claude = use_claude or ("claude" in (model or "").lower())

    # ------------------------------------------------------
    # 1. CLAUDE PATH
    # ------------------------------------------------------
    if want_claude:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set")
            return None

        try:
            from anthropic import Anthropic
        except ImportError:
            logger.error("anthropic package not installed. pip install anthropic")
            return None

        client = Anthropic(api_key=api_key)
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=900,
                temperature=0.0,
                system="You are an expert medical information evaluator. Respond only with valid JSON.",
                messages=[{"role": "user", "content": PROMPT}],
            )

            # Claude 3+ returns list of content blocks
            text_parts = []
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(block.text)
            raw = "".join(text_parts).strip()

            # try direct JSON
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # try to extract { ... }
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(raw[start : end + 1])
                    except Exception:
                        logger.error("Claude returned non-JSON, even after slicing")
                        logger.error(raw)
                        return None
                else:
                    logger.error("Claude did not return JSON")
                    logger.error(raw)
                    return None

        except Exception as e:
            logger.error(f"LLM-as-judge (Claude) failed: {e}")
            return None

    # ------------------------------------------------------
    # 2. OPENAI PATH
    # ------------------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return None

    try:
        import openai

        if hasattr(openai, "OpenAI"):
            # new SDK
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical information evaluator. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": PROMPT},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            return result
        else:
            # old SDK
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical information evaluator. Respond only with valid JSON.",
                    },
                    {"role": "user", "content": PROMPT},
                ],
                temperature=0.0,
            )
            result = json.loads(response.choices[0].message.content)
            return result

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM-as-judge evaluation failed: {e}")
        return None
