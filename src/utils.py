# src/utils.py
"""
Translation utilities using deep-translator (stable, synchronous)
"""
import logging
from typing import Optional

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
            # deep-translator requires explicit source language, not 'auto'
            # If 'auto' is specified, try to detect or default to common source
            if src == "auto":
                # For your use case, you know it's Hindi or English
                # You could add langdetect here if needed
                # For now, we'll just pass through if dest is English
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