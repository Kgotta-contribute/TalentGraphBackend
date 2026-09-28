import logging
from functools import lru_cache
import numpy as np
from huggingface_hub import InferenceClient
from app.core.config import settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_inference_client() -> InferenceClient:
    api_key = settings.hf_token.strip() if settings.hf_token else None
    if not api_key:
        logger.warning("HF_TOKEN is not configured. Falling back to public anonymous Hugging Face client.")
    return InferenceClient(provider="hf-inference", api_key=api_key)

def embed_text(text: str) -> list[float]:
    """
    Generate a 1024-dimensional normalized dense embedding for a single text using
    Hugging Face Serverless Inference API with BAAI/bge-m3.
    """
    if not text or not text.strip():
        return [0.0] * settings.embedding_dimensions

    client = get_inference_client()
    try:
        raw_res = client.feature_extraction(
            text=text,
            model=settings.embedding_model,
            normalize=True,
        )
        arr = np.array(raw_res, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.squeeze()
        return arr.tolist()
    except Exception as e:
        logger.error(f"Error generating Hugging Face embedding for text: {e}", exc_info=True)
        raise

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate 1024-dimensional normalized dense embeddings for a batch of texts using
    Hugging Face Serverless Inference API with BAAI/bge-m3.
    """
    if not texts:
        return []

    clean_texts = [t if t and t.strip() else " " for t in texts]
    client = get_inference_client()
    try:
        raw_res = client.feature_extraction(
            text=clean_texts,
            model=settings.embedding_model,
            normalize=True,
        )
        arr = np.array(raw_res, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)
        return arr.tolist()
    except Exception as e:
        logger.error(f"Error generating batch Hugging Face embeddings: {e}", exc_info=True)
        raise
