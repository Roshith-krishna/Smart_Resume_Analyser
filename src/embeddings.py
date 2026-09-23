"""Reusable wrapper around a pretrained Sentence Transformer model.

V1 uses a single pretrained model (``sentence-transformers/all-MiniLM-L6-v2``)
to generate semantic embeddings for both candidate skill profiles and job
postings. No model is trained from scratch — this class only ever loads
and calls an existing pretrained model.

Embeddings are L2-normalized (``normalize_embeddings=True``). For
normalized vectors, cosine similarity and the plain dot product are
mathematically equivalent:

    cos(a, b) = (a . b) / (||a|| * ||b||) = a . b   when ||a|| = ||b|| = 1

See:
https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html

``matcher.py`` still calls scikit-learn's explicit ``cosine_similarity``
rather than a raw dot product, purely for robustness/clarity (it is
correct even if a vector is not perfectly unit-norm due to floating point
drift) — this is documented there.

The Sentence Transformer library (and its ``torch`` dependency) is
imported lazily, inside ``_load()``, and a pre-built model instance can be
injected via the constructor. This keeps the rest of the codebase
testable without requiring the (large) model download in every
environment.
"""

from __future__ import annotations

import numpy as np

from src import config


class EmbeddingModel:
    """Lazy-loading wrapper around a Sentence Transformer model."""

    def __init__(self, model_name: str = config.DEFAULT_MODEL_NAME, model: object = None):
        """
        Args:
            model_name: HuggingFace/Sentence-Transformers model identifier.
            model: Optional pre-built model instance (must expose an
                ``.encode(list[str], normalize_embeddings=True) -> ndarray``
                method compatible with ``sentence_transformers.SentenceTransformer``).
                Intended for dependency injection in tests; leave as
                ``None`` in normal use so the real pretrained model loads.
        """
        self.model_name = model_name
        self._model = model

    def _load(self) -> object:
        if self._model is None:
            import warnings
            # Filter the specific torchvision warning that does not affect sentence-transformers
            warnings.filterwarnings(
                "ignore",
                message="Failed to load image Python extension",
                category=UserWarning,
                module="torchvision"
            )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required to generate embeddings "
                    "but is not installed. Install it with: "
                    "pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the vectors this model produces."""
        model = self._load()
        return int(model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        """Encode a single string into a normalized embedding vector."""
        model = self._load()
        vector = model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)

    def encode_batch(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        """Encode a batch of strings into a matrix of normalized embedding
        vectors, shape (len(texts), embedding_dim)."""
        model = self._load()
        matrix = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(matrix, dtype=np.float32)
