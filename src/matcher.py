"""Cosine-similarity job matching against precomputed job vectors.

Two-stage retrieval architecture (V1.2):

    Stage 1 — Semantic Retrieval:
        Compute cosine similarity of the candidate embedding against ALL
        precomputed job vectors.  Retrieve the top ``candidate_pool_size``
        (default 50) candidates ranked by descending similarity.

    Stage 2 — Deduplication:
        Remove duplicate jobs (by ``duplicate_signature`` metadata) from
        the pool.  Return the top ``top_k`` unique semantic matches.

Skill matching is deliberately NOT performed inside this module.  The CLI
(``src/cli.py``) applies lightweight skill-gap analysis only to the
final top-K results — never to the full dataset.

Job vectors are computed once (offline, by ``scripts/vectorize_jobs.py``)
and simply loaded and compared here — they are never recomputed at match
time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src import config


class JobMatcherError(Exception):
    """Raised for any problem loading or matching against the precomputed
    job vector store (missing files, row-count mismatch, dimension
    mismatch, invalid top-k, etc.)."""


@dataclass
class MatchInfo:
    """Debug/verbose statistics about a match run."""
    total_jobs_searched: int = 0
    candidate_pool_size: int = 0
    unique_candidates_found: int = 0
    final_results: int = 0
    embedding_dim: int = 0


class JobMatcher:
    """Loads precomputed job vectors + metadata and matches a candidate
    embedding against them using cosine similarity.

    V1.2 two-stage retrieval:
      1. Cosine similarity against all vectors → top candidate_pool_size
      2. Deduplicate → return top_k unique results
    """

    def __init__(self, vectors: np.ndarray, metadata: pd.DataFrame, vectorization_config: dict):
        self.vectors = vectors
        self.metadata = metadata
        self.vectorization_config = vectorization_config

    @classmethod
    def load(
        cls,
        vectors_path: Path | str = config.DEFAULT_JOB_VECTORS_PATH,
        metadata_path: Path | str = config.DEFAULT_JOBS_METADATA_PATH,
        vectorization_config_path: Path | str = config.DEFAULT_VECTORIZATION_CONFIG_PATH,
    ) -> "JobMatcher":
        """Load and validate the job vector store from disk.

        Raises:
            JobMatcherError: if any file is missing, unreadable, or the
                vectors/metadata are inconsistent with each other.
        """
        vectors_path = Path(vectors_path)
        metadata_path = Path(metadata_path)
        vectorization_config_path = Path(vectorization_config_path)

        if not vectors_path.exists():
            raise JobMatcherError(
                f"Job vectors file not found: {vectors_path}\n"
                "Run 'python -m scripts.vectorize_jobs --csv <your_dataset.csv>' first."
            )
        if not metadata_path.exists():
            raise JobMatcherError(
                f"Job metadata file not found: {metadata_path}\n"
                "Run 'python -m scripts.vectorize_jobs --csv <your_dataset.csv>' first."
            )

        try:
            vectors = np.load(vectors_path)
        except Exception as exc:
            raise JobMatcherError(f"Failed to load job vectors from {vectors_path}: {exc}") from exc

        try:
            metadata = pd.read_csv(metadata_path)
        except Exception as exc:
            raise JobMatcherError(f"Failed to load job metadata from {metadata_path}: {exc}") from exc

        # --- Vectorization config validation ---
        vectorization_config: dict = {}
        if not vectorization_config_path.exists():
            raise JobMatcherError(
                f"Vectorization config not found: {vectorization_config_path}\n"
                "Run 'python -m scripts.vectorize_jobs' to regenerate."
            )
        try:
            with open(vectorization_config_path, "r", encoding="utf-8") as file_handle:
                vectorization_config = json.load(file_handle)
        except Exception as exc:
            raise JobMatcherError(
                f"Failed to load vectorization config from {vectorization_config_path}: {exc}"
            ) from exc

        # Accept V1.1 vectors — the semantic representation is unchanged in V1.2.
        compatible_versions = {"v1.1", "v1.2"}
        config_version = vectorization_config.get("version")
        if config_version not in compatible_versions:
            raise JobMatcherError(
                f"Incompatible vector store version '{config_version}'. "
                f"Expected one of {compatible_versions}. "
                "Please re-run 'python -m scripts.vectorize_jobs' to regenerate."
            )

        if "Role" not in metadata.columns or "duplicate_signature" not in metadata.columns:
            raise JobMatcherError(
                "Job metadata is missing required columns (Role or duplicate_signature). "
                "Please re-run 'python -m scripts.vectorize_jobs'."
            )

        # --- Vector / metadata consistency ---
        if len(vectors) != len(metadata):
            raise JobMatcherError(
                f"Job vectors/metadata row-count mismatch: "
                f"{len(vectors)} vectors vs {len(metadata)} metadata rows. "
                "The vector store is corrupted or out of sync — re-run "
                "'python -m scripts.vectorize_jobs' to regenerate both files together."
            )

        if len(vectors) == 0:
            raise JobMatcherError("Job vector store is empty (0 jobs). Nothing to match against.")

        # --- Embedding dimension validation ---
        expected_dim = vectorization_config.get("embedding_dim")
        actual_dim = int(vectors.shape[1])
        if expected_dim is not None and int(expected_dim) != actual_dim:
            raise JobMatcherError(
                f"Vector dimension mismatch: config says {expected_dim}, "
                f"but vectors have {actual_dim} dimensions. "
                "The vector store may be corrupted — re-run 'python -m scripts.vectorize_jobs'."
            )

        return cls(vectors=vectors, metadata=metadata, vectorization_config=vectorization_config)

    @property
    def embedding_dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def num_jobs(self) -> int:
        return int(self.vectors.shape[0])

    def match(
        self,
        candidate_vector: np.ndarray,
        top_k: int = config.DEFAULT_TOP_K,
        candidate_pool_size: int = config.DEFAULT_CANDIDATE_POOL_SIZE,
    ) -> tuple[list[dict], MatchInfo]:
        """Two-stage semantic retrieval.

        Stage 1: Compute cosine similarity against all stored job vectors
                 and retrieve the top ``candidate_pool_size`` candidates.
        Stage 2: Remove duplicates (by duplicate_signature) and return
                 the top ``top_k`` unique semantic matches.

        Skill matching is NOT performed here — the caller (CLI) handles
        that as a separate, lightweight step on the returned results.

        Returns:
            (results, match_info) where results is a list of dicts
            (one per matched job) and match_info contains debug stats.

        Raises:
            JobMatcherError: if top_k/candidate_pool_size are invalid,
                or if the candidate vector's dimensionality does not
                match the stored job vectors.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise JobMatcherError(f"Invalid top_k value: {top_k!r}. top_k must be a positive integer.")
        if not isinstance(candidate_pool_size, int) or candidate_pool_size <= 0:
            raise JobMatcherError(
                f"Invalid candidate_pool_size: {candidate_pool_size!r}. Must be a positive integer."
            )
        if candidate_pool_size < top_k:
            candidate_pool_size = top_k

        candidate_vector = np.asarray(candidate_vector, dtype=np.float32).reshape(1, -1)
        if candidate_vector.shape[1] != self.embedding_dim:
            raise JobMatcherError(
                f"Candidate embedding dimension ({candidate_vector.shape[1]}) does not "
                f"match job vector dimension ({self.embedding_dim}). This usually means "
                "the candidate was encoded with a different model than the one used to "
                "vectorize the jobs. Check 'model' in vectorization_config.json."
            )

        # ── Stage 1: Semantic retrieval ──────────────────────────────
        # Single cosine similarity computation against ALL vectors.
        similarities = cosine_similarity(candidate_vector, self.vectors)[0]

        # ── Stage 2: Pool retrieval + deduplication ──────────────────
        # Start with the requested pool size; expand if too many dupes.
        current_pool = min(candidate_pool_size, self.num_jobs)
        results: list[dict] = []
        seen_signatures: set[str] = set()

        while len(results) < top_k:
            ranked_indices = np.argsort(-similarities)[:current_pool]

            for row_index in ranked_indices:
                if any(r["_row_index"] == row_index for r in results):
                    continue

                row = self.metadata.iloc[row_index].to_dict()
                signature = row.get("duplicate_signature", "")

                if signature and signature in seen_signatures:
                    continue

                if signature:
                    seen_signatures.add(signature)

                row["similarity"] = float(similarities[row_index])
                row["_row_index"] = int(row_index)
                results.append(row)

                if len(results) >= top_k:
                    break

            # If we still need more, expand the pool or give up.
            if len(results) < top_k:
                if current_pool >= self.num_jobs:
                    break  # Exhausted the entire dataset
                current_pool = min(current_pool * 2, self.num_jobs)

        # Assign final rank numbers.
        for rank, res in enumerate(results, start=1):
            res["rank"] = rank

        match_info = MatchInfo(
            total_jobs_searched=self.num_jobs,
            candidate_pool_size=candidate_pool_size,
            unique_candidates_found=len(seen_signatures),
            final_results=len(results),
            embedding_dim=self.embedding_dim,
        )

        return results, match_info

