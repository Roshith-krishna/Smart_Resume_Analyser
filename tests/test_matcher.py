"""Tests for the V1.2 two-stage JobMatcher.

Uses small synthetic vectors/metadata — no real model or dataset needed.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.matcher import JobMatcher, JobMatcherError, MatchInfo


# ── Helpers ──────────────────────────────────────────────────────────


def _make_store(
    num_jobs: int = 20,
    dim: int = 8,
    *,
    duplicate_indices: list[tuple[int, int]] | None = None,
    version: str = "v1.1",
    config_overrides: dict | None = None,
) -> tuple[Path, Path, Path]:
    """Create a minimal vector store on disk and return the three paths.

    ``duplicate_indices`` is a list of (src, dst) pairs; each dst row's
    duplicate_signature will be copied from src so the dedup logic can
    be tested.
    """
    tmp = Path(tempfile.mkdtemp())

    rng = np.random.default_rng(42)
    vectors = rng.random((num_jobs, dim), dtype=np.float32)
    # Normalize so cosine similarity is well-defined.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    titles = [f"Job_{i}" for i in range(num_jobs)]
    roles = [f"Role_{i}" for i in range(num_jobs)]
    sigs = [f"sig_{i}" for i in range(num_jobs)]
    extracted = ["Python|SQL" if i % 2 == 0 else "Java|REST API" for i in range(num_jobs)]

    if duplicate_indices:
        for src, dst in duplicate_indices:
            sigs[dst] = sigs[src]

    metadata = pd.DataFrame({
        "original_index": list(range(num_jobs)),
        "job_title": titles,
        "Role": roles,
        "job_description": ["desc"] * num_jobs,
        "skills_raw": ["python, sql"] * num_jobs,
        "qualifications": ["BSc"] * num_jobs,
        "experience": ["2 yrs"] * num_jobs,
        "responsibilities": ["coding"] * num_jobs,
        "extracted_skills": extracted,
        "duplicate_signature": sigs,
    })

    vectors_path = tmp / "job_vectors.npy"
    metadata_path = tmp / "jobs_metadata.csv"
    config_path = tmp / "vectorization_config.json"

    np.save(vectors_path, vectors)
    metadata.to_csv(metadata_path, index=False)

    vconfig = {
        "version": version,
        "engine_version": "test",
        "model_name": "test-model",
        "embedding_dim": dim,
        "num_rows": num_jobs,
    }
    if config_overrides:
        vconfig.update(config_overrides)
    with open(config_path, "w") as f:
        json.dump(vconfig, f)

    return vectors_path, metadata_path, config_path


# ── Tests: Loading & Validation ──────────────────────────────────────


class TestJobMatcherLoad:
    def test_load_valid_store(self):
        vp, mp, cp = _make_store(num_jobs=10, dim=8)
        matcher = JobMatcher.load(vp, mp, cp)
        assert matcher.num_jobs == 10
        assert matcher.embedding_dim == 8

    def test_missing_vectors_file(self):
        _, mp, cp = _make_store()
        with pytest.raises(JobMatcherError, match="Job vectors file not found"):
            JobMatcher.load(Path("/nonexistent/vectors.npy"), mp, cp)

    def test_missing_metadata_file(self):
        vp, _, cp = _make_store()
        with pytest.raises(JobMatcherError, match="Job metadata file not found"):
            JobMatcher.load(vp, Path("/nonexistent/metadata.csv"), cp)

    def test_missing_config_file(self):
        vp, mp, _ = _make_store()
        with pytest.raises(JobMatcherError, match="Vectorization config not found"):
            JobMatcher.load(vp, mp, Path("/nonexistent/config.json"))

    def test_incompatible_version(self):
        vp, mp, cp = _make_store(version="v0.5")
        with pytest.raises(JobMatcherError, match="Incompatible vector store version"):
            JobMatcher.load(vp, mp, cp)

    def test_v1_1_version_accepted(self):
        vp, mp, cp = _make_store(version="v1.1")
        matcher = JobMatcher.load(vp, mp, cp)
        assert matcher.num_jobs > 0

    def test_v1_2_version_accepted(self):
        vp, mp, cp = _make_store(version="v1.2")
        matcher = JobMatcher.load(vp, mp, cp)
        assert matcher.num_jobs > 0

    def test_vector_metadata_length_mismatch(self):
        """Vector count != metadata row count should raise."""
        vp, mp, cp = _make_store(num_jobs=10)
        # Overwrite vectors with wrong count
        np.save(vp, np.random.rand(5, 8).astype(np.float32))
        with pytest.raises(JobMatcherError, match="row-count mismatch"):
            JobMatcher.load(vp, mp, cp)

    def test_vector_dimension_mismatch_with_config(self):
        """Config says dim=8 but vectors have dim=16 should raise."""
        vp, mp, cp = _make_store(num_jobs=5, dim=16, config_overrides={"embedding_dim": 8})
        with pytest.raises(JobMatcherError, match="Vector dimension mismatch"):
            JobMatcher.load(vp, mp, cp)

    def test_empty_store(self):
        vp, mp, cp = _make_store(num_jobs=0)
        # num_jobs=0 produces an empty dataframe and 0-row vectors
        np.save(vp, np.empty((0, 8), dtype=np.float32))
        pd.DataFrame(columns=[
            "original_index", "job_title", "Role", "job_description",
            "skills_raw", "qualifications", "experience", "responsibilities",
            "extracted_skills", "duplicate_signature",
        ]).to_csv(mp, index=False)
        with pytest.raises(JobMatcherError, match="empty"):
            JobMatcher.load(vp, mp, cp)


# ── Tests: Two-Stage Matching ────────────────────────────────────────


class TestTwoStageMatch:
    def _load_matcher(self, **kwargs):
        vp, mp, cp = _make_store(**kwargs)
        return JobMatcher.load(vp, mp, cp)

    def test_returns_tuple(self):
        """match() returns (results, match_info)."""
        matcher = self._load_matcher(num_jobs=10, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        result = matcher.match(candidate, top_k=3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        results, info = result
        assert isinstance(results, list)
        assert isinstance(info, MatchInfo)

    def test_top_k_respected(self):
        """Final results should have exactly top_k items (when enough unique jobs)."""
        matcher = self._load_matcher(num_jobs=20, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        results, info = matcher.match(candidate, top_k=5, candidate_pool_size=50)
        assert len(results) == 5
        assert info.final_results == 5

    def test_candidate_pool_size_in_match_info(self):
        """match_info should report the requested pool size."""
        matcher = self._load_matcher(num_jobs=20, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        _, info = matcher.match(candidate, top_k=3, candidate_pool_size=15)
        assert info.candidate_pool_size == 15

    def test_results_ranked_by_similarity(self):
        """Results should be in descending similarity order."""
        matcher = self._load_matcher(num_jobs=20, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        results, _ = matcher.match(candidate, top_k=5)
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_results_have_rank_field(self):
        matcher = self._load_matcher(num_jobs=10, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        results, _ = matcher.match(candidate, top_k=3)
        ranks = [r["rank"] for r in results]
        assert ranks == [1, 2, 3]

    def test_duplicate_filtering(self):
        """Duplicate signatures should be filtered out."""
        # Make jobs 1,2,3 duplicates of job 0
        dupes = [(0, 1), (0, 2), (0, 3)]
        matcher = self._load_matcher(num_jobs=10, dim=8, duplicate_indices=dupes)
        candidate = np.random.rand(8).astype(np.float32)
        results, info = matcher.match(candidate, top_k=5)
        # All results should have unique signatures
        sigs = [r.get("duplicate_signature") for r in results]
        assert len(sigs) == len(set(sigs))

    def test_pool_expansion_on_heavy_duplication(self):
        """When the pool is full of dupes, it should expand to find unique jobs."""
        # Make all but 5 jobs duplicates of job 0
        dupes = [(0, i) for i in range(1, 16)]
        matcher = self._load_matcher(num_jobs=20, dim=8, duplicate_indices=dupes)
        candidate = np.random.rand(8).astype(np.float32)
        results, _ = matcher.match(candidate, top_k=5, candidate_pool_size=5)
        # Should still find 5 unique results (jobs 0, 16, 17, 18, 19 are unique)
        assert len(results) == 5
        sigs = [r.get("duplicate_signature") for r in results]
        assert len(sigs) == len(set(sigs))

    def test_insufficient_unique_jobs(self):
        """When fewer unique jobs exist than top_k, return what's available."""
        # Only 3 unique signatures in 10 jobs
        dupes = [(0, i) for i in range(1, 4)] + [(4, i) for i in range(5, 8)]
        # unique: sig_0, sig_4, sig_8, sig_9 = 4 unique
        matcher = self._load_matcher(num_jobs=10, dim=8, duplicate_indices=dupes)
        candidate = np.random.rand(8).astype(np.float32)
        results, info = matcher.match(candidate, top_k=10)
        # Should return only the available unique jobs (4), not 10
        assert len(results) <= 10
        sigs = [r.get("duplicate_signature") for r in results]
        assert len(sigs) == len(set(sigs))

    def test_no_skill_gap_in_match_results(self):
        """match() should NOT compute skill gaps — that's the CLI's job."""
        matcher = self._load_matcher(num_jobs=10, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        results, _ = matcher.match(candidate, top_k=3)
        for r in results:
            assert "matched_skills" not in r
            assert "missing_skills" not in r

    def test_match_info_total_jobs(self):
        matcher = self._load_matcher(num_jobs=15, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        _, info = matcher.match(candidate, top_k=3)
        assert info.total_jobs_searched == 15
        assert info.embedding_dim == 8

    def test_dimension_mismatch_at_match_time(self):
        """Candidate with wrong dim should raise."""
        matcher = self._load_matcher(num_jobs=10, dim=8)
        wrong_dim = np.random.rand(16).astype(np.float32)
        with pytest.raises(JobMatcherError, match="dimension"):
            matcher.match(wrong_dim, top_k=3)

    def test_invalid_top_k(self):
        matcher = self._load_matcher(num_jobs=10, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        with pytest.raises(JobMatcherError, match="top_k"):
            matcher.match(candidate, top_k=0)
        with pytest.raises(JobMatcherError, match="top_k"):
            matcher.match(candidate, top_k=-1)

    def test_invalid_candidate_pool_size(self):
        matcher = self._load_matcher(num_jobs=10, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        with pytest.raises(JobMatcherError, match="candidate_pool_size"):
            matcher.match(candidate, top_k=3, candidate_pool_size=0)

    def test_pool_size_clamped_to_top_k(self):
        """If candidate_pool_size < top_k, it should be clamped up."""
        matcher = self._load_matcher(num_jobs=10, dim=8)
        candidate = np.random.rand(8).astype(np.float32)
        results, _ = matcher.match(candidate, top_k=5, candidate_pool_size=2)
        assert len(results) == 5
