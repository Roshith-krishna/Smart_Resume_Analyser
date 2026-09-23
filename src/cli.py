"""Smart Resume Analyzer & Job Matcher — command line entry point (V1.2).

Usage (from the project root):

    python -m src.cli --resume sample/sample_resume.txt --top-k 5
    python -m src.cli --resume sample/sample_resume.txt --top-k 5 --verbose
    python -m src.cli --resume sample/sample_resume.txt --top-k 5 --candidate-pool-size 100

This module only wires together the other modules (resume_parser,
analyzer, embeddings, matcher, skill_gap) and formats their output — it
contains no matching or extraction logic of its own.

V1.2 two-stage architecture:
  Stage 1 (matcher): cosine similarity → top candidate_pool_size → dedup
  Stage 2 (here):    skill matching on final top-k only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import analyzer, config, resume_parser, skill_gap
from src.embeddings import EmbeddingModel
from src.matcher import JobMatcher, JobMatcherError


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Smart Resume Analyzer & Job Matcher (V1.2, CLI-only).",
    )
    parser.add_argument("--resume", required=True, type=Path, help="Path to a .pdf or .docx resume file.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Number of top matching jobs to display (default: {config.DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=config.DEFAULT_CANDIDATE_POOL_SIZE,
        help=(
            f"Number of semantic candidates to retrieve before deduplication and "
            f"skill matching (default: {config.DEFAULT_CANDIDATE_POOL_SIZE})."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug/diagnostic information (embedding dim, pool stats, etc.).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the full structured result as JSON.",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=config.DEFAULT_JOB_VECTORS_PATH,
        help=f"Path to precomputed job vectors (default: {config.DEFAULT_JOB_VECTORS_PATH}).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=config.DEFAULT_JOBS_METADATA_PATH,
        help=f"Path to job metadata CSV (default: {config.DEFAULT_JOBS_METADATA_PATH}).",
    )
    parser.add_argument(
        "--vectorization-config",
        type=Path,
        default=config.DEFAULT_VECTORIZATION_CONFIG_PATH,
        help=f"Path to vectorization_config.json (default: {config.DEFAULT_VECTORIZATION_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Sentence Transformer model name to encode the resume with. "
        "Defaults to the model recorded in vectorization_config.json, falling back to "
        f"{config.DEFAULT_MODEL_NAME} if that file is absent.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=config.DEFAULT_SIMILARITY_DECIMALS,
        help=f"Decimal places for similarity scores (default: {config.DEFAULT_SIMILARITY_DECIMALS}).",
    )
    return parser


def _print_header() -> None:
    banner = "SMART RESUME ANALYZER & JOB MATCHER"
    print("-" * len(banner))
    print(banner)
    print("-" * len(banner))


def _print_skill_list(label: str, skills: list[str]) -> None:
    print(f"\n{label}:")
    print(", ".join(skills) if skills else "(none found)")


def run(args: argparse.Namespace) -> dict:
    """Run the full two-stage pipeline and return the structured result dict.

    Stage 1 (matcher): semantic retrieval + deduplication
    Stage 2 (here):    skill matching on final top-k results only

    Raises on any unrecoverable error (resume_parser.ResumeParsingError,
    matcher.JobMatcherError, etc.) — the caller (main) is responsible for
    catching and reporting these cleanly.
    """
    _print_header()

    print(f"\nResume: {args.resume}")
    raw_text = resume_parser.extract_text(args.resume)
    cleaned_text = resume_parser.clean_text(raw_text)

    profile = analyzer.build_candidate_profile(cleaned_text)

    _print_skill_list("Explicit Skills", profile.skills)
    _print_skill_list("Project-derived Skills", profile.project_derived_skills)
    _print_skill_list("Experience-derived Skills", profile.experience_derived_skills)
    _print_skill_list("Combined Technical Skills", profile.combined_skills)

    if not profile.combined_skills:
        print(
            "\nWarning: no technical skills were detected from the vocabulary in "
            "src/config.py. Job matching will proceed using an embedding of an "
            "empty skill profile, which will likely produce low-quality matches.",
            file=sys.stderr,
        )

    # ── Load job vectors ─────────────────────────────────────────
    job_matcher = JobMatcher.load(
        vectors_path=args.vectors,
        metadata_path=args.metadata,
        vectorization_config_path=args.vectorization_config,
    )

    model_name = args.model_name or job_matcher.vectorization_config.get("model_name", config.DEFAULT_MODEL_NAME)
    embedding_model = EmbeddingModel(model_name=model_name)
    candidate_text = analyzer.candidate_embedding_text(profile)
    candidate_vector = embedding_model.encode(candidate_text)

    # ── Stage 1: Semantic retrieval + deduplication ───────────────
    matches, match_info = job_matcher.match(
        candidate_vector,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
    )

    # ── Verbose / debug output ───────────────────────────────────
    if args.verbose:
        print(f"\n--- Debug Info ---")
        print(f"Candidate embedding dim: {match_info.embedding_dim}")
        print(f"Total jobs searched: {match_info.total_jobs_searched:,}")
        print(f"Semantic candidate pool size: {match_info.candidate_pool_size}")
        print(f"Unique candidates after dedup: {match_info.unique_candidates_found}")
        print(f"Final results: {match_info.final_results}")
        print(f"-----------------")

    # ── Stage 2: Skill matching on final top-k only ──────────────
    print(f"\nTop Semantic Matches:")
    gaps: list[dict] = []
    for match in matches:
        title = match.get(config.JOB_TITLE_COLUMN, "(untitled job)")
        role = match.get("Role", "")
        similarity = match["similarity"]
        print(f"\n{match['rank']}. {title}")
        if role:
            print(f"   Role: {role}")
        print(f"   Similarity: {similarity:.{args.decimals}f}")

        # Skill matching — lightweight, only on these top-k results.
        job_skills_raw = match.get(config.JOB_EXTRACTED_SKILLS_COLUMN, "") or ""
        job_skills = [s for s in job_skills_raw.split(config.SKILL_LIST_SEPARATOR) if s]
        gap = skill_gap.compute_skill_gap(profile.combined_skills, job_skills)
        gaps.append(gap)

        matched_display = ", ".join(gap["matched_skills"]) if gap["matched_skills"] else "(none)"
        missing_display = ", ".join(gap["missing_skills"]) if gap["missing_skills"] else "(none)"

        print(f"   Matched: {matched_display}")
        print(f"   Missing: {missing_display}")

    missing_frequency = skill_gap.compute_missing_frequency([gap["missing_skills"] for gap in gaps])
    print(f"\nSkill gaps across top {len(matches)}:")
    if missing_frequency:
        for skill, stats in missing_frequency.items():
            print(f"   {skill:<20} {stats['fraction']}")
    else:
        print("   (none — combined skills cover every top job's extracted requirements)")

    print(
        "\nNote: Similarity is a semantic similarity measurement, not a hiring "
        "probability. Missing skills are descriptive, not a guarantee of rejection."
    )

    result = {
        "resume": {
            "file": str(args.resume),
            "skills": profile.skills,
            "project_derived_skills": profile.project_derived_skills,
            "experience_derived_skills": profile.experience_derived_skills,
            "combined_skills": profile.combined_skills,
            "education": profile.education,
            "experience": profile.experience,
            "certifications": profile.certifications,
            "hobbies": profile.hobbies,
        },
        "match_info": {
            "total_jobs_searched": match_info.total_jobs_searched,
            "candidate_pool_size": match_info.candidate_pool_size,
            "unique_candidates_found": match_info.unique_candidates_found,
            "final_results": match_info.final_results,
            "embedding_dim": match_info.embedding_dim,
        },
        "matches": [
            {
                "rank": match["rank"],
                "job_title": match.get(config.JOB_TITLE_COLUMN, ""),
                "role": match.get("Role", ""),
                "similarity": round(match["similarity"], args.decimals),
                "skills": match.get(config.JOB_SKILLS_RAW_COLUMN, ""),
                "qualifications": match.get(config.JOB_QUALIFICATIONS_COLUMN, ""),
                "experience": match.get(config.JOB_EXPERIENCE_COLUMN, ""),
                "matched_skills": gap["matched_skills"],
                "missing_skills": gap["missing_skills"],
            }
            for match, gap in zip(matches, gaps)
        ],
        "missing_skill_frequency": missing_frequency,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = run(args)
    except (resume_parser.ResumeParsingError, JobMatcherError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # last-resort guard against unhandled failures
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        try:
            with open(args.output, "w", encoding="utf-8") as file_handle:
                json.dump(result, file_handle, indent=2, ensure_ascii=False)
            print(f"\nSaved full result to: {args.output}")
        except Exception as exc:
            print(f"\nError: failed to save output JSON to {args.output}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
