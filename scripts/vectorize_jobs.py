"""Preprocess a job-description CSV and precompute job embedding vectors.

This is a separate, one-time (per dataset) program — job vectors are
never regenerated when a resume is matched (see ``src/matcher.py``).

Column names are fully configurable via CLI flags because the actual
downloaded CSV's column names must never be assumed (run
``scripts/inspect_dataset.py`` first to confirm them).

For V1, the primary semantic job-matching text is built from:

    Job Title + Job Description + Skills

Qualifications, Experience, and Responsibilities are kept as separate
metadata (not embedded by default) so they don't accidentally dominate
the technical similarity score — they are reserved for future structured
matching. Responsibilities can optionally be folded into the matching
text with ``--include-responsibilities``.

Usage (from the project root):

    python -m scripts.vectorize_jobs --csv data/raw/your_dataset.csv ^
        --title-col "Job Title" --description-col "Job Description" --skills-col "skills"

Outputs (default location: data/vectors/):

    job_vectors.npy            - float32 matrix, shape (num_jobs, embedding_dim)
    jobs_metadata.csv          - one row per job, aligned by row index with job_vectors.npy
    vectorization_config.json  - the exact configuration used to produce the above
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import analyzer, config
from src.embeddings import EmbeddingModel


class VectorizationError(Exception):
    """Raised for any problem preparing or vectorizing the job dataset."""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess a job-description CSV and precompute job embedding vectors."
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to the raw job-description CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.VECTORS_DIR,
        help=f"Directory to write outputs to (default: {config.VECTORS_DIR}).",
    )
    parser.add_argument("--title-col", default="Job Title", help="Column name for the job title.")
    parser.add_argument("--role-col", default="Role", help="Column name for the job role.")
    parser.add_argument(
        "--description-col", default="Job Description", help="Column name for the job description."
    )
    parser.add_argument("--skills-col", default="skills", help="Column name for the job's listed skills.")
    parser.add_argument(
        "--qualifications-col",
        default="Qualifications",
        help="Column name for qualifications (kept as metadata only).",
    )
    parser.add_argument(
        "--experience-col",
        default="Experience",
        help="Column name for experience requirements (kept as metadata only).",
    )
    parser.add_argument(
        "--responsibilities-col",
        default="Responsibilities",
        help="Column name for responsibilities (kept as metadata; optionally included in matching text).",
    )
    parser.add_argument(
        "--include-responsibilities",
        action="store_true",
        help="Fold the responsibilities column into the semantic matching text (default: off).",
    )
    parser.add_argument(
        "--model-name",
        default=config.DEFAULT_MODEL_NAME,
        help=f"Sentence Transformer model name (default: {config.DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--max-rows", "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows processed (useful for quick tests).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    return parser


def _clean_cell(value: object) -> str:
    """Turn any CSV cell into a clean string, treating NaN/None as ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise VectorizationError(f"Dataset CSV not found: {csv_path}")
    try:
        return pd.read_csv(csv_path)
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")
    except Exception as exc:
        raise VectorizationError(f"Failed to read CSV '{csv_path}': {exc}") from exc


def _validate_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        available = ", ".join(repr(column) for column in dataframe.columns)
        raise VectorizationError(
            f"Required column(s) not found in the dataset: {missing}.\n"
            f"Available columns: {available}\n"
            "Run 'python -m scripts.inspect_dataset --csv <your_csv>' to confirm the "
            "correct column names, then pass them via --title-col/--description-col/--skills-col/etc."
        )


def vectorize_jobs(args: argparse.Namespace) -> None:
    dataframe = _load_dataframe(args.csv)

    required_columns = [args.title_col, args.description_col, args.skills_col]
    _validate_columns(dataframe, required_columns)

    optional_columns = {
        "role": args.role_col,
        "qualifications": args.qualifications_col,
        "experience": args.experience_col,
        "responsibilities": args.responsibilities_col,
    }
    for label, column_name in optional_columns.items():
        if column_name not in dataframe.columns:
            print(
                f"Warning: optional column {column_name!r} ({label}) not found — "
                "that metadata field will be left blank for every job.",
                file=sys.stderr,
            )

    if args.max_rows is not None:
        dataframe = dataframe.head(args.max_rows)

    if len(dataframe) == 0:
        raise VectorizationError("Dataset has 0 rows after loading — nothing to vectorize.")

    titles = dataframe[args.title_col].map(_clean_cell)
    descriptions = dataframe[args.description_col].map(_clean_cell)
    skills_raw = dataframe[args.skills_col].map(_clean_cell)
    qualifications = (
        dataframe[args.qualifications_col].map(_clean_cell)
        if args.qualifications_col in dataframe.columns
        else pd.Series([""] * len(dataframe))
    )
    experience = (
        dataframe[args.experience_col].map(_clean_cell)
        if args.experience_col in dataframe.columns
        else pd.Series([""] * len(dataframe))
    )
    responsibilities = (
        dataframe[args.responsibilities_col].map(_clean_cell)
        if args.responsibilities_col in dataframe.columns
        else pd.Series([""] * len(dataframe))
    )

    roles = (
        dataframe[args.role_col].map(_clean_cell)
        if args.role_col in dataframe.columns
        else pd.Series([""] * len(dataframe))
    )

    matching_texts: list[str] = []
    extracted_skills_column: list[str] = []
    duplicate_signatures: list[str] = []
    
    for title, role, description, skills_text, responsibilities_text in zip(
        titles, roles, descriptions, skills_raw, responsibilities
    ):
        text_parts = []
        if title: text_parts.append(f"Job Title: {title}")
        if role: text_parts.append(f"Role: {role}")
        if description: text_parts.append(f"Description: {description}")
        if skills_text: text_parts.append(f"Required Skills: {skills_text}")
        if args.include_responsibilities and responsibilities_text:
            text_parts.append(f"Responsibilities: {responsibilities_text}")
            
        matching_text = "\n".join(text_parts)
        matching_texts.append(matching_text)
        
        # Simple duplicate signature string
        duplicate_signatures.append(matching_text)

        # Extract canonical skills from title + description + skills so the
        # skill-gap comparison uses the same vocabulary as candidate skills.
        source_text = " ".join([title, description, skills_text])
        extracted = analyzer.extract_skills(source_text)
        extracted_skills_column.append(config.SKILL_LIST_SEPARATOR.join(extracted))

    print(f"Encoding {len(matching_texts)} job(s) with model '{args.model_name}'...")
    embedding_model = EmbeddingModel(model_name=args.model_name)
    vectors = embedding_model.encode_batch(matching_texts, show_progress_bar=True)

    metadata = pd.DataFrame(
        {
            config.JOB_ORIGINAL_INDEX_COLUMN: dataframe.index,
            config.JOB_TITLE_COLUMN: titles,
            "Role": roles,
            config.JOB_DESCRIPTION_COLUMN: descriptions,
            config.JOB_SKILLS_RAW_COLUMN: skills_raw,
            config.JOB_QUALIFICATIONS_COLUMN: qualifications,
            config.JOB_EXPERIENCE_COLUMN: experience,
            config.JOB_RESPONSIBILITIES_COLUMN: responsibilities,
            config.JOB_EXTRACTED_SKILLS_COLUMN: extracted_skills_column,
            "duplicate_signature": duplicate_signatures,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = output_dir / "job_vectors.npy"
    metadata_path = output_dir / "jobs_metadata.csv"
    vectorization_config_path = output_dir / "vectorization_config.json"

    np.save(vectors_path, vectors)
    metadata.to_csv(metadata_path, index=False, encoding="utf-8")

    vectorization_config = {
        "version": "v1.1",
        "engine_version": config.ENGINE_VERSION,
        "model_name": args.model_name,
        "source_csv": str(Path(args.csv).resolve()),
        "embedded_fields": [
            args.title_col,
            args.role_col,
            args.description_col,
            args.skills_col,
        ] + ([args.responsibilities_col] if args.include_responsibilities else []),
        "columns": {
            "title": args.title_col,
            "role": args.role_col,
            "description": args.description_col,
            "skills": args.skills_col,
            "qualifications": args.qualifications_col,
            "experience": args.experience_col,
            "responsibilities": args.responsibilities_col,
        },
        "include_responsibilities_in_matching_text": args.include_responsibilities,
        "num_rows": int(len(dataframe)),
        "embedding_dim": int(vectors.shape[1]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(vectorization_config_path, "w", encoding="utf-8") as file_handle:
        json.dump(vectorization_config, file_handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataframe)} job vectors (dim={vectors.shape[1]}) to: {vectors_path}")
    print(f"Saved job metadata to: {metadata_path}")
    print(f"Saved vectorization config to: {vectorization_config_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        vectorize_jobs(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
