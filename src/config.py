"""Central configuration for Smart Resume Analyzer & Job Matcher.

This module intentionally holds all "tunable" data in one place:

- the technical skill vocabulary (and its known spelling variants), so it
  can be extended without touching any matching logic;
- the resume section heading variants used by the heuristic section
  extractor;
- default paths, model name, and CLI defaults.

Nothing in here performs any matching itself — see ``src/analyzer.py``.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Project root = the directory that contains this file's parent (src/..).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
VECTORS_DIR: Path = DATA_DIR / "vectors"

DEFAULT_JOB_VECTORS_PATH: Path = VECTORS_DIR / "job_vectors.npy"
DEFAULT_JOBS_METADATA_PATH: Path = VECTORS_DIR / "jobs_metadata.csv"
DEFAULT_VECTORIZATION_CONFIG_PATH: Path = VECTORS_DIR / "vectorization_config.json"

# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

# Pretrained Sentence Transformer used for all semantic embeddings.
# See: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# CLI / matching defaults
# --------------------------------------------------------------------------

DEFAULT_TOP_K: int = 5
DEFAULT_CANDIDATE_POOL_SIZE: int = 50
DEFAULT_SIMILARITY_DECIMALS: int = 4
ENGINE_VERSION: str = "smart-resume-analyzer-v1.2"

# --------------------------------------------------------------------------
# Job metadata CSV column names (as written by scripts/vectorize_jobs.py
# and read by src/matcher.py, src/skill_gap.py, and src/cli.py). These are
# OUR generated metadata columns, not the raw dataset's column names —
# see scripts/vectorize_jobs.py's --title-col/--description-col/etc. for
# how raw dataset columns are mapped into these.
# --------------------------------------------------------------------------

JOB_ORIGINAL_INDEX_COLUMN: str = "original_index"
JOB_TITLE_COLUMN: str = "job_title"
JOB_DESCRIPTION_COLUMN: str = "job_description"
JOB_SKILLS_RAW_COLUMN: str = "skills_raw"
JOB_QUALIFICATIONS_COLUMN: str = "qualifications"
JOB_EXPERIENCE_COLUMN: str = "experience"
JOB_RESPONSIBILITIES_COLUMN: str = "responsibilities"
JOB_EXTRACTED_SKILLS_COLUMN: str = "extracted_skills"

# Separator used to pack the extracted canonical skill list into a single
# CSV cell (job_extracted_skills column).
SKILL_LIST_SEPARATOR: str = "|"

# --------------------------------------------------------------------------
# Resume section headings (heuristic section extraction)
# --------------------------------------------------------------------------
# Each key is the canonical section name used throughout the codebase.
# Each value is a list of heading phrases that are recognised, case
# insensitively, as the start of that section. Matching is done against a
# *whole line* (after stripping punctuation/decoration), not a substring
# of a paragraph, to keep false positives low. See
# ``analyzer.extract_sections`` for the exact algorithm.

SECTION_HEADINGS: dict[str, list[str]] = {
    "skills": [
        "skills",
        "technical skills",
        "technologies",
        "technical skills & tools",
        "technical skills and tools",
        "skills & tools",
        "core competencies",
        "key skills",
    ],
    "projects": [
        "projects",
        "project experience",
        "academic projects",
        "personal projects",
        "personal projects & contributions",
        "key projects",
    ],
    "education": [
        "education",
        "academic background",
        "educational qualifications",
        "academic qualifications",
        "education & qualifications",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "internship",
        "internships",
        "work history",
        "employment history",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "certifications & courses",
        "licenses & certifications",
        "courses & certifications",
    ],
    "hobbies": [
        "hobbies",
        "interests",
        "hobbies & interests",
        "hobbies and interests",
        "personal interests",
    ],
}

# The fixed, ordered set of section keys every candidate profile carries.
SECTION_KEYS: tuple[str, ...] = (
    "skills",
    "projects",
    "education",
    "experience",
    "certifications",
    "hobbies",
)

# --------------------------------------------------------------------------
# Technical skill vocabulary
# --------------------------------------------------------------------------
# Maps a canonical skill name -> list of surface-form aliases that should
# normalize to it. Aliases are matched case-insensitively with strict
# "not preceded/followed by an alphanumeric character" boundaries (see
# analyzer.py), so partial-word matches are avoided (e.g. "Java" will not
# match inside "JavaScript").
#
# Deliberately excluded: very short, highly ambiguous abbreviations (e.g.
# bare "ml", "cv", "np", "pd", "tf") that collide with common English
# words or unrelated abbreviations (e.g. "CV" = curriculum vitae). This
# vocabulary is intentionally conservative for V1; extend it as needed.

SKILL_VOCABULARY: dict[str, list[str]] = {
    "Python": ["python"],
    "Java": ["java"],
    "C": ["c"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c-sharp", "csharp"],
    "JavaScript": ["javascript", "java script"],
    "TypeScript": ["typescript"],
    "SQL": ["sql"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb", "mongo"],
    "Redis": ["redis"],
    "HTML": ["html", "html5"],
    "CSS": ["css", "css3"],
    "Tailwind CSS": ["tailwind css", "tailwindcss", "tailwind"],
    "React": ["react", "react.js", "reactjs"],
    "Node.js": ["node.js", "nodejs", "node js"],
    "Express": ["express.js", "expressjs", "express"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi", "fast api"],
    "Spring": ["spring boot", "springboot", "spring"],
    "REST API": ["rest api", "restful api", "rest apis", "restful"],
    "GraphQL": ["graphql"],
    "Git": ["git"],
    "GitHub": ["github"],
    "Linux": ["linux"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure"],
    "GCP": ["gcp", "google cloud platform", "google cloud"],
    "Machine Learning": ["machine learning"],
    "Deep Learning": ["deep learning"],
    "Artificial Intelligence": ["artificial intelligence"],
    "Data Science": ["data science"],
    "Statistics": ["statistics"],
    "NLP": ["nlp", "natural language processing"],
    "Computer Vision": ["computer vision"],
    "OpenCV": ["opencv", "open cv"],
    "MediaPipe": ["mediapipe", "media pipe"],
    "NumPy": ["numpy"],
    "Pandas": ["pandas"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "Keras": ["keras"],
    "Spark": ["apache spark", "pyspark", "spark"],
    "Hadoop": ["hadoop"],
    "Tableau": ["tableau"],
    "Power BI": ["power bi", "powerbi"],
    "Arduino": ["arduino"],
    "Raspberry Pi": ["raspberry pi", "raspberrypi"],
}
