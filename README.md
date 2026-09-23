# Smart Resume Analyzer & Job Matcher (V1.1)

A CLI-only Python tool that parses a resume, builds a normalized technical
skill profile, matches it against a precomputed set of job postings using
semantic embeddings + cosine similarity, and reports skill gaps for the
top matches.

There is no UI in V1.1. Everything runs from the command line.

---

## 1. Project Overview

Given a resume (PDF or DOCX), the system:

1. Extracts and cleans the resume text.
2. Heuristically splits it into sections (Skills, Projects, Education,
   Experience, Certifications, Hobbies).
3. Extracts technical skills from the Skills section, Experience section, **and independently**
   from the Projects section (a project description like *"Built an eye
   tracking app using Python, OpenCV and MediaPipe"* yields skills even if
   they're never listed under "Skills").
4. Combines explicit, project-derived, and experience-derived skills into one deduplicated technical skill profile.
5. Embeds that profile with a pretrained Sentence Transformer.
6. Compares it against a set of **precomputed** job embedding vectors using
   cosine similarity, returning the Top-K most similar jobs.
7. For those Top-K jobs only, reports matched/missing skills and how often
   each missing skill recurs across the Top-K.
8. Prints everything to the CLI, optionally saving the full result as JSON.

## 2. Problem Statement

Resumes and job descriptions are unstructured text. Manually comparing a
candidate's skills against dozens of job postings — and figuring out
which skills are actually missing — is slow and inconsistent. This tool
automates that comparison using semantic similarity rather than exact
keyword matching, so e.g. "built REST APIs with Django" can be compared
meaningfully against "Django developer needed."

## 3. Objectives

- Parse real PDF/DOCX resumes without assuming a fixed layout.
- Extract a normalized technical skill profile from both explicit skills
  and project descriptions.
- Represent both candidates and jobs in the same semantic embedding space.
- Rank jobs by cosine similarity, not a hand-tuned scoring formula.
- Report skill gaps only where they matter (the jobs the candidate is
  actually closest to), with honest, non-overstated language.

## 4. Architecture

```
                    RESUME
                       |
                       v
                Resume Parser              (src/resume_parser.py)
                       |
                       v
              Section Extraction           (src/analyzer.py)
                       |
            +----------+----------+
            |                     |
            v                     v
     Explicit Skills          Projects
                                  |
                                  v
                         Project Skill Analyzer   (src/analyzer.py)
                                  |
                                  v
                         Project-derived Skills
            |                     |
            +----------+----------+
                       |
                       v
                Combined Skills             (src/analyzer.py)
                       |
                       v
             Candidate Embedding            (src/embeddings.py)
                       |
                       v
                Cosine Similarity           (src/matcher.py)
                       |
                       v
              Precomputed Job Vectors       (scripts/vectorize_jobs.py, offline)
                       |
                       v
                  Top-K Jobs
                       |
                       v
              Skill Gap Analysis            (src/skill_gap.py)
```

`src/cli.py` wires all of the above together and formats the output; it
contains no extraction/matching logic itself.

Education and experience are extracted and kept as **structured metadata**
(`CandidateProfile.education` / `.experience`) but are **not** part of the
technical embedding — see section 8 ("Candidate Representation") below.

## 5. Data Flow

1. **Offline, once per dataset:** `scripts/vectorize_jobs.py` reads the raw
   job CSV, builds a matching text per job (title + description + skills),
   embeds every job, and saves `job_vectors.npy` + `jobs_metadata.csv` +
   `vectorization_config.json`.
2. **Online, once per resume:** `src/cli.py` parses the resume, builds one
   candidate embedding, loads the precomputed job vectors, computes cosine
   similarity against all of them, and reports the Top-K.

Job vectors are **never** recomputed when a resume is analyzed — that
would be needlessly slow and is unnecessary since the job dataset doesn't
change per resume.

## 6. The ML Component

V1 uses exactly one pretrained model:
[`sentence-transformers/all-MiniLM-L6-v2`](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html),
loaded through `src/embeddings.py::EmbeddingModel`. **No model is trained
from scratch.** The "ML/NLP component" in V1 is the use of this pretrained
model to generate semantic embeddings — there is no supervised
classifier, no fine-tuning, and no labeled candidate/job training data.

## 7. Why Sentence Transformers

Sentence Transformers give a single dense vector per text (candidate
profile or job posting) that captures semantic meaning rather than exact
words, so "Django REST APIs" and "backend web framework, RESTful
services" land close together in vector space even without shared
vocabulary. `all-MiniLM-L6-v2` is a small, fast, well-established general
purpose model, which is a good fit for a V1's CPU-only, no-training
constraint.

## 8. Why Cosine Similarity

Cosine similarity measures the angle between two vectors, ignoring their
magnitude — appropriate here because we only care about *directional*
(semantic) similarity between a candidate profile and a job, not the
length of either text. All embeddings are computed with
`normalize_embeddings=True`; for unit-length vectors, cosine similarity
and the plain dot product are mathematically identical (see
`src/embeddings.py`'s docstring). `src/matcher.py` still computes cosine
similarity explicitly via scikit-learn, rather than assuming a raw dot
product, purely for robustness against floating-point drift.

Similarity is reported as a **Similarity Score**, not a hiring
probability — the CLI never claims e.g. "87% chance of getting the job."

## 9. Why Vectors Are Precomputed

Embedding every job on every single resume run would be slow and wasteful
— the job dataset doesn't change between resumes. `scripts/vectorize_jobs.py`
computes each job's vector exactly once; `src/matcher.py` only ever loads
and compares against the saved `.npy` file.

## 10. Resume Parsing

`src/resume_parser.py` supports **PDF** (via PyMuPDF / `fitz`, using
`page.get_text("text", sort=True)`) and **DOCX** (via `python-docx`).

- File existence and extension are validated before parsing.
- If a PDF has no extractable text (e.g. a scanned/image-only PDF), the
  tool raises a clear error explaining that **OCR is not supported in
  V1** — it will never silently return an empty resume.
- `resume_parser.clean_text()` normalizes line endings/tabs, collapses
  repeated horizontal whitespace and excessive blank lines, while
  preserving genuine line breaks (which section extraction relies on).

## 11. Project Skill Extraction

`src/analyzer.analyze_projects()` runs the same skill-vocabulary matching
used on the Skills section against the Projects section text
independently, since job postings/resumes often mention technologies only
inside project descriptions (e.g. "OpenCV", "MediaPipe") and the job
dataset itself may not contain a "projects" concept to compare against.

## 12. Combined Technical Profile

`src/analyzer.combine_skills()` merges explicit + project-derived skills
into one deduplicated list (explicit skills first, in the vocabulary's
configured order, then any additional skills found only in projects).
This combined list is what gets embedded — see
`src/analyzer.candidate_embedding_text()`, which produces text like:

```
Technical Skills:
Python, Java, Git, OpenCV, MediaPipe, SQL

Experience:
...

Projects:
...
```

No arbitrary numeric weights (e.g. "skills = 60%, projects = 25%") are
used — once everything is folded into one embedding, such weights have no
direct interpretable meaning, so V1 keeps this simple by design.

## 13. Skill-Gap Analysis

Only performed for the Top-K matched jobs (never the whole dataset), via
`src/skill_gap.py`:

- `compute_skill_gap(candidate_skills, job_required_skills)` → matched /
  missing skills for one job.
- `compute_missing_frequency(...)` → how often each missing skill recurs
  across the Top-K jobs, e.g. `{"SQL": {"count": 3, "out_of": 5, "fraction": "3/5"}}`.

This is descriptive analysis only — missing a skill is never reported as
a guarantee of rejection.

## 14. Education / Experience Handling

Both are extracted by `extract_sections()` and kept as separate
structured metadata on `CandidateProfile` (`.education`, `.experience`).
They are **not** included in the technical embedding in V1 — see section
12. They exist today so a future version can implement structured
qualification/experience matching without having to re-parse resumes.

## 15. Dataset Preparation

The intended dataset is the Kaggle
[`ravindrasinghrana/job-description-dataset`](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset),
but **this project never hardcodes its column names.** You provide the
CSV; `scripts/inspect_dataset.py` shows you its actual columns, and
`scripts/vectorize_jobs.py` accepts every relevant column name as a
`--*-col` flag (defaults are provided, matching the likely Kaggle column
names, but are only defaults).

A small synthetic dataset, `data/raw/sample_jobs.csv` (15 fictional job
postings), ships with the project so the full pipeline can be exercised
without downloading the real dataset. **Do not treat results from this
synthetic dataset as validation of real-world matching quality** — it
exists purely so the code path can be tested end-to-end.

## 16. Installation

### Windows (PowerShell)

```powershell
git clone <this-repo-or-copy-the-folder>
cd smart_resume_analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows (Command Prompt)

```cmd
cd smart_resume_analyzer
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd smart_resume_analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first install pulls in `torch` as a dependency of
`sentence-transformers` — this can take a few minutes and needs a working
internet connection. The `all-MiniLM-L6-v2` model itself (~90 MB) is
downloaded automatically the first time it's used, and cached locally
afterwards (no internet needed on subsequent runs).

## 17. Windows Setup Notes

- All file paths in the code use `pathlib.Path`, not hardcoded
  `/`-style paths, so they work unmodified on Windows.
- `requirements.txt` and every `.py`/`.md`/`.json` file in this project
  are plain UTF-8 text with no BOM/NULL bytes.
- Run everything as a module (`python -m ...`) from the project root, as
  shown throughout this README, so package imports resolve the same way
  on Windows, macOS, and Linux.

## 18. Dataset Inspection

Before vectorizing your real dataset, confirm its actual column names:

```powershell
python -m scripts.inspect_dataset --csv data\raw\your_dataset.csv
```

This prints the column names, row count, a sample of rows, and null
counts per column, plus a ready-to-copy `vectorize_jobs` command.

## 19. Job Vectorization

Run this **once** per dataset (or whenever the dataset changes):

```powershell
python -m scripts.vectorize_jobs --csv data\raw\your_dataset.csv --title-col "Job Title" --description-col "Job Description" --skills-col "skills" --role-col "Role" --responsibilities-col "Responsibilities"
```

Useful flags:

- `--qualifications-col`, `--experience-col` —
  kept as metadata only, by default.
- `--include-responsibilities` — fold the responsibilities text into the
  semantic matching text too (off by default).
- `--output-dir` — where to write `job_vectors.npy` / `jobs_metadata.csv`
  / `vectorization_config.json` (default: `data/vectors/`).
- `--model-name` — override the embedding model (default:
  `sentence-transformers/all-MiniLM-L6-v2`).
- `--limit` (or `--max-rows`) — cap the number of rows processed (e.g. `--limit 10000` to test on 10,000 rows before full vectorization).

To try the whole pipeline immediately with the bundled synthetic dataset:

```powershell
python -m scripts.vectorize_jobs --csv data\raw\sample_jobs.csv
```

## 20. Resume Matching

```powershell
python -m src.cli --resume .\sample\sample_resume.txt --top-k 5
```

or, saving the full structured result as JSON:

```powershell
python -m src.cli --resume .\sample\sample_resume.txt --top-k 5 --output results.json
```

All CLI options:

| Flag                      | Default                                  | Description |
|---------------------------|-------------------------------------------|-------------|
| `--resume`                | *(required)*                             | Path to a `.pdf` or `.docx` resume |
| `--top-k`                 | `5`                                       | Number of top jobs to return (clamped to the dataset size if larger) |
| `--output`                | *(none)*                                  | Optional path to save the full result as JSON |
| `--vectors`               | `data/vectors/job_vectors.npy`           | Precomputed job vectors |
| `--metadata`              | `data/vectors/jobs_metadata.csv`         | Job metadata aligned to the vectors |
| `--vectorization-config`  | `data/vectors/vectorization_config.json` | Records which model/columns were used |
| `--model-name`            | *(from vectorization_config.json)*       | Override the embedding model used to encode the resume |
| `--decimals`              | `4`                                       | Decimal places for similarity scores |

## 21. CLI Example

```
-----------------------------------
SMART RESUME ANALYZER & JOB MATCHER
-----------------------------------

Resume: data/test_resume.docx

Explicit Skills:
Python, Java, SQL, MySQL, REST API, Git, Linux, Docker

Project-derived Skills:
Python, MongoDB, React, Node.js, Flask, GitHub, Docker, OpenCV, MediaPipe, Raspberry Pi

Combined Technical Skills:
Python, Java, SQL, MySQL, REST API, Git, Linux, Docker, MongoDB, React, Node.js, Flask, GitHub, OpenCV, MediaPipe, Raspberry Pi

Top 5 Job Matches:
1. Database Administrator
   Similarity: 0.7820
...

Skill Gap Analysis (Top-K jobs):

1. Database Administrator
   Matched skills: SQL, MySQL, Linux
   Missing skills: PostgreSQL, Redis
...

Missing Skills Across Top-K:
   PostgreSQL           2/5
   ...

Note: Similarity is a semantic similarity measurement, not a hiring probability.
Missing skills are descriptive, not a guarantee of rejection.
```

*(This example's exact numbers came from a validation run using a
placeholder embedding backend — see "Limitations" below. Real similarity
values will differ once run with the actual pretrained model.)*

## 22. JSON Output

Passing `--output results.json` writes:

```json
{
  "resume": {
    "file": "...",
    "skills": ["Python", "..."],
    "project_derived_skills": ["..."],
    "combined_skills": ["..."],
    "education": "...",
    "experience": "...",
    "certifications": "...",
    "hobbies": "..."
  },
  "matches": [
    {
      "rank": 1,
      "job_title": "...",
      "similarity": 0.7821,
      "skills": "...",
      "qualifications": "...",
      "experience": "...",
      "matched_skills": ["..."],
      "missing_skills": ["..."]
    }
  ],
  "missing_skill_frequency": {
    "SQL": {"count": 3, "out_of": 5, "fraction": "3/5"}
  }
}
```

## 23. Testing

```powershell
pip install -r requirements.txt   # includes pytest
python -m pytest tests -v
```

Tests cover skill normalization, explicit/project skill extraction,
duplicate removal, section parsing, DOCX parsing, cosine-similarity
matching/validation logic, skill-gap computation, and missing-skill
frequency — all without requiring the embedding model download (see
"What Was Actually Tested" below). The one PDF-parsing test
(`tests/test_resume_parser.py::test_extract_text_from_pdf`) is
automatically skipped in any environment without PyMuPDF installed,
rather than being silently omitted.

## 24. Project Structure

```
smart_resume_analyzer/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py           # vocab, section headings, defaults — no logic
│   ├── resume_parser.py    # PDF/DOCX -> text
│   ├── analyzer.py         # sections, skills, candidate profile
│   ├── embeddings.py       # Sentence Transformer wrapper
│   ├── matcher.py          # cosine similarity + Top-K
│   ├── skill_gap.py        # matched/missing + frequency
│   └── cli.py              # entry point (python -m src.cli)
│
├── scripts/
│   ├── __init__.py
│   ├── inspect_dataset.py  # python -m scripts.inspect_dataset
│   └── vectorize_jobs.py   # python -m scripts.vectorize_jobs
│
├── data/
│   ├── raw/                # your dataset goes here (sample_jobs.csv is tracked)
│   ├── processed/          # scratch space (gitignored)
│   └── vectors/            # generated by vectorize_jobs.py (gitignored)
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_resume_parser.py
│   ├── test_skill_gap.py
│   └── test_matcher.py
│
└── sample/
    └── sample_resume.txt   # plain-text resume used to test analysis logic directly
```

## 25. Limitations

- **V1 heuristics, not NLP models, for structure.** Section extraction
  looks for whole-line heading matches from a configured list
  (`src/config.py::SECTION_HEADINGS`); unusual resume layouts (multi-column
  PDFs, creative/graphic resumes, uncommon heading names) may not be
  split correctly.
- **Skill vocabulary is fixed and conservative.** Only technologies listed
  in `src/config.py::SKILL_VOCABULARY` are recognized. A few short,
  ambiguous abbreviations (e.g. bare "ML", "CV", "TF") were deliberately
  excluded to avoid false positives (e.g. "CV" often means "curriculum
  vitae," not "Computer Vision").
- **No OCR.** Image-only/scanned PDFs are explicitly rejected with a clear
  error rather than silently parsed as empty.
- **No fuzzy/typo-tolerant matching.** Skill matching is exact (post
  alias-normalization), so unlisted spelling variants won't be caught.
- **This sandbox could not install `sentence-transformers`, `torch`, or
  `pymupdf` (no network/model-download access here) — see "What Was
  Actually Tested" below for exactly what that means for this delivery.**

## 26. V1 Design Decisions

- No supervised model is trained; a pretrained Sentence Transformer
  provides the only "ML" in the pipeline.
- No numeric weighting scheme (skills vs. education vs. experience) is
  introduced — everything technical goes into one embedding, and
  education/experience are kept as separate, un-weighted metadata.
- Skill-gap analysis is restricted to the Top-K jobs, never the whole
  dataset, both for relevance and to keep runtime reasonable on large
  datasets.
- Job vectors are computed once and reused, never recomputed per resume.

## 27. Future V2 Improvements

- Better resume parsing (multi-column layouts, OCR fallback for scanned
  PDFs).
- A richer, extensible skill taxonomy with fuzzy/typo-tolerant alias
  matching.
- Structured qualification matching (education) and experience matching
  (years, seniority) as additional, separately-weighted signals.
- Cross-Encoder reranking of the Top-K for higher-precision ranking.
- Supervised ranking/classification using labeled candidate-job outcome
  data, with proper evaluation metrics.
- A UI (explicitly out of scope for V1).

---

## What Was Actually Tested (and What Wasn't)

This project was built and validated in a sandboxed environment with
**no internet access**, so `sentence-transformers`, `torch`, and
`pymupdf` (PyMuPDF) could not be installed here — they must be installed
via `pip install -r requirements.txt` on a machine with internet access
before first use (this is normal and only needs to happen once; the
embedding model itself is then cached locally).

**What was verified in this environment, and how:**

1. **`python -m compileall src scripts tests`** — every `.py` file
   compiles with no syntax errors.
2. **Unit tests** (`tests/test_analyzer.py`, `tests/test_skill_gap.py`,
   `tests/test_matcher.py`, and the DOCX/error-handling parts of
   `tests/test_resume_parser.py`) — **29 tests passed, 0 failed, 1
   skipped.** The single skip is `test_extract_text_from_pdf`, which
   requires PyMuPDF; it is *designed* to skip cleanly rather than fail
   or be silently omitted when that dependency is absent.
3. **`scripts/inspect_dataset.py`** — run directly against
   `data/raw/sample_jobs.csv`; output verified (columns, row count,
   sample rows, null counts).
4. **End-to-end pipeline plumbing** (`scripts/vectorize_jobs.py` →
   `src/matcher.py` → `src/cli.py`, including JSON output) — run
   successfully against the synthetic dataset and a DOCX version of
   `sample/sample_resume.txt`, using a **placeholder embedding backend**
   (a small deterministic stand-in for `sentence-transformers`, used only
   because the real package could not be installed in this sandbox). This
   validated: CSV column validation, skill extraction integration,
   `.npy`/CSV/JSON file generation, the `len(vectors) == len(metadata)`
   check, cosine-similarity ranking and Top-K clamping, dimension-mismatch
   detection, skill-gap computation, missing-skill frequency aggregation,
   and the full CLI output format — but **not** the real model's semantic
   embedding quality.
5. **CLI error handling** — verified directly: missing resume file,
   unsupported file extension, missing job-vector files, `--top-k 0`,
   `--top-k` larger than the dataset (clamped, not an error), and a
   candidate/job embedding-dimension mismatch all produce the intended
   clear error messages (or clamped behavior) and correct exit codes.

**What could not be run here, and needs to be verified on your machine
after `pip install -r requirements.txt`:**

- Parsing an actual PDF resume (`src/resume_parser.py`'s PyMuPDF path) —
  the DOCX path was fully tested; the PDF path uses the same
  `extract_text`/`clean_text` functions and the same "no extractable
  text" safeguard, but was not exercised against a real PDF file here.
- Real embedding quality / real similarity scores from
  `sentence-transformers/all-MiniLM-L6-v2` — everything upstream and
  downstream of "call `.encode(...)`" was validated; the actual model
  call itself was not.

To confirm both on your machine in one step, run:

```powershell
python -m scripts.vectorize_jobs --csv data\raw\sample_jobs.csv
python -m src.cli --resume <a real PDF or DOCX resume of yours> --top-k 5
python -m pytest tests -v
```

If PyMuPDF and sentence-transformers installed correctly, all 30 tests
should now pass (0 skipped), and the CLI should print real similarity
scores.

## Assumptions Requiring Your Input

- **The real Kaggle dataset's exact column names.** Run
  `scripts/inspect_dataset.py` on your downloaded CSV and pass the
  correct `--title-col`/`--description-col`/`--skills-col`/etc. to
  `scripts/vectorize_jobs.py` if they differ from the defaults
  (`"Job Title"`, `"Job Description"`, `"skills"`, `"Qualifications"`,
  `"Experience"`, `"Responsibilities"`).
- **A real resume file** (PDF or DOCX) to test end-to-end matching
  against your vectorized dataset.
