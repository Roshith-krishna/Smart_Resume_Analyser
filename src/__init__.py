"""Smart Resume Analyzer & Job Matcher — core package.

This package contains the V1.2 implementation:

- ``resume_parser``: extract raw text from PDF / DOCX resumes.
- ``analyzer``: heuristic section extraction, skill vocabulary matching,
  project-derived skill analysis, and candidate profile assembly.
- ``embeddings``: a thin, reusable wrapper around a pretrained
  Sentence Transformer model.
- ``matcher``: cosine-similarity based job matching against precomputed
  job vectors.
- ``skill_gap``: matched/missing skill analysis for the top-K matches.
- ``cli``: the command line entry point that wires everything together.
"""

__version__ = "1.0.0"
