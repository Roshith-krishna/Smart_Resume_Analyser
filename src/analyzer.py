"""Resume analysis: section extraction, skill extraction, and candidate
profile assembly.

This module implements steps 3–8 of the project's core architecture:

1. Heuristic section extraction (``extract_sections``).
2. Configurable-vocabulary skill extraction (``extract_skills``), used for
   both the explicit Skills section and for project descriptions.
3. Combining explicit + project-derived skills into one normalized,
   deduplicated technical skill profile (``combine_skills``).
4. Assembling the full structured candidate profile
   (``build_candidate_profile``) and the short text used to build the
   candidate's embedding (``candidate_embedding_text``).

Section extraction and skill extraction are both heuristic by design (see
each function's docstring) — V1 deliberately avoids a bespoke NER model in
favor of a documented, extensible, testable rule set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src import config

# --------------------------------------------------------------------------
# Section extraction
# --------------------------------------------------------------------------


def _normalize_heading(line: str) -> str:
    """Normalize a line for heading comparison: lowercase, strip
    surrounding decoration (colons, dashes, box-drawing, etc.), collapse
    internal whitespace, keep only letters/digits/spaces/&."""
    lowered = line.strip().lower()
    # Keep letters, digits, spaces and '&' (e.g. "Skills & Tools");
    # everything else (colons, dashes, bullets, box characters) is dropped.
    stripped = re.sub(r"[^a-z0-9 &]", " ", lowered)
    return re.sub(r"\s+", " ", stripped).strip()


# Precompute a flat lookup: normalized heading phrase -> section key.
_HEADING_LOOKUP: dict[str, str] = {
    _normalize_heading(variant): section
    for section, variants in config.SECTION_HEADINGS.items()
    for variant in variants
}

# A heading line is expected to be short (a label, not a sentence).
_MAX_HEADING_WORDS = 6


def _match_heading(line: str) -> str | None:
    """Return the section key if `line` is (heuristically) a section
    heading on its own, otherwise None.

    A line qualifies only if, once normalized, it *exactly* equals one of
    the configured heading phrases AND is short enough to plausibly be a
    label rather than a sentence that happens to contain a heading word.
    This whole-line, exact-match requirement (rather than substring
    matching) is what keeps false positives low.
    """
    normalized = _normalize_heading(line)
    if not normalized:
        return None
    if len(normalized.split()) > _MAX_HEADING_WORDS:
        return None
    return _HEADING_LOOKUP.get(normalized)


def extract_sections(text: str) -> dict[str, str]:
    """Split resume text into named sections using heuristic heading
    detection.

    Recognized section keys are exactly ``config.SECTION_KEYS``: skills,
    projects, education, experience, certifications, hobbies. Any section
    not found in the resume is returned as an empty string. Content that
    appears before the first recognized heading, or under an
    unrecognized heading, is not attributed to any section.

    This is a heuristic, not a guarantee: it recognizes common heading
    wordings (see ``config.SECTION_HEADINGS``) written as their own line,
    case-insensitively, and will not perfectly understand every resume
    layout (e.g. multi-column PDFs, headings embedded in tables, or
    unusual custom section names not present in the configured list).
    """
    sections: dict[str, list[str]] = {key: [] for key in config.SECTION_KEYS}

    current_section: str | None = None
    for line in text.split("\n"):
        heading = _match_heading(line)
        if heading is not None:
            current_section = heading
            continue
        if current_section is not None:
            sections[current_section].append(line)

    return {
        key: "\n".join(lines).strip()
        for key, lines in sections.items()
    }


# --------------------------------------------------------------------------
# Skill extraction
# --------------------------------------------------------------------------


def _compile_skill_patterns() -> list[tuple[str, list[re.Pattern[str]]]]:
    """Compile one case-insensitive, alphanumeric-boundary regex per
    skill alias.

    A custom boundary — "not preceded/followed by a letter or digit" — is
    used instead of ``\\b`` because ``\\b`` behaves inconsistently around
    punctuation-heavy aliases like "C++" or "C#" (the trailing symbols
    are non-word characters, so ``\\b`` would happily match "C+" inside
    "C+++"). The custom lookaround treats such symbols as part of the
    token, avoiding both partial-word and partial-symbol false matches.
    """
    compiled: list[tuple[str, list[re.Pattern[str]]]] = []
    for canonical, aliases in config.SKILL_VOCABULARY.items():
        patterns = [
            re.compile(
                r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            for alias in aliases
        ]
        compiled.append((canonical, patterns))
    return compiled


_SKILL_PATTERNS = _compile_skill_patterns()


def extract_skills(text: str) -> list[str]:
    """Extract canonical technical skills mentioned in `text`.

    Matching is case-insensitive with alphanumeric-boundary checks (see
    ``_compile_skill_patterns``), so "JavaScript" will not register a
    false "Java" match, and "sklearn" / "scikit learn" / "scikit-learn"
    all normalize to the single canonical name "scikit-learn".

    Returns skills in the vocabulary's configured order (deterministic),
    not in the order they appear in the text.
    """
    if not text:
        return []

    found: list[str] = []
    for canonical, patterns in _SKILL_PATTERNS:
        if any(pattern.search(text) for pattern in patterns):
            found.append(canonical)
    return found


def analyze_projects(projects_text: str) -> dict[str, object]:
    """Analyze the Projects section text independently of the Skills
    section, since job postings/resumes may mention technologies inside
    project descriptions that are never listed explicitly as "skills".

    Returns:
        {"projects": <raw text>, "project_derived_skills": [...]}
    """
    return {
        "projects": projects_text,
        "project_derived_skills": extract_skills(projects_text),
    }


def analyze_experience(experience_text: str) -> dict[str, object]:
    """Analyze the Experience section text independently to extract
    technical skills mentioned in job roles/descriptions.

    Returns:
        {"experience": <raw text>, "experience_derived_skills": [...]}
    """
    return {
        "experience": experience_text,
        "experience_derived_skills": extract_skills(experience_text),
    }


def combine_skills(
    explicit: list[str], project_derived: list[str], experience_derived: list[str]
) -> list[str]:
    """Merge explicit, project-derived, and experience-derived skills into
    one deduplicated, order-preserving list."""
    combined: list[str] = []
    seen: set[str] = set()
    for skill in [*explicit, *project_derived, *experience_derived]:
        if skill not in seen:
            seen.add(skill)
            combined.append(skill)
    return combined


# --------------------------------------------------------------------------
# Candidate profile assembly
# --------------------------------------------------------------------------


@dataclass
class CandidateProfile:
    """Structured representation of a parsed candidate resume."""

    skills: list[str] = field(default_factory=list)
    project_derived_skills: list[str] = field(default_factory=list)
    experience_derived_skills: list[str] = field(default_factory=list)
    combined_skills: list[str] = field(default_factory=list)
    education: str = ""
    experience: str = ""
    certifications: str = ""
    hobbies: str = ""
    projects: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "skills": self.skills,
            "project_derived_skills": self.project_derived_skills,
            "experience_derived_skills": self.experience_derived_skills,
            "combined_skills": self.combined_skills,
            "education": self.education,
            "experience": self.experience,
            "certifications": self.certifications,
            "hobbies": self.hobbies,
            "projects": self.projects,
        }


def build_candidate_profile(resume_text: str) -> CandidateProfile:
    """Run the full resume-analysis pipeline (sections -> skills ->
    combined technical profile) on cleaned resume text."""
    sections = extract_sections(resume_text)

    explicit_skills = extract_skills(sections["skills"])
    
    project_analysis = analyze_projects(sections["projects"])
    project_derived_skills = project_analysis["project_derived_skills"]
    
    experience_analysis = analyze_experience(sections["experience"])
    experience_derived_skills = experience_analysis["experience_derived_skills"]
    
    combined = combine_skills(explicit_skills, project_derived_skills, experience_derived_skills)

    return CandidateProfile(
        skills=explicit_skills,
        project_derived_skills=project_derived_skills,
        experience_derived_skills=experience_derived_skills,
        combined_skills=combined,
        education=sections["education"],
        experience=sections["experience"],
        certifications=sections["certifications"],
        hobbies=sections["hobbies"],
        projects=sections["projects"],
    )


def candidate_embedding_text(profile: CandidateProfile) -> str:
    """Build the short text used to generate the candidate's technical
    embedding, retaining rich semantic information from experience and projects.
    Deliberately excludes education and hobbies — see section 8 ("Candidate Representation")."""
    parts = []
    
    if profile.combined_skills:
        parts.append(f"Technical Skills:\n{', '.join(profile.combined_skills)}")
        
    if profile.experience.strip():
        parts.append(f"Experience:\n{profile.experience.strip()}")
        
    if profile.projects.strip():
        parts.append(f"Projects:\n{profile.projects.strip()}")
        
    if not parts:
        return "No technical skills or experience found."
        
    return "\n\n".join(parts)
