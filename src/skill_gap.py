"""Skill-gap analysis for the top-K matched jobs.

Per the project spec, detailed skill-gap analysis is performed *only* for
the top-K jobs returned by the matcher — never against the entire job
dataset. All results here are descriptive: a missing skill is not a
guarantee of rejection, and a similarity score is not a hiring
probability (see ``matcher.py`` / the CLI output).
"""

from __future__ import annotations


import re

def normalize_skill(skill: str) -> str:
    """Normalize common skill variations for more robust matching."""
    s = skill.strip().lower()
    
    # Common standardizations
    replacements = {
        "nodejs": "node.js",
        "rest": "rest api",
        "reactjs": "react",
        "vuejs": "vue.js",
    }
    
    return replacements.get(s, s)


def compute_skill_gap(candidate_skills: list[str], job_required_skills: list[str]) -> dict[str, list[str]]:
    """Compare a candidate's combined technical skills against one job's
    required/extracted skills.

    Returns:
        {"matched_skills": [...], "missing_skills": [...]}
        Both lists preserve the order of `job_required_skills` and contain
        no duplicates.
    """
    candidate_set = set(normalize_skill(skill) for skill in candidate_skills)

    matched: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw_skill in job_required_skills:
        skill = normalize_skill(raw_skill)
        if skill in seen:
            continue
        seen.add(skill)
        
        # Use the original raw_skill for display to match job dataset
        if skill in candidate_set:
            matched.append(raw_skill)
        else:
            missing.append(raw_skill)

    return {"matched_skills": matched, "missing_skills": missing}


def compute_missing_frequency(missing_skill_lists: list[list[str]]) -> dict[str, dict[str, int | str]]:
    """Aggregate missing-skill frequency across multiple jobs' skill-gap
    results (typically the top-K jobs).

    Args:
        missing_skill_lists: one list of missing skills per job (e.g.
            [gap["missing_skills"] for gap in top_k_gaps]).

    Returns:
        A dict mapping skill -> {"count": int, "out_of": int, "fraction": "n/N"},
        sorted by descending count. `out_of` is the number of jobs the
        frequency was computed over (i.e. len(missing_skill_lists)).
    """
    total_jobs = len(missing_skill_lists)
    counts: dict[str, int] = {}
    for missing_skills in missing_skill_lists:
        for skill in set(missing_skills):  # count each job once per skill
            counts[skill] = counts.get(skill, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    return {
        skill: {"count": count, "out_of": total_jobs, "fraction": f"{count}/{total_jobs}"}
        for skill, count in ranked
    }
