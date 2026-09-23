import pytest
from src.analyzer import analyze_projects, analyze_experience

def test_project_extraction_multiple_skills():
    text = "Built a distributed log analysis system using Python, Redis, MongoDB, and microservices."
    result = analyze_projects(text)
    skills = result["project_derived_skills"]
    assert "Python" in skills
    assert "Redis" in skills
    assert "MongoDB" in skills
    assert len(skills) >= 3

def test_experience_extraction_skills():
    text = "Worked as a backend engineer building REST APIs with Node.js and PostgreSQL."
    result = analyze_experience(text)
    skills = result["experience_derived_skills"]
    assert "REST API" in skills
    assert "Node.js" in skills
    assert "PostgreSQL" in skills
    assert len(skills) >= 3
