#!/usr/bin/env python3
"""
Data validation tests
"""
import pytest
import json
import os

def test_requirements_txt_exists():
    """Test that requirements.txt exists"""
    assert os.path.exists("requirements.txt"), "requirements.txt should exist"

def test_requirements_format():
    """Test that requirements.txt has proper format"""
    with open("requirements.txt", "r") as f:
        lines = f.readlines()
    
    # Check it's not empty
    assert len(lines) > 0, "requirements.txt should not be empty"
    
    # Check for common required packages
    content = "".join(lines).lower()
    assert "torch" in content, "Should specify torch"
    assert "transformers" in content, "Should specify transformers"

def test_gitignore_exists():
    """Test that .gitignore exists"""
    assert os.path.exists(".gitignore"), ".gitignore should exist"

def test_gitignore_has_common_patterns():
    """Test that .gitignore has common exclusions"""
    with open(".gitignore", "r") as f:
        content = f.read()
    
    common_patterns = [
        "__pycache__",
        ".pyc",
        ".env",
        "venv",
        ".vscode",
    ]
    
    # Check for at least some common patterns
    found_patterns = [p for p in common_patterns if p in content]
    assert len(found_patterns) > 0, \
        ".gitignore should have common exclusion patterns"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])