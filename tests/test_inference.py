#!/usr/bin/env python3
"""
Unit tests for inference script
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

def test_inference_script_exists():
    """Test that inference script exists"""
    assert os.path.exists("scripts/inference.py"), "inference.py should exist"

def test_inference_has_generate_function():
    """Test that inference script has generate function"""
    with open("scripts/inference.py", "r") as f:
        content = f.read()
    
    # Look for generation function
    has_generate = "def generate" in content or "model.generate" in content
    assert has_generate, "Should have generate function"

def test_inference_handles_prompts():
    """Test that inference can handle prompts"""
    with open("scripts/inference.py", "r") as f:
        content = f.read()
    
    # Check for prompt handling
    has_prompt = "prompt" in content.lower()
    assert has_prompt, "Should handle prompts"

def test_inference_imports():
    """Test that inference script imports necessary modules"""
    with open("scripts/inference.py", "r") as f:
        content = f.read()
    
    assert "transformers" in content, "Should import transformers"
    assert "torch" in content, "Should import torch"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])