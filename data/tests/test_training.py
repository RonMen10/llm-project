#!/usr/bin/env python3
"""
Unit tests for training script
"""
import pytest
import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

def test_train_script_exists():
    """Test that training script exists"""
    assert os.path.exists("scripts/train.py"), "train.py should exist"

def test_train_has_main_function():
    """Test that training script has main function"""
    with open("scripts/train.py", "r") as f:
        content = f.read()
    assert "def main()" in content or "if __name__" in content, \
        "train.py should have main function or entry point"

def test_train_imports():
    """Test that training script imports necessary modules"""
    with open("scripts/train.py", "r") as f:
        content = f.read()
    
    # Check for common imports
    assert "import torch" in content or "from torch" in content, \
        "Should import torch"
    assert "transformers" in content, "Should import transformers"
    assert "datasets" in content, "Should import datasets"

def test_train_arguments():
    """Test that training script accepts arguments"""
    with open("scripts/train.py", "r") as f:
        content = f.read()
    
    # Look for argument parsing
    has_argparse = "argparse" in content or "ArgumentParser" in content
    has_sys_argv = "sys.argv" in content
    
    assert has_argparse or has_sys_argv, \
        "Should accept command line arguments"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])