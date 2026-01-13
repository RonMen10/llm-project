#!/usr/bin/env python3
"""
Simple test functions for your LLM project
"""
import sys
import os
import json

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig
        import datasets
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_data_structure():
    """Test that required files exist"""
    print("\nTesting project structure...")
    
    required_files = [
        "requirements.txt",
        "scripts/train.py",
        "scripts/inference.py",
        "data/training_data.json",
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"{file} exists")
        else:
            print(f"{file} missing")
            all_exist = False
    
    return all_exist

def test_training_script_syntax():
    """Test that training script has valid syntax"""
    print("\nTesting training script syntax...")
    
    try:
        with open("scripts/train.py", "r") as f:
            code = f.read()
        compile(code, "train.py", "exec")
        print("Training script has valid syntax")
        return True
    except SyntaxError as e:
        print(f"Syntax error in train.py: {e}")
        return False

def test_inference_script_syntax():
    """Test that inference script has valid syntax"""
    print("\nTesting inference script syntax...")
    
    try:
        with open("scripts/inference.py", "r") as f:
            code = f.read()
        compile(code, "inference.py", "exec")
        print("Inference script has valid syntax")
        return True
    except SyntaxError as e:
        print(f"Syntax error in inference.py: {e}")
        return False

def test_dependencies():
    """Test that requirements.txt has basic dependencies"""
    print("\nTesting dependencies...")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        required_packages = [
            "torch",
            "transformers",
            "bitsandbytes",
            "peft",
            'datasets',
        ]
        
        missing = []
        for pkg in required_packages:
            if pkg not in content.lower():
                missing.append(pkg)
        
        if missing:
            print(f"Missing packages in requirements.txt: {missing}")
            return False
        else:
            print("Basic dependencies found in requirements.txt")
            return True
    except FileNotFoundError:
        print("requirements.txt not found")
        return False

def run_all_tests():
    """Run all test functions"""
    print("=" * 60)
    print("Running Test Functions")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_data_structure),
        ("Training Script Syntax", test_training_script_syntax),
        ("Inference Script Syntax", test_inference_script_syntax),
        ("Dependencies", test_dependencies),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Error running {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{status} - {test_name}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())