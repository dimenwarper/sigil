#!/usr/bin/env python3
"""
Correctness checker for symbolic regression optimization.
Verifies that f(x) = x**2 for a set of test cases.
"""

import sys
import os

# Add the current directory to Python path to import target_function
sys.path.insert(0, os.path.dirname(__file__))

from target_function import f


def test_quadratic_behavior():
    """Test that the function behaves like f(x) = x**2 and return accuracy score"""
    test_cases = [0, 1, 2, 3, -1, -2, 0.5, 1.5]
    
    total_tests = len(test_cases)
    passed_tests = 0
    tolerance = 1e-10
    
    for x in test_cases:
        result = f(x)
        expected = x**2
        
        if abs(result - expected) <= tolerance:
            print(f"PASS: f({x}) = {result}")
            passed_tests += 1
        else:
            print(f"FAIL: f({x}) = {result}, expected {expected}")
    
    # Calculate accuracy as percentage of passed tests
    accuracy = (passed_tests / total_tests) * 100.0
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"SCORE: {accuracy}")
    

if __name__ == "__main__":
    test_quadratic_behavior()
