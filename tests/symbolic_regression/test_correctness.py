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
    """Test that the function behaves like f(x) = x**2"""
    test_cases = [0, 1, 2, 3, -1, -2, 0.5, 1.5]
    tolerance = 1e-10
    
    all_passed = True
    for x in test_cases:
        result = f(x)
        expected = x**2
        
        if abs(result - expected) > tolerance:
            print(f"FAIL: f({x}) = {result}, expected {expected}")
            all_passed = False
        else:
            print(f"PASS: f({x}) = {result}")
    
    if all_passed:
        print("All tests passed! Function is correctly f(x) = x**2")
        sys.exit(0)
    else:
        print("Some tests failed! Function is not yet f(x) = x**2")
        sys.exit(1)


if __name__ == "__main__":
    test_quadratic_behavior()
