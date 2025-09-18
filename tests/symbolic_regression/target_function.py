"""
Target function for symbolic regression optimization.
The goal is to optimize f(x) = x to become f(x) = x**2.
"""

def f(x):
    """Function to be optimized from f(x) = x to f(x) = x**2"""
    return x


def test_function():
    """Test function to verify the behavior of f(x)"""
    test_cases = [0, 1, 2, 3, -1, -2, 0.5, 1.5]
    
    print("Testing function f(x):")
    for x in test_cases:
        result = f(x)
        expected_linear = x
        expected_quadratic = x**2
        print(f"f({x}) = {result}, linear: {expected_linear}, quadratic: {expected_quadratic}")
        
    return True


if __name__ == "__main__":
    test_function()
