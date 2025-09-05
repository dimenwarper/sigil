#!/usr/bin/env python3
"""Example usage of Sigil framework."""

from sigil import improve, Spec, track


# Define a spec for our example
spec = Spec(
    name="example_spec",
    description="Example specification for testing Sigil framework"
)

# Register the spec for tracking
track(spec)


def simple_eval(result):
    """Simple evaluation function that scores based on result value."""
    if isinstance(result, (int, float)):
        return float(result)
    return 0.0


@improve(with_eval=simple_eval, serve_workspace="v1", spec=spec)
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b


@improve(with_eval=simple_eval, serve_workspace="v1", spec=spec)
def multiply_numbers(x, y):
    """Multiply two numbers."""
    return x * y


def main():
    """Run example usage."""
    print("=== Sigil Framework Example ===")
    print()
    
    print("Testing calculate_sum function:")
    result1 = calculate_sum(5, 3)
    print(f"calculate_sum(5, 3) = {result1}")
    
    result2 = calculate_sum(10, 20)
    print(f"calculate_sum(10, 20) = {result2}")
    
    print()
    print("Testing multiply_numbers function:")
    result3 = multiply_numbers(4, 6)
    print(f"multiply_numbers(4, 6) = {result3}")
    
    result4 = multiply_numbers(7, 8)
    print(f"multiply_numbers(7, 8) = {result4}")
    
    print()
    print("Example complete!")
    print()
    print("Next steps:")
    print("1. Run 'sigil inspect-samples example_spec' to see collected samples")
    print("2. Run 'sigil run example_spec --optimizer llm_sampler --niter 10' to optimize")
    print("3. Run 'sigil inspect-solutions example_spec' to see optimized solutions")


if __name__ == "__main__":
    main()