# Symbolic Regression Test Case

This directory contains a simple test case for the Sigil framework to optimize a function from `f(x) = x` to `f(x) = x**2`.

## Files

- `target_function.py`: Contains the target function `f(x)` that needs to be optimized. Currently returns `x`, should be optimized to return `x**2`.
- `test_correctness.py`: Correctness checker that verifies the function behaves like `f(x) = x**2`.
- `.sigil/symbolic_regression.sigil.yaml`: Spec file defining the optimization task with pin pointing to the target function.
- `.sigil/quadratic_correctness.eval.yaml`: Evaluation file defining how to test correctness of the transformation.

## Usage

The SIGIL region in `target_function.py` is marked with:
```python
# SIGIL:BEGIN target_function
return x
# SIGIL:END target_function
```

The goal is to optimize this to:
```python
# SIGIL:BEGIN target_function
return x**2
# SIGIL:END target_function
```

## Testing

Run the correctness test:
```bash
python3 test_correctness.py
```

This should fail initially (exit code 1) since `f(x) = x` != `f(x) = x**2`, and pass (exit code 0) after optimization.
