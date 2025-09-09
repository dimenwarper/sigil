"""
Basic example of using Sigil for code optimization.

This demonstrates the core workflow from the CONTEXT.md specification.
"""

from sigil import Spec, improve
import sigil.spec.spec as sigil


def myeval(result):
    """
    Evaluation function that scores the result.
    
    In this example, we want to maximize the result value.
    """
    if isinstance(result, (int, float)):
        return result
    return 0


# Create and track a spec
spec = Spec(
    name="myspec",
    description="Example optimization spec"
)
sigil.track(spec)

# Set as default spec for @improve decorator
from sigil.core.config import get_config
config = get_config()
config.default_spec = "myspec"

# Add the evaluation function to the spec
spec.add_evaluator("myeval", myeval)


@improve(with_eval=myeval, serve_workspace="v1")
def myfun(x):
    """
    Example function to optimize.
    
    This simple function just returns x * 2, but Sigil will try to
    improve it through LLM-guided optimization.
    """
    return x * 2


if __name__ == "__main__":
    # Use the function to generate samples
    print("Running function to collect samples...")
    
    test_values = [1, 2, 3, 4, 5]
    
    for v in test_values:
        result = myfun(v)
        print(f"myfun({v}) = {result}")
    
    print("\nTo run optimization:")
    print("1. Start tracking: sigil tracker start")
    print("2. Run this script to collect samples")
    print("3. Stop tracking: sigil tracker stop") 
    print("4. Inspect samples: sigil inspect-samples myspec")
    print("5. Run optimization: sigil run myspec --name v1 --optimizer simple --niter 5")
    print("6. Inspect solutions: sigil inspect-solutions myspec")