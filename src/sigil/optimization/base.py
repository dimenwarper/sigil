"""
Base optimizer class for Sigil.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from ..core.models import Candidate, Pin, EvaluationResult
from ..workspace.workspace import Workspace


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""
    max_iterations: int = 10
    max_candidates: int = 50
    timeout_seconds: int = 300
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    system_prompt: str = ""
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class BaseOptimizer(ABC):
    """
    Base class for code optimization algorithms.
    
    Optimizers generate candidate implementations and guide the search
    for improved versions of functions.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.name = self.__class__.__name__.lower().replace("optimizer", "")
    
    @abstractmethod
    def optimize(
        self,
        pin: Pin,
        workspace: Workspace,
        evaluator: Optional[Callable] = None
    ) -> List[Candidate]:
        """
        Run optimization to generate improved candidates.
        
        Args:
            pin: The function pin to optimize
            workspace: Workspace to store candidates
            evaluator: Optional evaluation function
            
        Returns:
            List of generated candidates
        """
        pass
    
    def _get_system_prompt(self, pin: Pin) -> str:
        """Get the system prompt for LLM generation."""
        base_prompt = """You are a code optimization expert. Your task is to improve the given Python function while maintaining its core functionality.

Goals:
- Improve performance, readability, or correctness
- Maintain the same function signature and behavior
- Write clean, well-structured code
- Consider edge cases and error handling

Function to improve:
```python
{original_code}
```

Please provide an improved version of this function."""

        return base_prompt.format(original_code=pin.original_source)
    
    def _call_llm(self, prompt: str, system_prompt: str = "") -> str:
        """
        Call the LLM to generate code.
        
        This is a placeholder - in a real implementation, you'd integrate
        with OpenAI, Anthropic, or other LLM providers.
        """
        # Placeholder implementation
        import random
        
        # Simple mock responses for demonstration
        mock_improvements = [
            "# Improved with better error handling\ndef improved_function():\n    try:\n        return original_logic()\n    except Exception as e:\n        logging.error(f'Error: {e}')\n        return None",
            "# Optimized version\ndef improved_function():\n    # More efficient implementation\n    return optimized_logic()",
            "# Enhanced with type hints\ndef improved_function(x: int) -> int:\n    # Better typed version\n    return enhanced_logic(x)"
        ]
        
        return random.choice(mock_improvements)
    
    def _extract_function_from_response(self, response: str, function_name: str) -> str:
        """Extract the function code from LLM response."""
        lines = response.split('\n')
        function_lines = []
        in_function = False
        
        for line in lines:
            if f'def {function_name}(' in line:
                in_function = True
            
            if in_function:
                function_lines.append(line)
                
                # Stop at the next function definition or end of indentation
                if line.strip() and not line.startswith(' ') and not line.startswith('\t') and not line.startswith('def'):
                    if function_lines and len(function_lines) > 1:
                        function_lines = function_lines[:-1]  # Remove the last line
                        break
        
        return '\n'.join(function_lines)
    
    def _validate_candidate(self, original_source: str, candidate_source: str) -> bool:
        """Basic validation of candidate code."""
        try:
            import ast
            # Try to parse the candidate
            ast.parse(candidate_source)
            return True
        except SyntaxError:
            return False