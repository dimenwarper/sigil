"""
Simple optimization algorithms for Sigil.
"""

import random
import time
from typing import List, Optional, Callable
from datetime import datetime

from .base import BaseOptimizer, OptimizationConfig
from ..core.models import Candidate, Pin
from ..core.ids import CandidateID
from ..workspace.workspace import Workspace


class SimpleOptimizer(BaseOptimizer):
    """
    Simple optimizer that uses direct LLM prompting.
    
    Generates a small number of candidates by prompting the LLM
    with different optimization objectives.
    """
    
    def optimize(
        self,
        pin: Pin,
        workspace: Workspace,
        evaluator: Optional[Callable] = None
    ) -> List[Candidate]:
        """Generate candidates using simple LLM prompting."""
        candidates = []
        
        optimization_prompts = [
            "Improve this function for better performance:",
            "Refactor this function for better readability:",
            "Add better error handling to this function:",
            "Optimize this function for memory efficiency:",
            "Make this function more robust and maintainable:"
        ]
        
        for i, prompt_prefix in enumerate(optimization_prompts[:self.config.max_iterations]):
            if len(candidates) >= self.config.max_candidates:
                break
            
            try:
                # Construct the prompt
                system_prompt = self._get_system_prompt(pin)
                user_prompt = f"{prompt_prefix}\n\n```python\n{pin.original_source}\n```"
                
                # Call LLM
                response = self._call_llm(user_prompt, system_prompt)
                
                # Extract function from response
                function_name = pin.function_id.qualname.split('.')[-1]
                candidate_source = response  # For now, just use the full response
                
                if not candidate_source or not self._validate_candidate(pin.original_source, candidate_source):
                    continue
                
                # Create candidate
                candidate_id = CandidateID.from_source(pin.original_source, candidate_source)
                
                candidate = Candidate(
                    candidate_id=candidate_id,
                    function_id=pin.function_id,
                    source_code=candidate_source,
                    diff_text="",  # Would compute actual diff
                    generator=self.name,
                    metadata={
                        "iteration": i,
                        "prompt": prompt_prefix,
                        "optimization_objective": prompt_prefix.split()[1] if len(prompt_prefix.split()) > 1 else "general"
                    }
                )
                
                # Store in workspace
                workspace.store_candidate(candidate)
                candidates.append(candidate)
                
                # Brief delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error generating candidate {i}: {e}")
                continue
        
        return candidates


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random search optimizer with variation prompts.
    
    Uses random variations in prompts and temperature to explore
    the space of possible improvements.
    """
    
    def optimize(
        self,
        pin: Pin,
        workspace: Workspace,
        evaluator: Optional[Callable] = None
    ) -> List[Candidate]:
        """Generate candidates using randomized prompting."""
        candidates = []
        
        objectives = [
            "performance", "readability", "maintainability", "robustness", 
            "efficiency", "clarity", "simplicity", "extensibility"
        ]
        
        techniques = [
            "refactoring", "optimization", "restructuring", "enhancement",
            "improvement", "modernization", "streamlining"
        ]
        
        for i in range(self.config.max_iterations):
            if len(candidates) >= self.config.max_candidates:
                break
            
            try:
                # Random prompt generation
                objective = random.choice(objectives)
                technique = random.choice(techniques)
                
                prompt_templates = [
                    f"Improve this function's {objective} through {technique}:",
                    f"Apply {technique} to enhance {objective}:",
                    f"Focus on {objective} while {technique} this function:",
                    f"Use {technique} techniques to optimize for {objective}:"
                ]
                
                prompt_prefix = random.choice(prompt_templates)
                
                # Random temperature for variation
                temp = random.uniform(0.3, 1.0)
                
                # Construct the prompt
                system_prompt = self._get_system_prompt(pin)
                user_prompt = f"{prompt_prefix}\n\n```python\n{pin.original_source}\n```"
                
                # Call LLM (would pass temperature in real implementation)
                response = self._call_llm(user_prompt, system_prompt)
                
                # Extract function from response
                function_name = pin.function_id.qualname.split('.')[-1]
                candidate_source = response  # For now, just use the full response
                
                if not candidate_source or not self._validate_candidate(pin.original_source, candidate_source):
                    continue
                
                # Create candidate
                candidate_id = CandidateID.from_source(pin.original_source, candidate_source)
                
                # Check if we've already generated this candidate
                if any(c.candidate_id == candidate_id for c in candidates):
                    continue
                
                candidate = Candidate(
                    candidate_id=candidate_id,
                    function_id=pin.function_id,
                    source_code=candidate_source,
                    diff_text="",  # Would compute actual diff
                    generator=self.name,
                    metadata={
                        "iteration": i,
                        "objective": objective,
                        "technique": technique,
                        "temperature": temp,
                        "prompt": prompt_prefix
                    }
                )
                
                # Store in workspace
                workspace.store_candidate(candidate)
                candidates.append(candidate)
                
                # Brief delay
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error generating candidate {i}: {e}")
                continue
        
        return candidates


class GreedyOptimizer(BaseOptimizer):
    """
    Simple greedy optimizer.
    
    Generates candidates and keeps the best performing ones as parents
    for the next generation.
    """
    
    def optimize(
        self,
        pin: Pin,
        workspace: Workspace,
        evaluator: Optional[Callable] = None
    ) -> List[Candidate]:
        """Generate candidates using greedy improvement."""
        candidates = []
        current_best = None
        current_best_source = pin.original_source
        
        for iteration in range(self.config.max_iterations):
            if len(candidates) >= self.config.max_candidates:
                break
            
            try:
                # Prompt based on current best
                if current_best:
                    prompt_prefix = "Further improve this already optimized function:"
                    source_to_improve = current_best_source
                else:
                    prompt_prefix = "Optimize this function:"
                    source_to_improve = pin.original_source
                
                system_prompt = self._get_system_prompt(pin)
                user_prompt = f"{prompt_prefix}\n\n```python\n{source_to_improve}\n```"
                
                # Generate a few candidates for this iteration
                iteration_candidates = []
                for _ in range(3):  # Generate 3 candidates per iteration
                    response = self._call_llm(user_prompt, system_prompt)
                    
                    function_name = pin.function_id.qualname.split('.')[-1]
                    candidate_source = self._extract_function_from_response(response, function_name)
                    
                    if not candidate_source or not self._validate_candidate(pin.original_source, candidate_source):
                        continue
                    
                    candidate_id = CandidateID.from_source(pin.original_source, candidate_source)
                    
                    # Skip duplicates
                    if any(c.candidate_id == candidate_id for c in candidates):
                        continue
                    
                    candidate = Candidate(
                        candidate_id=candidate_id,
                        function_id=pin.function_id,
                        source_code=candidate_source,
                        diff_text="",
                        generator=self.name,
                        parent_candidate=current_best.candidate_id if current_best else None,
                        metadata={
                            "iteration": iteration,
                            "parent_iteration": iteration - 1 if current_best else -1
                        }
                    )
                    
                    workspace.store_candidate(candidate)
                    iteration_candidates.append(candidate)
                
                # If we have an evaluator, pick the best candidate from this iteration
                if evaluator and iteration_candidates:
                    # This is a placeholder - would actually evaluate candidates
                    best_candidate = random.choice(iteration_candidates)
                    current_best = best_candidate
                    current_best_source = best_candidate.source_code
                elif iteration_candidates:
                    # Without evaluator, just pick the first valid candidate
                    current_best = iteration_candidates[0]
                    current_best_source = current_best.source_code
                
                candidates.extend(iteration_candidates)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                continue
        
        return candidates