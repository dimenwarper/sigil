from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .backend import Backend, EvalResult, get_backend
from .evals import EvalDef, load_eval
from .llm import LLMProvider, PatchRequest, PatchResponse, OpenAICompatibleProvider, StubProvider
from .patches import Candidate
from .spec import Spec, spec_uri, pin_uri


def build_patch_request(spec: Spec) -> PatchRequest:
    files: Dict[str, str] = {}
    pins: Dict[str, str] = {}
    for pin in spec.pins:
        pins[pin_uri(spec, pin)] = pin.id
        for f in (pin.files or []):
            path = spec.repo_root / f
            if path.exists():
                files[f] = path.read_text()
    return PatchRequest(
        spec_uri=spec_uri(spec),
        objective="Improve performance while preserving correctness; return a unified diff strictly within pin regions.",
        files=files,
        pins=pins,
        constraints={"edit_format": "unified_diff"},
    )


def get_provider(kind: str) -> LLMProvider:
    if kind == "openai":
        return OpenAICompatibleProvider()
    elif kind == "stub":
        return StubProvider()
    else:
        raise ValueError(f"Unknown provider: {kind}")


def get_optimizer(kind: str, **kwargs) -> BaseOptimizer:
    """Get an optimizer instance by kind."""
    if kind == "simple":
        return SimpleOptimizer()
    elif kind == "alphaevolve":
        return AlphaEvolveOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {kind}")


class BaseOptimizer:
    def propose(self, spec: Spec, provider: LLMProvider, num: int = 1) -> List[PatchResponse]:
        raise NotImplementedError


@dataclass
class SimpleOptimizer(BaseOptimizer):
    def propose(self, spec: Spec, provider: LLMProvider, num: int = 1) -> List[PatchResponse]:
        req = build_patch_request(spec)
        responses: List[PatchResponse] = []
        n = max(1, int(num))
        for _ in range(n):
            responses.append(provider.propose(req))
        return responses


@dataclass
class Individual:
    """An individual in the evolutionary algorithm."""
    candidate: Candidate
    fitness: Optional[float] = None
    eval_result: Optional[EvalResult] = None
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)


@dataclass 
class Island:
    """A population island for island-based evolution."""
    id: str
    population: List[Individual] = field(default_factory=list)
    generation: int = 0
    best_fitness: Optional[float] = None
    best_individual: Optional[Individual] = None
    
    def add_individual(self, individual: Individual) -> None:
        """Add an individual to the island population."""
        self.population.append(individual)
        if individual.fitness is not None:
            if self.best_fitness is None or individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = individual
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """Get the n best individuals from the island."""
        valid_individuals = [ind for ind in self.population if ind.fitness is not None]
        return sorted(valid_individuals, key=lambda x: x.fitness, reverse=True)[:n]
    
    def replace_worst(self, new_individuals: List[Individual]) -> None:
        """Replace the worst individuals with new ones."""
        valid_individuals = [ind for ind in self.population if ind.fitness is not None]
        if not valid_individuals:
            self.population.extend(new_individuals)
            return
            
        sorted_pop = sorted(valid_individuals, key=lambda x: x.fitness, reverse=True)
        # Keep the best, replace the worst
        num_to_replace = min(len(new_individuals), len(sorted_pop))
        self.population = sorted_pop[:-num_to_replace] + new_individuals


@dataclass
class AlphaEvolveOptimizer(BaseOptimizer):
    """
    AlphaEvolve optimizer implementing island-based evolutionary algorithm.
    
    This optimizer maintains multiple islands (populations) that evolve independently
    with periodic migration between islands for cross-pollination.
    """
    num_islands: int = 4
    population_size: int = 20
    num_generations: int = 10
    migration_interval: int = 3
    migration_rate: float = 0.1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    backend_type: str = "local"
    eval_name: Optional[str] = None
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        if self.random_seed is not None:
            random.seed(self.random_seed)
    
    def propose(self, spec: Spec, provider: LLMProvider, num: int = 1) -> List[PatchResponse]:
        """
        Run the island-based evolutionary algorithm to propose optimized patches.
        """
        # Initialize backend and evaluation
        backend = get_backend(self.backend_type)
        eval_def = self._get_eval_def(spec)
        
        # Initialize islands with random populations
        islands = self._initialize_islands(spec, provider)
        
        # Evaluate initial populations
        self._evaluate_populations(islands, backend, eval_def, spec.repo_root)
        
        # Run evolutionary algorithm
        for generation in range(self.num_generations):
            # Evolve each island independently
            for island in islands:
                self._evolve_island(island, spec, provider, backend, eval_def, generation)
            
            # Migrate between islands periodically
            if (generation + 1) % self.migration_interval == 0:
                self._migrate_between_islands(islands)
        
        # Collect best individuals from all islands
        best_individuals = []
        for island in islands:
            best_individuals.extend(island.get_best(max(1, num // self.num_islands)))
        
        # Sort by fitness and return top candidates
        best_individuals.sort(key=lambda x: x.fitness or 0, reverse=True)
        
        # Convert back to PatchResponse format
        responses = []
        for individual in best_individuals[:num]:
            responses.append(PatchResponse(
                patch_text=individual.candidate.patch_text,
                reasoning=f"Evolved candidate (fitness: {individual.fitness:.4f})" if individual.fitness else "Evolved candidate"
            ))
        
        return responses
    
    def _get_eval_def(self, spec: Spec) -> EvalDef:
        """Get the evaluation definition for fitness assessment."""
        if self.eval_name:
            return load_eval(spec.repo_root, self.eval_name)
        elif spec.evals:
            return load_eval(spec.repo_root, spec.evals[0])
        else:
            raise ValueError("No evaluation specified for AlphaEvolve optimizer")
    
    def _initialize_islands(self, spec: Spec, provider: LLMProvider) -> List[Island]:
        """Initialize islands with random populations."""
        islands = []
        req = build_patch_request(spec)
        
        for i in range(self.num_islands):
            island = Island(id=f"island_{i}")
            
            # Generate initial population for this island
            for j in range(self.population_size):
                response = provider.propose(req)
                candidate = Candidate(
                    id=f"gen0_island{i}_ind{j}",
                    patch_text=response.patch_text
                )
                individual = Individual(
                    candidate=candidate,
                    generation=0
                )
                island.add_individual(individual)
            
            islands.append(island)
        
        return islands
    
    def _evaluate_populations(self, islands: List[Island], backend: Backend, 
                            eval_def: EvalDef, repo_root: Path) -> None:
        """Evaluate fitness for all individuals in all islands."""
        # Collect all candidates that need evaluation
        candidates_to_eval = []
        individual_map = {}
        
        for island in islands:
            for individual in island.population:
                if individual.fitness is None:
                    candidates_to_eval.append(individual.candidate)
                    individual_map[individual.candidate.id] = individual
        
        if not candidates_to_eval:
            return
        
        # Batch evaluate all candidates
        eval_results = backend.evaluate(eval_def, repo_root, candidates_to_eval)
        
        # Assign fitness scores
        for result in eval_results:
            if result.id in individual_map:
                individual = individual_map[result.id]
                individual.eval_result = result
                individual.fitness = self._compute_fitness(result, eval_def)
                
                # Update island best
                island = next(isl for isl in islands if individual in isl.population)
                if individual.fitness is not None:
                    if island.best_fitness is None or individual.fitness > island.best_fitness:
                        island.best_fitness = individual.fitness
                        island.best_individual = individual
    
    def _compute_fitness(self, eval_result: EvalResult, eval_def: EvalDef) -> Optional[float]:
        """Compute fitness score from evaluation result."""
        if eval_result.error:
            return 0.0
        
        # Simple fitness function: prioritize correctness, then optimize for performance
        metrics = eval_result.metrics
        
        # Check for correctness metric
        correctness = metrics.get("correctness", False)
        if not correctness:
            return 0.0  # Invalid if not correct
        
        # Base fitness for correct solutions
        fitness = 1.0
        
        # Add performance bonuses
        if "latency_ms" in metrics and metrics["latency_ms"] is not None:
            # Lower latency is better, so invert and normalize
            latency = float(metrics["latency_ms"])
            if latency > 0:
                fitness += 1.0 / (1.0 + latency / 1000.0)  # Normalize around 1 second
        
        # Add other numeric metrics as bonuses
        for key, value in metrics.items():
            if key != "correctness" and key != "latency_ms" and isinstance(value, (int, float)):
                fitness += float(value) * 0.1  # Small bonus for other metrics
        
        return fitness
    
    def _evolve_island(self, island: Island, spec: Spec, provider: LLMProvider,
                      backend: Backend, eval_def: EvalDef, generation: int) -> None:
        """Evolve a single island for one generation."""
        if not island.population:
            return
        
        new_individuals = []
        req = build_patch_request(spec)
        
        # Generate new individuals through evolution operations
        num_offspring = max(1, int(self.population_size * 0.5))  # Replace half the population
        
        for i in range(num_offspring):
            if random.random() < self.crossover_rate and len(island.population) >= 2:
                # Crossover
                parent1, parent2 = self._tournament_selection(island, 2)
                child = self._crossover(parent1, parent2, spec, provider, generation + 1)
            else:
                # Mutation or new random individual
                if random.random() < self.mutation_rate and island.population:
                    parent = self._tournament_selection(island, 1)[0]
                    child = self._mutate(parent, spec, provider, generation + 1)
                else:
                    # Generate completely new individual
                    response = provider.propose(req)
                    candidate = Candidate(
                        id=f"gen{generation+1}_island{island.id}_new{i}",
                        patch_text=response.patch_text
                    )
                    child = Individual(
                        candidate=candidate,
                        generation=generation + 1
                    )
            
            new_individuals.append(child)
        
        # Evaluate new individuals
        candidates_to_eval = [ind.candidate for ind in new_individuals]
        eval_results = backend.evaluate(eval_def, spec.repo_root, candidates_to_eval)
        
        # Assign fitness to new individuals
        for result in eval_results:
            for individual in new_individuals:
                if individual.candidate.id == result.id:
                    individual.eval_result = result
                    individual.fitness = self._compute_fitness(result, eval_def)
                    break
        
        # Replace worst individuals with new ones
        island.replace_worst(new_individuals)
        island.generation = generation + 1
    
    def _tournament_selection(self, island: Island, num_parents: int) -> List[Individual]:
        """Select parents using tournament selection."""
        parents = []
        valid_individuals = [ind for ind in island.population if ind.fitness is not None]
        
        if not valid_individuals:
            return []
        
        for _ in range(num_parents):
            tournament = random.sample(valid_individuals, min(self.tournament_size, len(valid_individuals)))
            winner = max(tournament, key=lambda x: x.fitness or 0)
            parents.append(winner)
        
        return parents
    
    def _crossover(self, parent1: Individual, parent2: Individual, spec: Spec,
                  provider: LLMProvider, generation: int) -> Individual:
        """Create offspring through crossover of two parents."""
        # For patch crossover, we'll use the LLM to combine insights from both patches
        req = build_patch_request(spec)
        
        # TODO: This needs to include a richer prompt: include  parent info and eval feedback
        crossover_objective = (
            f"Combine the best aspects of these two optimization approaches:\n"
            f"Approach 1:\n{parent1.candidate.patch_text}\n\n"
            f"Approach 2:\n{parent2.candidate.patch_text}\n\n"
            f"Create an improved unified diff that incorporates the strengths of both."
        )
        
        req.objective = crossover_objective
        response = provider.propose(req)
        
        candidate = Candidate(
            id=f"gen{generation}_crossover_{parent1.candidate.id}_{parent2.candidate.id}",
            patch_text=response.patch_text
        )
        
        return Individual(
            candidate=candidate,
            generation=generation,
            parent_ids=[parent1.candidate.id, parent2.candidate.id]
        )
    
    def _mutate(self, parent: Individual, spec: Spec, provider: LLMProvider, generation: int) -> Individual:
        """Create offspring through mutation of a parent."""
        req = build_patch_request(spec)
        
        # TODO: This needs to include a richer prompt: include  parent info and eval feedback
        mutation_objective = (
            f"Improve upon this optimization approach with a small variation:\n"
            f"{parent.candidate.patch_text}\n\n"
            f"Make a targeted improvement while preserving the core optimization strategy."
        )
        
        req.objective = mutation_objective
        response = provider.propose(req)
        
        candidate = Candidate(
            id=f"gen{generation}_mutant_{parent.candidate.id}",
            patch_text=response.patch_text
        )
        
        return Individual(
            candidate=candidate,
            generation=generation,
            parent_ids=[parent.candidate.id]
        )
    
    def _migrate_between_islands(self, islands: List[Island]) -> None:
        """Migrate best individuals between islands for cross-pollination."""
        if len(islands) < 2:
            return
        
        # Each island sends its best individuals to random other islands
        for source_island in islands:
            num_migrants = max(1, int(len(source_island.population) * self.migration_rate))
            migrants = source_island.get_best(num_migrants)
            
            if not migrants:
                continue
            
            # Send migrants to random other islands
            other_islands = [isl for isl in islands if isl.id != source_island.id]
            for migrant in migrants:
                target_island = random.choice(other_islands)
                
                # Create a copy of the migrant for the target island
                migrant_copy = Individual(
                    candidate=Candidate(
                        id=f"{migrant.candidate.id}_migrant_to_{target_island.id}",
                        patch_text=migrant.candidate.patch_text
                    ),
                    fitness=migrant.fitness,
                    eval_result=migrant.eval_result,
                    generation=migrant.generation,
                    parent_ids=migrant.parent_ids.copy()
                )
                
                target_island.add_individual(migrant_copy)

