"""
Tests for optimization algorithms
"""

import tempfile
import shutil
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from sigil.optimization import (
    AlphaEvolveOptimizer, 
    SimpleOptimizer,
    Individual, 
    Island,
    get_optimizer
)
from sigil.spec import Spec, Pin
from sigil.patches import Candidate
from sigil.backend import EvalResult
from sigil.llm import PatchResponse


@pytest.fixture
def test_spec():
    """Create a test spec with minimal setup."""
    test_dir = Path(tempfile.mkdtemp(prefix="sigil_test_"))
    
    # Create a simple test file
    test_file = test_dir / "example.py"
    test_file.write_text("""
def slow_function(n):
    # Inefficient implementation
    result = 0
    for i in range(n):
        for j in range(n):
            result += 1
    return result

def main():
    print(slow_function(100))
""")
    
    # Create .sigil directory and eval
    sigil_dir = test_dir / ".sigil"
    sigil_dir.mkdir()
    
    # Create a test evaluation
    eval_config = {
        "version": "0.1",
        "name": "performance_test",
        "metrics": [
            {
                "id": "correctness",
                "kind": "checker",
                "command": "python -c 'import example; example.main()'",
                "parse": "exit_code==0"
            },
            {
                "id": "latency_ms",
                "kind": "timer",
                "command": "python -c 'import time; import example; start=time.time(); example.main(); print(f\"Time: {(time.time()-start)*1000:.2f}ms\")'",
                "parse": r"Time: ([0-9.]+)ms"
            }
        ],
        "budgets": {"candidate_timeout_s": 10}
    }
    
    eval_file = sigil_dir / "performance_test.eval.yaml"
    eval_file.write_text(yaml.safe_dump(eval_config))
    
    spec = Spec(
        version="0.1",
        name="test_optimization",
        description="Test optimization spec",
        repo_root=test_dir,
        evals=["performance_test"],
        pins=[Pin(
            id="slow_function",
            language="python", 
            files=["example.py"],
            symbol="slow_function"
        )]
    )
    
    yield spec
    
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


class TestIndividual:
    """Test Individual class."""
    
    def test_individual_creation(self):
        candidate = Candidate(id="test1", patch_text="diff content")
        individual = Individual(candidate=candidate, fitness=0.8)
        
        assert individual.candidate.id == "test1"
        assert individual.fitness == 0.8
        assert individual.generation == 0
        assert individual.parent_ids == []
    
    def test_individual_with_parents(self):
        candidate = Candidate(id="test1", patch_text="diff content")
        individual = Individual(
            candidate=candidate,
            fitness=0.9,
            generation=2,
            parent_ids=["parent1", "parent2"]
        )
        
        assert individual.generation == 2
        assert individual.parent_ids == ["parent1", "parent2"]


class TestIsland:
    """Test Island class."""
    
    def test_island_creation(self):
        island = Island(id="test_island")
        assert island.id == "test_island"
        assert island.population == []
        assert island.generation == 0
        assert island.best_fitness is None
        assert island.best_individual is None
    
    def test_add_individual_updates_best(self):
        island = Island(id="test_island")
        
        # Add individuals with different fitness
        candidates = [
            Candidate(id="ind1", patch_text="patch1"),
            Candidate(id="ind2", patch_text="patch2"),
            Candidate(id="ind3", patch_text="patch3")
        ]
        
        individuals = [
            Individual(candidate=candidates[0], fitness=0.6),
            Individual(candidate=candidates[1], fitness=0.9),  # Best
            Individual(candidate=candidates[2], fitness=0.7)
        ]
        
        for individual in individuals:
            island.add_individual(individual)
        
        assert len(island.population) == 3
        assert island.best_fitness == 0.9
        assert island.best_individual.candidate.id == "ind2"
    
    def test_get_best(self):
        island = Island(id="test_island")
        
        # Add individuals with different fitness
        candidates = [
            Candidate(id="ind1", patch_text="patch1"),
            Candidate(id="ind2", patch_text="patch2"), 
            Candidate(id="ind3", patch_text="patch3"),
            Candidate(id="ind4", patch_text="patch4")
        ]
        
        individuals = [
            Individual(candidate=candidates[0], fitness=0.6),
            Individual(candidate=candidates[1], fitness=0.9),
            Individual(candidate=candidates[2], fitness=0.7),
            Individual(candidate=candidates[3], fitness=None)  # No fitness
        ]
        
        for individual in individuals:
            island.add_individual(individual)
        
        # Get top 2
        best = island.get_best(2)
        assert len(best) == 2
        assert best[0].fitness == 0.9  # Best first
        assert best[1].fitness == 0.7  # Second best
    
    def test_replace_worst(self):
        island = Island(id="test_island")
        
        # Add initial population
        initial_candidates = [
            Candidate(id="old1", patch_text="old1"),
            Candidate(id="old2", patch_text="old2"),
            Candidate(id="old3", patch_text="old3")
        ]
        
        initial_individuals = [
            Individual(candidate=initial_candidates[0], fitness=0.5),  # Worst
            Individual(candidate=initial_candidates[1], fitness=0.7),  # Middle
            Individual(candidate=initial_candidates[2], fitness=0.9)   # Best
        ]
        
        for individual in initial_individuals:
            island.add_individual(individual)
        
        # Add new better individuals
        new_candidates = [
            Candidate(id="new1", patch_text="new1"),
            Candidate(id="new2", patch_text="new2")
        ]
        
        new_individuals = [
            Individual(candidate=new_candidates[0], fitness=0.8),
            Individual(candidate=new_candidates[1], fitness=0.6)
        ]
        
        island.replace_worst(new_individuals)
        
        # Should keep best original and replace worst with new ones
        remaining_ids = [ind.candidate.id for ind in island.population]
        assert "old3" in remaining_ids  # Best original kept
        assert "new1" in remaining_ids  # New individuals added
        assert "new2" in remaining_ids
        assert "old1" not in remaining_ids  # Worst removed


class TestAlphaEvolveOptimizer:
    """Test AlphaEvolve optimizer."""
    
    def test_optimizer_creation(self):
        optimizer = AlphaEvolveOptimizer(
            num_islands=2,
            population_size=4,
            num_generations=3,
            random_seed=42
        )
        
        assert optimizer.num_islands == 2
        assert optimizer.population_size == 4
        assert optimizer.num_generations == 3
        assert optimizer.random_seed == 42
    
    def test_fitness_computation(self):
        optimizer = AlphaEvolveOptimizer()
        
        # Test correct result with good performance
        result1 = EvalResult(
            id="test1",
            metrics={"correctness": True, "latency_ms": 100.0},
            logs={}
        )
        fitness1 = optimizer._compute_fitness(result1, Mock())
        assert fitness1 > 1.0  # Base fitness + performance bonus
        
        # Test incorrect result
        result2 = EvalResult(
            id="test2", 
            metrics={"correctness": False, "latency_ms": 50.0},
            logs={}
        )
        fitness2 = optimizer._compute_fitness(result2, Mock())
        assert fitness2 == 0.0  # Incorrect solutions get 0 fitness
        
        # Test error case
        result3 = EvalResult(
            id="test3",
            metrics={},
            logs={},
            error="Evaluation failed"
        )
        fitness3 = optimizer._compute_fitness(result3, Mock())
        assert fitness3 == 0.0
    
    def test_tournament_selection(self):
        optimizer = AlphaEvolveOptimizer(tournament_size=2, random_seed=42)
        island = Island(id="test")
        
        # Add individuals with different fitness
        candidates = [
            Candidate(id="weak", patch_text="patch1"),
            Candidate(id="strong", patch_text="patch2"),
            Candidate(id="medium", patch_text="patch3")
        ]
        
        individuals = [
            Individual(candidate=candidates[0], fitness=0.3),
            Individual(candidate=candidates[1], fitness=0.9), 
            Individual(candidate=candidates[2], fitness=0.6)
        ]
        
        for individual in individuals:
            island.add_individual(individual)
        
        # Tournament selection should favor higher fitness
        parents = optimizer._tournament_selection(island, 2)
        assert len(parents) == 2
        # With random seed, should be deterministic but still fitness-biased
        
    @patch('sigil.optimization.get_backend')
    @patch('sigil.optimization.load_eval')
    def test_initialize_islands(self, mock_load_eval, mock_get_backend, test_spec):
        """Test island initialization."""
        mock_provider = Mock()
        mock_provider.propose.return_value = PatchResponse(
            patch="test patch",
            rationale="test reasoning"
        )
        
        optimizer = AlphaEvolveOptimizer(
            num_islands=2,
            population_size=3,
            random_seed=42
        )
        
        islands = optimizer._initialize_islands(test_spec, mock_provider)
        
        assert len(islands) == 2
        assert all(len(island.population) == 3 for island in islands)
        assert all(island.id.startswith("island_") for island in islands)
        
        # Check that individuals were created
        for island in islands:
            for individual in island.population:
                assert individual.candidate.patch_text == "test patch"
                assert individual.generation == 0
    
    @patch('sigil.optimization.get_backend')
    @patch('sigil.optimization.load_eval')
    def test_evaluate_populations(self, mock_load_eval, mock_get_backend, test_spec):
        """Test population evaluation."""
        # Setup mocks
        mock_backend = Mock()
        mock_eval_def = Mock()
        mock_get_backend.return_value = mock_backend
        mock_load_eval.return_value = mock_eval_def
        
        # Mock evaluation results
        mock_backend.evaluate.return_value = [
            EvalResult(id="ind1", metrics={"correctness": True, "latency_ms": 100}, logs={}),
            EvalResult(id="ind2", metrics={"correctness": True, "latency_ms": 200}, logs={})
        ]
        
        optimizer = AlphaEvolveOptimizer()
        
        # Create test islands with individuals
        islands = [Island(id="island_0")]
        candidates = [
            Candidate(id="ind1", patch_text="patch1"),
            Candidate(id="ind2", patch_text="patch2")
        ]
        individuals = [
            Individual(candidate=candidates[0]),
            Individual(candidate=candidates[1])
        ]
        
        for individual in individuals:
            islands[0].add_individual(individual)
        
        optimizer._evaluate_populations(islands, mock_backend, mock_eval_def, test_spec.repo_root)
        
        # Check that fitness was assigned
        assert islands[0].population[0].fitness is not None
        assert islands[0].population[1].fitness is not None
        assert islands[0].best_fitness is not None
        
        # Verify backend was called
        mock_backend.evaluate.assert_called_once()
    
    def test_migration_between_islands(self):
        """Test migration between islands."""
        optimizer = AlphaEvolveOptimizer(migration_rate=0.5, random_seed=42)
        
        # Create two islands with different populations
        island1 = Island(id="island_1")
        island2 = Island(id="island_2")
        
        # Island 1 has better individuals
        candidates1 = [
            Candidate(id="i1_good", patch_text="good patch"),
            Candidate(id="i1_bad", patch_text="bad patch")
        ]
        individuals1 = [
            Individual(candidate=candidates1[0], fitness=0.9),
            Individual(candidate=candidates1[1], fitness=0.3)
        ]
        
        # Island 2 has mediocre individuals  
        candidates2 = [
            Candidate(id="i2_med1", patch_text="med patch 1"),
            Candidate(id="i2_med2", patch_text="med patch 2")
        ]
        individuals2 = [
            Individual(candidate=candidates2[0], fitness=0.6),
            Individual(candidate=candidates2[1], fitness=0.5)
        ]
        
        for ind in individuals1:
            island1.add_individual(ind)
        for ind in individuals2:
            island2.add_individual(ind)
        
        initial_pop1 = len(island1.population)
        initial_pop2 = len(island2.population)
        
        optimizer._migrate_between_islands([island1, island2])
        
        # Populations should have grown (migrants added)
        assert len(island1.population) >= initial_pop1
        assert len(island2.population) >= initial_pop2
        
        # Check that migrants were created (should have "_migrant_to_" in ID)
        migrant_ids1 = [ind.candidate.id for ind in island1.population if "_migrant_to_" in ind.candidate.id]
        migrant_ids2 = [ind.candidate.id for ind in island2.population if "_migrant_to_" in ind.candidate.id]
        
        # At least one migration should have occurred
        assert len(migrant_ids1) > 0 or len(migrant_ids2) > 0


class TestOptimizerFactory:
    """Test optimizer factory functions."""
    
    def test_get_optimizer_simple(self):
        optimizer = get_optimizer("simple")
        assert isinstance(optimizer, SimpleOptimizer)
    
    def test_get_optimizer_alphaevolve(self):
        optimizer = get_optimizer("alphaevolve", num_islands=3, population_size=10)
        assert isinstance(optimizer, AlphaEvolveOptimizer)
        assert optimizer.num_islands == 3
        assert optimizer.population_size == 10
    
    def test_get_optimizer_unknown(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            get_optimizer("unknown_optimizer")


class TestIntegration:
    """Integration tests for AlphaEvolve."""
    
    @patch('sigil.optimization.get_backend')
    @patch('sigil.optimization.load_eval')
    def test_alphaevolve_propose_integration(self, mock_load_eval, mock_get_backend, test_spec):
        """Test full AlphaEvolve propose method with mocked dependencies."""
        # Setup mocks
        mock_backend = Mock()
        mock_eval_def = Mock()
        mock_get_backend.return_value = mock_backend
        mock_load_eval.return_value = mock_eval_def
        
        # Mock LLM provider
        mock_provider = Mock()
        mock_provider.propose.return_value = PatchResponse(
            patch="--- a/example.py\n+++ b/example.py\n@@ -1,5 +1,5 @@\n def optimized():\n-    pass\n+    return 42",
            rationale="Optimized the function"
        )
        
        # Mock evaluation results - make some candidates successful
        def mock_evaluate(eval_def, repo_root, candidates):
            results = []
            for i, candidate in enumerate(candidates):
                # Make every other candidate successful
                success = i % 2 == 0
                results.append(EvalResult(
                    id=candidate.id,
                    metrics={"correctness": success, "latency_ms": 100 + i * 10},
                    logs={}
                ))
            return results
        
        mock_backend.evaluate.side_effect = mock_evaluate
        
        # Create optimizer with small parameters for fast test
        optimizer = AlphaEvolveOptimizer(
            num_islands=2,
            population_size=4,
            num_generations=2,
            migration_interval=2,
            backend_type="local",
            random_seed=42
        )
        
        # Run the optimization
        responses = optimizer.propose(test_spec, mock_provider, num=2)
        
        # Verify results
        assert len(responses) == 2
        assert all(isinstance(r, PatchResponse) for r in responses)
        assert all("Evolved candidate" in r.rationale for r in responses)
        
        # Verify backend was called multiple times (initial + evolution)
        assert mock_backend.evaluate.call_count >= 2
        
        # Verify provider was called multiple times (initial population + evolution)
        assert mock_provider.propose.call_count >= 8  # 2 islands * 4 population
