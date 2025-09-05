"""Command-line interface for Sigil."""

import click
import inspect
from pathlib import Path
from typing import Optional

from .core import get_tracker
from .optimizers import get_optimizer


@click.group()
@click.version_option()
def main():
    """Sigil: A framework for auto-improving code through LLM-guided optimization."""
    pass


@main.command()
@click.argument("spec_name")
@click.option("--workspace", default=".", help="Path to workspace directory")
def inspect_samples(spec_name: str, workspace: str) -> None:
    """Inspect samples collected for a spec."""
    tracker = get_tracker()
    
    if spec_name not in tracker.specs:
        click.echo(f"Error: Spec '{spec_name}' not found.")
        return
    
    # Filter samples for this spec
    spec_samples = [s for s in tracker.samples if s["spec"] == spec_name]
    
    if not spec_samples:
        click.echo(f"No samples found for spec '{spec_name}'.")
        return
    
    # Display samples in table format
    click.echo("-" * 80)
    click.echo(f"{'workspace':<12} | {'sample':<20} | {'eval_function':<15} | {'score':<10}")
    click.echo("-" * 80)
    
    for sample in spec_samples:
        workspace_name = "default"  # TODO: Get from function metadata
        sample_desc = f"args={sample['args']}"[:18] + ".." if len(str(sample['args'])) > 18 else f"args={sample['args']}"
        eval_func = sample.get("eval_function", "N/A")
        score = sample.get("eval_score", "N/A")
        
        click.echo(f"{workspace_name:<12} | {sample_desc:<20} | {eval_func:<15} | {score:<10}")


@main.command()
@click.argument("spec_name")
@click.option("--workspace", default=".", help="Path to workspace directory")
def inspect_solutions(spec_name: str, workspace: str) -> None:
    """Inspect solutions found for a spec."""
    tracker = get_tracker()
    
    if spec_name not in tracker.specs:
        click.echo(f"Error: Spec '{spec_name}' not found.")
        return
    
    spec = tracker.specs[spec_name]
    solutions_path = spec.workspace_path / "solutions"
    
    if not solutions_path.exists():
        click.echo(f"No solutions found for spec '{spec_name}'.")
        return
    
    # Display placeholder for now
    click.echo("-" * 80)
    click.echo(f"{'workspace':<12} | {'summary':<30} | {'eval_function':<15} | {'score':<10}")
    click.echo("-" * 80)
    click.echo("Solutions inspection coming soon...")


@main.command()
@click.argument("spec_name")
@click.option("--name", default="default", help="Name for the optimization run")
@click.option("--optimizer", default="llm_sampler", type=click.Choice(["llm_sampler"]), help="Optimization algorithm")
@click.option("--niter", default=100, help="Number of iterations")
def run(spec_name: str, name: str, optimizer: str, niter: int) -> None:
    """Execute code optimization for a spec."""
    tracker = get_tracker()
    
    if spec_name not in tracker.specs:
        click.echo(f"Error: Spec '{spec_name}' not found.")
        return
    
    click.echo(f"Starting optimization run for spec '{spec_name}'...")
    click.echo(f"  Workspace: {name}")
    click.echo(f"  Optimizer: {optimizer}")
    click.echo(f"  Iterations: {niter}")
    
    # Find functions to optimize in this spec
    spec_samples = [s for s in tracker.samples if s["spec"] == spec_name]
    functions_to_optimize = list(set(s["function"] for s in spec_samples))
    
    if not functions_to_optimize:
        click.echo(f"No functions found to optimize for spec '{spec_name}'.")
        click.echo("Make sure you have run code with @improve decorators first.")
        return
    
    click.echo(f"Found {len(functions_to_optimize)} function(s) to optimize: {', '.join(functions_to_optimize)}")
    
    # Run optimization for each function
    for func_name in functions_to_optimize:
        click.echo(f"\nOptimizing function: {func_name}")
        
        # Get samples for this function
        func_samples = [s for s in spec_samples if s["function"] == func_name]
        
        # Create optimizer
        opt = get_optimizer(optimizer, spec_name, func_name, name)
        
        # For now, use placeholder code and eval function
        original_code = f"def {func_name}(*args, **kwargs):\n    # Original implementation\n    pass"
        eval_function = lambda x: 50.0  # Placeholder
        
        # Run optimization
        solution = opt.optimize(original_code, eval_function, func_samples, niter)
        
        if solution:
            click.echo(f"✓ Optimization completed for {func_name}")
        else:
            click.echo(f"✗ Optimization failed for {func_name}")
    
    click.echo(f"\nOptimization run complete for spec '{spec_name}'")


if __name__ == "__main__":
    main()