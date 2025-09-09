"""
Command-line interface for Sigil.
"""

import click
import json
from pathlib import Path
from typing import Optional

from .core.config import get_config, save_config
from .core.models import SigilConfig, ResolverMode
from .optimization.base import OptimizationConfig
from .spec.spec import Spec, track, get_active_spec, list_active_specs
from .tracking.tracker import start_tracking, stop_tracking, is_tracking, get_tracker
from .workspace.workspace import Workspace
from .optimization.simple import SimpleOptimizer, RandomSearchOptimizer, GreedyOptimizer
from .resolver.resolver import get_resolver


@click.group()
@click.version_option()
def cli():
    """Sigil: A framework for LLM-guided code optimization."""
    pass


@cli.command()
@click.option("--workspace-dir", type=click.Path(), help="Workspace directory")
@click.option("--llm-provider", default="openai", help="LLM provider")
@click.option("--llm-model", default="gpt-4", help="LLM model")
def init(workspace_dir: Optional[str], llm_provider: str, llm_model: str):
    """Initialize a new Sigil configuration."""
    config = SigilConfig()
    
    if workspace_dir:
        config.workspace_dir = Path(workspace_dir)
    
    config.llm_provider = llm_provider
    config.llm_model = llm_model
    
    # Create workspace directory
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config)
    
    click.echo(f"Initialized Sigil workspace at {config.workspace_dir}")


@cli.group(name="tracker")
def tracker_group():
    """Sample tracking commands."""
    pass


@tracker_group.command(name="start")
def start_tracker():
    """Start sample tracking."""
    if is_tracking():
        click.echo("Tracking is already active")
        return
    
    start_tracking()
    click.echo("Started sample tracking")


@tracker_group.command(name="stop")
def stop_tracker():
    """Stop sample tracking."""
    if not is_tracking():
        click.echo("Tracking is not active")
        return
    
    stop_tracking()
    click.echo("Stopped sample tracking")


@tracker_group.command(name="status")
def tracker_status():
    """Show tracking status."""
    status = "active" if is_tracking() else "inactive"
    click.echo(f"Sample tracking: {status}")


@cli.command()
@click.argument("spec_name")
def inspect_samples(spec_name: str):
    """Inspect samples collected for a spec."""
    config = get_config()
    tracker = get_tracker()
    
    # Try to load the spec
    try:
        spec = Spec.load(spec_name)
    except FileNotFoundError:
        click.echo(f"Spec '{spec_name}' not found")
        return
    
    # Get samples for all pins in the spec
    samples = []
    for pin in spec.list_pins():
        pin_samples = tracker.read_samples(pin.function_id)
        samples.extend(pin_samples)
    
    if not samples:
        click.echo(f"No samples found for spec '{spec_name}'")
        return
    
    # Display samples in a table format
    click.echo("=" * 80)
    click.echo(f"{'Workspace':<15} {'Sample':<20} {'Eval Function':<15} {'Score':<10}")
    click.echo("=" * 80)
    
    for sample in samples[-20:]:  # Show last 20 samples
        workspace = sample.get("resolver_mode", "unknown")
        sample_id = sample.get("trace_id", "")[:8]
        eval_func = "unknown"
        score = sample.get("metrics", {}).get("score", "N/A")
        
        click.echo(f"{workspace:<15} {sample_id:<20} {eval_func:<15} {score:<10}")


@cli.command()
@click.argument("spec_name")
def inspect_solutions(spec_name: str):
    """Inspect optimization solutions for a spec."""
    try:
        spec = Spec.load(spec_name)
    except FileNotFoundError:
        click.echo(f"Spec '{spec_name}' not found")
        return
    
    # Find all workspaces for this spec
    config = get_config()
    spec_dir = config.workspace_dir / "workspaces" / spec_name
    
    if not spec_dir.exists():
        click.echo(f"No workspaces found for spec '{spec_name}'")
        return
    
    click.echo("=" * 100)
    click.echo(f"{'Workspace':<15} {'Summary':<40} {'Eval Function':<15} {'Score':<15}")
    click.echo("=" * 100)
    
    for workspace_dir in spec_dir.iterdir():
        if not workspace_dir.is_dir():
            continue
        
        workspace = Workspace(workspace_dir.name, spec_name, workspace_dir)
        
        # Get candidates for each pin
        for pin in spec.list_pins():
            candidates = workspace.list_candidates(pin.function_id)
            if candidates:
                best_candidate = workspace.get_best_candidate(pin.function_id)
                if best_candidate:
                    summary = f"Generated {len(candidates)} candidates"
                    eval_func = pin.eval_function or "none"
                    
                    # Get evaluation score
                    evaluations = workspace.get_evaluations(best_candidate.candidate_id)
                    score = "N/A"
                    if evaluations:
                        latest_eval = evaluations[-1]
                        score = str(latest_eval.metrics.get("score", "N/A"))
                    
                    click.echo(f"{workspace.name:<15} {summary:<40} {eval_func:<15} {score:<15}")


@cli.command()
@click.argument("spec_name")
@click.option("--name", default="default", help="Workspace name")
@click.option("--optimizer", default="simple", help="Optimizer to use")
@click.option("--niter", default=10, help="Number of iterations")
@click.option("--max-candidates", default=20, help="Maximum candidates")
def run(spec_name: str, name: str, optimizer: str, niter: int, max_candidates: int):
    """Run code optimization for a spec."""
    # Load the spec
    try:
        spec = Spec.load(spec_name)
    except FileNotFoundError:
        click.echo(f"Spec '{spec_name}' not found. Make sure to track it first.")
        return
    
    if not spec.list_pins():
        click.echo(f"No pins found in spec '{spec_name}'")
        return
    
    # Create workspace
    workspace = Workspace(name, spec_name)
    
    # Configure optimizer
    config = OptimizationConfig(
        max_iterations=niter,
        max_candidates=max_candidates
    )
    
    # Select optimizer
    optimizer_classes = {
        "simple": SimpleOptimizer,
        "random": RandomSearchOptimizer, 
        "greedy": GreedyOptimizer
    }
    
    if optimizer not in optimizer_classes:
        click.echo(f"Unknown optimizer '{optimizer}'. Available: {list(optimizer_classes.keys())}")
        return
    
    opt = optimizer_classes[optimizer](config)
    
    click.echo(f"Starting optimization run for spec '{spec_name}' using {optimizer} optimizer")
    click.echo(f"Workspace: {name}")
    click.echo(f"Max iterations: {niter}")
    click.echo(f"Max candidates: {max_candidates}")
    click.echo("=" * 60)
    
    total_candidates = 0
    
    # Optimize each pin
    for pin in spec.list_pins():
        click.echo(f"Optimizing function: {pin.function_id.qualname}")
        
        # Get evaluator if specified
        evaluator = None
        if pin.eval_function:
            evaluator = spec.get_evaluator(pin.eval_function)
            if not evaluator:
                click.echo(f"Warning: evaluator '{pin.eval_function}' not found")
        
        # Run optimization
        try:
            candidates = opt.optimize(pin, workspace, evaluator)
            total_candidates += len(candidates)
            click.echo(f"Generated {len(candidates)} candidates")
        except Exception as e:
            click.echo(f"Error optimizing {pin.function_id.qualname}: {e}")
    
    click.echo("=" * 60)
    click.echo(f"Optimization complete. Generated {total_candidates} total candidates.")
    click.echo(f"Results stored in workspace: {workspace.workspace_dir}")


@cli.command()
@click.argument("spec_name")
@click.argument("workspace_name")
@click.option("--function", help="Specific function to compare")
def compare(spec_name: str, workspace_name: str, function: Optional[str]):
    """Compare candidates to baseline."""
    try:
        spec = Spec.load(spec_name)
    except FileNotFoundError:
        click.echo(f"Spec '{spec_name}' not found")
        return
    
    workspace = Workspace(workspace_name, spec_name)
    
    pins_to_compare = spec.list_pins()
    if function:
        pins_to_compare = [pin for pin in pins_to_compare if function in pin.function_id.qualname]
    
    for pin in pins_to_compare:
        click.echo(f"\nComparing candidates for: {pin.function_id.qualname}")
        click.echo("-" * 60)
        
        candidates = workspace.list_candidates(pin.function_id)
        if not candidates:
            click.echo("No candidates found")
            continue
        
        for candidate_id in candidates[:5]:  # Show top 5
            candidate = workspace.get_candidate(candidate_id)
            if candidate:
                click.echo(f"Candidate: {str(candidate_id)[:16]}...")
                click.echo(f"Generator: {candidate.generator}")
                click.echo(f"Created: {candidate.created_at}")
                
                # Show evaluation results
                evaluations = workspace.get_evaluations(candidate_id)
                if evaluations:
                    latest = evaluations[-1]
                    click.echo(f"Score: {latest.metrics}")
                click.echo()


@cli.command()
@click.option("--mode", type=click.Choice(["off", "dev", "prod"]), help="Resolver mode")
def config(mode: Optional[str]):
    """Configure Sigil settings."""
    config = get_config()
    
    if mode:
        config.resolver_mode = ResolverMode(mode)
        save_config(config)
        click.echo(f"Resolver mode set to: {mode}")
    else:
        # Show current configuration
        click.echo("Current Sigil configuration:")
        click.echo(f"Workspace directory: {config.workspace_dir}")
        click.echo(f"Resolver mode: {config.resolver_mode.value}")
        click.echo(f"Tracking enabled: {config.tracker_enabled}")
        click.echo(f"LLM provider: {config.llm_provider}")
        click.echo(f"LLM model: {config.llm_model}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()