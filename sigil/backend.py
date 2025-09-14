from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .evals import EvalDef, run_eval_commands
import logging
from pathlib import Path as _Path
from .patches import make_patched_worktree, Candidate



@dataclass
class EvalResult:
    id: str
    metrics: Dict[str, Any]
    logs: Dict[str, Any]
    error: str | None = None


class Backend:
    def evaluate(self, eval_def: EvalDef, repo_root: Path, candidates: Iterable[Candidate]) -> List[EvalResult]:  # pragma: no cover - interface
        raise NotImplementedError


class LocalBackend(Backend):
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def _eval_one(self, eval_def: EvalDef, repo_root: Path, candidate: Candidate) -> EvalResult:
        try:
            worktree = make_patched_worktree(repo_root, candidate.patch_text)
            budgets = eval_def.budgets or {}
            timeout = budgets.get("candidate_timeout_s")
            out = run_eval_commands(eval_def, repo_root=worktree, workdir=worktree, timeout_s=timeout)
            return EvalResult(id=candidate.id, metrics=out.get("metrics", {}), logs=out.get("logs", {}))
        except Exception as e:
            return EvalResult(id=candidate.id, metrics={}, logs={}, error=str(e))

    def evaluate(self, eval_def: EvalDef, repo_root: Path, candidates: Iterable[Candidate]) -> List[EvalResult]:
        results: List[EvalResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers or min(4, len(candidates) or 1)) as ex:
            futs = {ex.submit(self._eval_one, eval_def, repo_root, candidate): candidate for candidate in candidates}
            for fut in as_completed(futs):
                results.append(fut.result())
        return results


class RayBackend(Backend):
    def __init__(self):
        try:
            import ray  # type: ignore
        except Exception as e:  # pragma: no cover - import path
            raise RuntimeError("Ray is not installed. Install with: pip install 'ray'") from e
        self._ray = ray
        if not ray.is_initialized():  # local, in-process cluster
            pkg_dir = _Path(__file__).resolve().parent  # .../<repo>/sigil
            ray.init(
                logging_level="INFO",
                runtime_env={
                    "py_modules": [str(pkg_dir)],
                    "working_dir": ".",
                },
            )
            logging.getLogger("sigil.backend").info("Initialized Ray")

        @ray.remote
        def _eval_remote(eval_def: EvalDef, repo_root: str, item: Candidate) -> EvalResult:  # type: ignore
            try:
                from sigil.patches import make_patched_worktree  # re-import in worker
                from sigil.evals import run_eval_commands

                worktree = make_patched_worktree(Path(repo_root), item.patch_text)
                budgets = eval_def.budgets or {}
                timeout = budgets.get("candidate_timeout_s")
                out = run_eval_commands(eval_def, repo_root=worktree, workdir=worktree, timeout_s=timeout)
                return EvalResult(id=item.id, metrics=out.get("metrics", {}), logs=out.get("logs", {}))
            except Exception as e:
                return EvalResult(id=item.id, metrics={}, logs={}, error=str(e))

        self._remote = _eval_remote

    def evaluate(self, eval_def: EvalDef, repo_root: Path, candidates: Iterable[Candidate]) -> List[EvalResult]:
        ray = self._ray
        items = list(candidates)
        if not items:
            return []
        logging.getLogger("sigil.backend").info("Submitting %d candidates to Ray", len(items))
        tasks = [self._remote.remote(eval_def, str(repo_root), it) for it in items]
        try:
            results = list(ray.get(tasks))
        except Exception as e:  # pragma: no cover - propagate ray errors with context
            raise RuntimeError(f"Ray evaluation failed: {e}") from e
        return results


def get_backend(kind: str) -> Backend:
    if kind == "local":
        return LocalBackend()
    elif kind == "ray":
        return RayBackend()
    else:
        raise ValueError(f"Unknown backend: {kind}")
