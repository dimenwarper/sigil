from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pathlib import Path as _Path

from .evals import EvalDef, run_eval_commands
from .patches import make_patched_worktree, Candidate


def _discover_local_python_packages(repo_root: Path) -> List[Path]:
    """Return top-level packages (directories containing __init__.py)."""
    packages: List[Path] = []
    for entry in repo_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        if (entry / "__init__.py").exists():
            packages.append(entry)
    return packages


def _zip_directory(src: Path) -> bytes:
    """Zip directory contents and return as bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(src)
            zf.write(path, arcname=str(rel))
    return buffer.getvalue()


def _unzip_into(data: bytes, dest: Path) -> None:
    """Extract zip bytes into destination directory."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest)



@dataclass
class EvalResult:
    id: str
    metrics: Dict[str, Any]
    logs: Dict[str, Any]
    error: str | None = None


class Backend:
    def evaluate(self, eval_def: EvalDef, repo_root: Path, candidates: Iterable[Candidate]) -> List[EvalResult]:
        raise NotImplementedError


class LocalBackend(Backend):
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def _eval_one(self, eval_def: EvalDef, repo_root: Path, candidate: Candidate) -> EvalResult:
        try:
            worktree = make_patched_worktree(repo_root, candidate.patch_text)
            budgets = eval_def.budgets or {}
            timeout = budgets.get("candidate_timeout_s")
            out = run_eval_commands(eval_def, repo_root=worktree, timeout_s=timeout)
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
        except Exception as e: 
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
                out = run_eval_commands(eval_def, repo_root=worktree, timeout_s=timeout)
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
        except Exception as e:
            raise RuntimeError(f"Ray evaluation failed: {e}") from e
        return results


class ModalBackend(Backend):
    """Execute evaluations inside a Modal container image."""

    def __init__(
        self,
        *,
        app_name: str | None = None,
        pyproject_path: Path | None = None,
        volume_name: str | None = None,
    ) -> None:
        try:
            import modal  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised only when modal missing
            raise RuntimeError(
                "Modal backend requires the 'modal' package. Install with: pip install modal"
            ) from exc

        self._modal = modal
        self._pyproject_path = pyproject_path
        self._app_name = app_name or "sigil-modal-backend"
        self._volume_name = volume_name
        self._remote_cache: Dict[Path, Any] = {}

    # -- image construction helpers -------------------------------------------------
    def _load_dependencies(self, repo_root: Path) -> List[str]:
        pyproject = self._pyproject_path or (repo_root / "pyproject.toml")
        if not pyproject.exists():
            return []
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # pragma: no cover - python <3.11 fallback
                import tomli as tomllib  # type: ignore

            with pyproject.open("rb") as f:
                data = tomllib.load(f)
        except Exception as exc:
            logging.getLogger("sigil.backend").warning(
                "Failed to parse %s for dependencies: %s", pyproject, exc
            )
            return []

        project = data.get("project") or {}
        deps = project.get("dependencies") or []
        return [str(d) for d in deps]

    def _build_image(self, repo_root: Path):
        modal = self._modal
        image = modal.Image.debian_slim().pip_install("uv")
        deps = self._load_dependencies(repo_root)
        if deps:
            image = image.pip_install(*deps)
        for pkg_path in _discover_local_python_packages(repo_root):
            image = image.add_local_python_source(str(pkg_path))
        return image

    def _resolve_volume_map(self):
        if not self._volume_name:
            return None
        modal = self._modal
        vol = modal.Volume.from_name(self._volume_name, create_if_missing=True)
        return {"/artifacts": vol}

    # -- remote function setup ------------------------------------------------------
    def _ensure_remote(self, repo_root: Path):
        repo_root = repo_root.resolve()
        cached = self._remote_cache.get(repo_root)
        if cached:
            return cached

        modal = self._modal
        app = modal.App(f"{self._app_name}-{repo_root.name}")
        image = self._build_image(repo_root)
        volume_map = self._resolve_volume_map()

        @app.function(image=image, volumes=volume_map)
        def _eval_remote(eval_def: EvalDef, candidate_id: str, archive: bytes, timeout_s: int | None):
            import shutil as _shutil
            import tempfile as _tempfile
            from pathlib import Path as _PathRemote

            workdir = _PathRemote(_tempfile.mkdtemp(prefix="sigil-modal-"))
            try:
                _unzip_into(archive, workdir)
                result = run_eval_commands(eval_def, repo_root=workdir, timeout_s=timeout_s)
                return {
                    "id": candidate_id,
                    "metrics": result.get("metrics", {}),
                    "logs": result.get("logs", {}),
                    "error": None,
                }
            except Exception as exc:  # pragma: no cover - remote execution path
                return {
                    "id": candidate_id,
                    "metrics": {},
                    "logs": {},
                    "error": str(exc),
                }
            finally:
                _shutil.rmtree(workdir, ignore_errors=True)

        self._remote_cache[repo_root] = _eval_remote
        return _eval_remote

    # -- backend interface ---------------------------------------------------------
    def _package_candidate(self, repo_root: Path, candidate: Candidate) -> bytes:
        worktree = make_patched_worktree(repo_root, candidate.patch_text)
        try:
            return _zip_directory(worktree)
        finally:
            shutil.rmtree(worktree, ignore_errors=True)

    def evaluate(
        self, eval_def: EvalDef, repo_root: Path, candidates: Iterable[Candidate]
    ) -> List[EvalResult]:
        items = list(candidates)
        if not items:
            return []

        remote = self._ensure_remote(repo_root)
        budgets = eval_def.budgets or {}
        timeout = budgets.get("candidate_timeout_s")

        results: List[EvalResult] = []
        for candidate in items:
            try:
                archive = self._package_candidate(repo_root, candidate)
            except Exception as exc:
                results.append(
                    EvalResult(id=candidate.id, metrics={}, logs={}, error=str(exc))
                )
                continue

            try:
                payload = remote.call(eval_def, candidate.id, archive, timeout)
            except Exception as exc:  # pragma: no cover - remote failure path
                results.append(
                    EvalResult(id=candidate.id, metrics={}, logs={}, error=str(exc))
                )
                continue

            results.append(
                EvalResult(
                    id=str(payload.get("id", candidate.id)),
                    metrics=payload.get("metrics", {}),
                    logs=payload.get("logs", {}),
                    error=payload.get("error"),
                )
            )

        return results


def get_backend(kind: str) -> Backend:
    if kind == "local":
        return LocalBackend()
    elif kind == "ray":
        return RayBackend()
    elif kind == "modal":
        return ModalBackend()
    else:
        raise ValueError(f"Unknown backend: {kind}")
