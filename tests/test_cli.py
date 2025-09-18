import json
from pathlib import Path

import pytest

from sigil.cli import main


def run_cli(tmp_path: Path, *args: str) -> int:
    argv = ["--repo-root", str(tmp_path), *args]
    return main(argv)


def read_yaml(p: Path):
    import yaml

    return yaml.safe_load(p.read_text())


def test_generate_spec_creates_yaml_and_workspaces(tmp_path: Path):
    code = run_cli(tmp_path, "generate-spec", "myspec", "desc")
    assert code == 0
    spec_file = tmp_path / ".sigil" / "myspec.sigil.yaml"
    assert spec_file.exists()
    data = read_yaml(spec_file)
    assert data["name"] == "myspec"
    assert data.get("id"), "spec id should be generated"
    assert data.get("pins") and data["pins"][0].get("uuid")
    # workspace home exists
    assert (tmp_path / ".sigil" / "myspec" / "workspaces").exists()


def test_generate_eval_links_spec(tmp_path: Path):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    code = run_cli(tmp_path, "generate-eval", "--spec", "myspec", "walltime", "measure")
    assert code == 0
    eval_file = tmp_path / ".sigil" / "walltime.eval.yaml"
    assert eval_file.exists()
    spec_file = tmp_path / ".sigil" / "myspec.sigil.yaml"
    sdata = read_yaml(spec_file)
    assert "walltime" in (sdata.get("evals") or [])


def _prepare_eval_commands_echo_ok(tmp_path: Path):
    # Replace default checker command with a portable no-op: echo ok
    eval_file = tmp_path / ".sigil" / "walltime.eval.yaml"
    data = read_yaml(eval_file)
    for m in data.get("metrics", []):
        if m.get("kind") == "checker":
            m["command"] = "echo ok"
            m["parse"] = "exit_code==0"
    eval_file.write_text(__import__("yaml").safe_dump(data, sort_keys=False))


def test_inspect_prints_nodes(tmp_path: Path, capsys):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    run_cli(tmp_path, "generate-eval", "--spec", "myspec", "walltime", "measure")
    _prepare_eval_commands_echo_ok(tmp_path)
    run_cli(tmp_path, "run", "--spec", "myspec", "--workspace", "ws1")
    code = run_cli(tmp_path, "inspect", "--spec", "myspec", "--workspace", "ws1")
    assert code == 0
    out = capsys.readouterr().out
    assert "Nodes:" in out


def _write_pinned_file(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    f = src / "example.py"
    f.write_text(
        "\n".join(
            [
                "# header",
                "",
            ]
        )
    )
    return f


def _write_valid_diff(tmp_path: Path) -> Path:
    patch = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1,2 +1,2 @@\n"
        " # header\n"
        "-x = 1\n"
        "+x = 2\n"
    )
    pf = tmp_path / "valid.diff"
    pf.write_text(patch)
    return pf


def _write_invalid_diff(tmp_path: Path) -> Path:
    patch = (
        "--- a/src/dontmodify.py\n"
        "+++ b/src/dontmodify.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-# header\n"
        "+# hacked\n"
        " x = 1\n"
    )
    pf = tmp_path / "invalid.diff"
    pf.write_text(patch)
    return pf


def test_validate_patch_accepts_and_rejects(tmp_path: Path, capsys):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    _write_pinned_file(tmp_path)
    valid = _write_valid_diff(tmp_path)
    code_ok = run_cli(tmp_path, "validate-patch", "--spec", "myspec", "--patch-file", str(valid))
    assert code_ok == 0
    out = capsys.readouterr().out
    assert "VALID" in out

    invalid = _write_invalid_diff(tmp_path)
    code_bad = run_cli(tmp_path, "validate-patch", "--spec", "myspec", "--patch-file", str(invalid))
    assert code_bad != 0
    out2 = capsys.readouterr().out
    assert "INVALID" in out2


def test_add_candidate_stores_content_addressed(tmp_path: Path):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    run_cli(tmp_path, "generate-eval", "--spec", "myspec", "walltime", "measure")
    _prepare_eval_commands_echo_ok(tmp_path)
    run_cli(tmp_path, "run", "--spec", "myspec", "--workspace", "ws1")
    _write_pinned_file(tmp_path)
    patch = _write_valid_diff(tmp_path)

    # resolve latest run id
    runs_root = tmp_path / ".sigil" / "myspec" / "workspaces" / "ws1" / "runs"
    rd = sorted([p for p in runs_root.iterdir() if p.is_dir()])[-1]
    rid = rd.name

    code = run_cli(
        tmp_path,
        "add-candidate",
        "--spec",
        "myspec",
        "--workspace",
        "ws1",
        "--patch-file",
        str(patch),
        "--run",
        rid,
        "--parent",
        "BASELINE",
    )
    assert code == 0

    # candidates dir should now contain a content-addressed leaf with files
    croot = rd / "candidates"
    # find a deepest leaf containing patch.diff
    leaves = list(croot.rglob("patch.diff"))
    assert leaves, "expected at least one stored candidate"


def test_run_simple_llm_stub_creates_candidate_and_metrics(tmp_path: Path):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    run_cli(tmp_path, "generate-eval", "--spec", "myspec", "walltime", "measure")
    _prepare_eval_commands_echo_ok(tmp_path)
    # Pinned file with region and target content so stub can patch
    _write_pinned_file(tmp_path)

    num_candidates = 2

    # Run simple-llm with stub provider
    code = run_cli(
        tmp_path,
        "run",
        "--spec",
        "myspec",
        "--workspace",
        "ws2",
        "--llm",
        "simple-llm",
        "--provider",
        "stub",
        "--backend",
        "local",
        "--num",
        str(num_candidates),
    )
    assert code == 0

    # Validate run structure: baseline + one candidate with metrics
    runs_root = tmp_path / ".sigil" / "myspec" / "workspaces" / "ws2" / "runs"
    rd = sorted([p for p in runs_root.iterdir() if p.is_dir()])[-1]
    index = json.loads((rd / "index.json").read_text())
    nodes = index.get("nodes", [])
    # note the number of nodes need not match the number of candidates proposed (since they can be duplicates)
    # but they should be more than 1 and less than num_candidates
    assert 1 <= len(nodes) <= num_candidates

    # candidate directory
    croot = rd / "candidates"
    leaves = list(croot.rglob("patch.diff"))
    assert len(leaves) == len(nodes)
    cdir = leaves[0].parent
    assert (cdir / "metrics.json").exists()
    assert (cdir / "logs.txt").exists()


def test_sigil_run_in_symbolic_regression_directory():
    """Test that sigil run works in the symbolic_regression subdirectory"""
    import subprocess
    import os
    
    # Get the path to the symbolic_regression directory
    test_dir = Path(__file__).parent
    symbolic_regression_dir = test_dir / "symbolic_regression"
    
    # Ensure we're in the right directory
    assert symbolic_regression_dir.exists(), f"Symbolic regression directory not found: {symbolic_regression_dir}"
    assert (symbolic_regression_dir / ".sigil" / "symbolic_regression.sigil.yaml").exists(), "Spec file not found"
    assert (symbolic_regression_dir / ".sigil" / "quadratic_correctness.eval.yaml").exists(), "Eval file not found"
    assert (symbolic_regression_dir / "target_function.py").exists(), "Target function file not found"
    assert (symbolic_regression_dir / "test_correctness.py").exists(), "Test correctness file not found"
    
    # Change to the symbolic_regression directory and run sigil
    original_cwd = os.getcwd()
    try:
        os.chdir(symbolic_regression_dir)
        
        # Run sigil run command with stub provider to avoid external dependencies
        result = subprocess.run([
            "python", "-m", "sigil", "run",
            "--spec", "symbolic_regression",
            "--workspace", "test_workspace",
            "--llm", "simple-llm",
            "--provider", "stub",
            "--backend", "local",
            "--num", "2"
        ], capture_output=True, text=True, timeout=60)
        
        # Check that the command completed successfully
        assert result.returncode == 0, f"sigil run failed with code {result.returncode}. stdout: {result.stdout}, stderr: {result.stderr}"
        
        # Verify that output contains expected information
        assert "Simple LLM run completed" in result.stdout, f"Expected completion message not found in output: {result.stdout}"
        
        # Verify that run directory and files were created
        sigil_dir = symbolic_regression_dir / ".sigil"
        workspace_dir = sigil_dir / "symbolic_regression" / "workspaces" / "test_workspace" / "runs"
        
        assert workspace_dir.exists(), f"Workspace runs directory not created: {workspace_dir}"
        
        # Find the latest run directory
        run_dirs = [d for d in workspace_dir.iterdir() if d.is_dir()]
        assert len(run_dirs) > 0, "No run directories found"
        
        latest_run = sorted(run_dirs)[-1]
        
        # Verify expected files exist in the run directory
        assert (latest_run / "index.json").exists(), "index.json not found in run directory"
        assert (latest_run / "run.json").exists(), "run.json not found in run directory"
        assert (latest_run / "candidates").exists(), "candidates directory not found in run directory"
        
        # Verify index.json contains nodes
        index_data = json.loads((latest_run / "index.json").read_text())
        assert "nodes" in index_data, "nodes key not found in index.json"
        assert len(index_data["nodes"]) > 0, "No nodes found in index.json"
        
        print(f"Test passed: sigil run completed successfully in {latest_run}")
        
    finally:
        os.chdir(original_cwd)


import importlib.util as _importlib_util

"""
TODO: Come back to testing ray backend later

@pytest.mark.skipif(_importlib_util.find_spec("ray") is not None, reason="ray installed; error path not applicable")
def test_run_with_ray_backend_errors_if_missing(tmp_path: Path, monkeypatch):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    run_cli(tmp_path, "generate-eval", "--spec", "myspec", "walltime", "measure")
    _prepare_eval_commands_echo_ok(tmp_path)
    _write_pinned_file(tmp_path)

    # Force import error for ray by removing from sys.modules and PATH
    import sys

    sys.modules.pop("ray", None)

    with pytest.raises(SystemExit):
        run_cli(
            tmp_path,
            "run",
            "--spec",
            "myspec",
            "--workspace",
            "ws3",
            "--llm",
            "simple-llm",
            "--provider",
            "stub",
            "--backend",
            "ray",
        )

"""