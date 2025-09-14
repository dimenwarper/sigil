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
                "# SIGIL:BEGIN example_region",
                "x = 1",
                "# SIGIL:END example_region",
                "",
            ]
        )
    )
    return f


def _write_valid_diff(tmp_path: Path) -> Path:
    patch = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1,4 +1,4 @@\n"
        " # header\n"
        " # SIGIL:BEGIN example_region\n"
        "-x = 1\n"
        "+x = 2\n"
        " # SIGIL:END example_region\n"
    )
    pf = tmp_path / "valid.diff"
    pf.write_text(patch)
    return pf


def _write_invalid_diff(tmp_path: Path) -> Path:
    patch = (
        "--- a/src/example.py\n"
        "+++ b/src/example.py\n"
        "@@ -1,4 +1,4 @@\n"
        "-# header\n"
        "+# hacked\n"
        " # SIGIL:BEGIN example_region\n"
        " x = 1\n"
        " # SIGIL:END example_region\n"
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
        "--mode",
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
            "--mode",
            "simple-llm",
            "--provider",
            "stub",
            "--backend",
            "ray",
        )

"""