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


def test_run_baseline_creates_run_structure(tmp_path: Path):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    run_cli(tmp_path, "generate-eval", "--spec", "myspec", "walltime", "measure")
    _prepare_eval_commands_echo_ok(tmp_path)
    code = run_cli(tmp_path, "run", "--spec", "myspec", "--workspace", "ws1")
    assert code == 0
    runs_root = tmp_path / ".sigil" / "myspec" / "workspaces" / "ws1" / "runs"
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    assert run_dirs, "expected a run directory to be created"
    rd = sorted(run_dirs)[-1]
    assert (rd / "baseline" / "metrics.json").exists()
    assert (rd / "baseline" / "env.lock").exists()
    assert (rd / "run.json").exists()
    assert (rd / "index.json").exists()


def test_inspect_prints_nodes(tmp_path: Path, capsys):
    run_cli(tmp_path, "generate-spec", "myspec", "desc")
    run_cli(tmp_path, "generate-eval", "--spec", "myspec", "walltime", "measure")
    _prepare_eval_commands_echo_ok(tmp_path)
    run_cli(tmp_path, "run", "--spec", "myspec", "--workspace", "ws1")
    code = run_cli(tmp_path, "inspect", "--spec", "myspec", "--workspace", "ws1")
    assert code == 0
    out = capsys.readouterr().out
    assert "Nodes:" in out
    assert "BASELINE" in out


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
    cdir = leaves[0].parent
    assert (cdir / "parent").exists()
