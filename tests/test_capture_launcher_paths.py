from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / "run_capture_container.sh"
TEACHER_MODEL = Path("/home/brandonmusic/models/GLM-5.2-EXL3-TR3v4-3.5bpw-CORRECTED")
RUNTIME_OVERLAY = PROJECT / "evaluation/runtime_overlay"
VLLM_SOURCE = Path("/home/brandonmusic/KLC_SANDBOXES/.r10_prompt_logits_port/vllm")
RUNTIME_ROOT = Path(
    "/tmp/claude-1000/-home-brandonmusic-KLC-SANDBOXES/"
    "50980f6d-56ae-4115-a6bf-0d17377be8eb/scratchpad/prwork"
)
PROTECTED_ROOTS = (
    TEACHER_MODEL,
    PROJECT,
    RUNTIME_OVERLAY,
    VLLM_SOURCE,
    RUNTIME_ROOT,
)


def _run_launcher(capture: Path, cache: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LAUNCHER), "smoke"],
        env={
            "PATH": "/usr/bin:/bin",
            "FRESH_CAPTURE_PARENT": os.fspath(capture),
            "FRESH_JIT_CACHE": os.fspath(cache),
        },
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("protected", PROTECTED_ROOTS)
@pytest.mark.parametrize("writable", ("capture", "cache"))
def test_launcher_rejects_every_protected_root_for_each_writable_mount(
    tmp_path: Path,
    protected: Path,
    writable: str,
) -> None:
    assert protected.is_dir(), f"launcher fixture is absent: {protected}"
    capture = tmp_path / "capture"
    cache = tmp_path / "cache"
    capture.mkdir()
    cache.mkdir()
    if writable == "capture":
        capture = protected
    else:
        cache = protected

    result = _run_launcher(capture, cache)

    assert result.returncode == 2
    assert f"{writable.replace('cache', 'JIT cache')}" in result.stderr
    assert "must be disjoint from protected" in result.stderr


@pytest.mark.parametrize("writable", ("capture", "cache"))
def test_launcher_canonicalizes_symlink_aliases_to_protected_roots(
    tmp_path: Path,
    writable: str,
) -> None:
    capture = tmp_path / "capture"
    cache = tmp_path / "cache"
    alias = tmp_path / "protected-alias"
    capture.mkdir()
    cache.mkdir()
    alias.symlink_to(PROJECT, target_is_directory=True)
    if writable == "capture":
        capture = alias
    else:
        cache = alias

    result = _run_launcher(capture, cache)

    assert result.returncode == 2
    assert "must be disjoint from protected" in result.stderr
    assert str(PROJECT) in result.stderr


@pytest.mark.parametrize(
    ("overlap", "direction"),
    (
        (RUNTIME_OVERLAY / "b12x_sqg", "descendant"),
        (PROJECT.parent, "ancestor"),
    ),
)
@pytest.mark.parametrize("writable", ("capture", "cache"))
def test_launcher_rejects_protected_containment_in_both_directions(
    tmp_path: Path,
    overlap: Path,
    direction: str,
    writable: str,
) -> None:
    assert overlap.is_dir(), f"{direction} fixture is absent: {overlap}"
    capture = tmp_path / "capture"
    cache = tmp_path / "cache"
    capture.mkdir()
    cache.mkdir()
    if writable == "capture":
        capture = overlap
    else:
        cache = overlap

    result = _run_launcher(capture, cache)

    assert result.returncode == 2
    assert "must be disjoint from protected" in result.stderr


@pytest.mark.parametrize("writable", ("capture", "cache"))
def test_launcher_rejects_filesystem_root(
    tmp_path: Path,
    writable: str,
) -> None:
    capture = tmp_path / "capture"
    cache = tmp_path / "cache"
    capture.mkdir()
    cache.mkdir()
    if writable == "capture":
        capture = Path("/")
    else:
        cache = Path("/")

    result = _run_launcher(capture, cache)

    assert result.returncode == 2
    label = "capture parent" if writable == "capture" else "JIT cache"
    assert f"{label} must not be the filesystem root" in result.stderr


@pytest.mark.parametrize("direction", ("cache-inside-capture", "capture-inside-cache"))
def test_launcher_retains_bidirectional_capture_jit_disjointness(
    tmp_path: Path,
    direction: str,
) -> None:
    outer = tmp_path / "outer"
    inner = outer / "inner"
    outer.mkdir()
    inner.mkdir()
    capture, cache = (
        (outer, inner) if direction == "cache-inside-capture" else (inner, outer)
    )

    result = _run_launcher(capture, cache)

    assert result.returncode == 2
    assert "capture and JIT paths must be disjoint" in result.stderr


def test_launcher_canonicalizes_capture_jit_symlink_aliases(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    cache_alias = tmp_path / "cache-alias"
    capture.mkdir()
    cache_alias.symlink_to(capture, target_is_directory=True)

    result = _run_launcher(capture, cache_alias)

    assert result.returncode == 2
    assert "capture and JIT paths must be disjoint" in result.stderr


def test_smoke_launcher_normalizes_then_cross_checks_before_host_stamp() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    smoke_tail = source[source.index('if [[ "$mode" == smoke ]]'):]

    normalize_at = smoke_tail.index("normalize_jit_cache_permissions")
    digest_gate_at = smoke_tail.index(
        '[[ "$container_inventory_sha" =~ ^[0-9a-f]{64}$ ]]'
    )
    stamp_at = smoke_tail.index("write_jit_cache_binding")
    compare_at = smoke_tail.index("expected_inventory_sha256=sys.argv[5]")
    assert normalize_at < digest_gate_at < stamp_at < compare_at
    assert "--network none" in smoke_tail[:stamp_at]
    assert "--user 0:0" in smoke_tail[:stamp_at]
    assert '--mount "type=bind,src=$jit_cache,dst=/cache/jit"' in smoke_tail[:stamp_at]
    assert "--gpus" not in smoke_tail[:stamp_at]
