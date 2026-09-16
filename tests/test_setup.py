"""Bootstrap regressions that run without a GPU or simulator installation."""

import builtins
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bash():
    candidates = [shutil.which("bash")]
    if os.name == "nt":
        candidates.insert(0, str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("Bash is needed to exercise the Linux installer helpers")


def archive(tmp_path, bindings, compressed=False):
    target = tmp_path / ("gym package.tar.gz" if compressed else "gym package.tar")
    with tarfile.open(target, "w:gz" if compressed else "w") as tf:
        for binding in bindings:
            member = tarfile.TarInfo("isaacgym/python/isaacgym/_bindings/linux-x86_64/" + binding)
            tf.addfile(member, io.BytesIO())
    return target


def select_python(bash, package, requested=""):
    return subprocess.run(
        [bash, "-c", 'set -euo pipefail; source setup/isaacgym_common.sh; isaacgym_select_python "$1" "$2"',
         "test", package.as_posix(), requested],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )


@pytest.mark.parametrize("compressed", [False, True])
def test_package_python_version_handles_both_archive_formats(bash, tmp_path, compressed):
    package = archive(tmp_path, ["gym_36.so", "gym_38.so"], compressed)
    result = select_python(bash, package)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3.8"


def test_package_versions_sort_minor_numerically_and_honor_override(bash, tmp_path):
    package = archive(tmp_path, ["gym_39.so", "gym_310.so"])
    assert select_python(bash, package).stdout.strip() == "3.10"
    assert select_python(bash, package, "3.9").stdout.strip() == "3.9"
    assert select_python(bash, package, "3.8").returncode != 0


@pytest.mark.parametrize("bindings", [[], ["gym_37.so"]])
def test_missing_or_unsupported_bindings_do_not_silently_choose_python(bash, tmp_path, bindings):
    assert select_python(bash, archive(tmp_path, bindings)).returncode != 0


def test_invalid_archive_is_an_error(bash, tmp_path):
    package = tmp_path / "bad.tar"
    package.write_text("not an archive")
    assert select_python(bash, package).returncode != 0


@pytest.mark.parametrize("ptx", ["", "compute_80 compute_100"])
def test_package_checker_survives_missing_ptx_and_accepts_lower_target(bash, tmp_path, ptx):
    package = archive(tmp_path, ["gym_38.so", "libPhysXGpu_64.so"])
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "cuobjdump"
    stub.write_text(
        '#!/usr/bin/env bash\ncase "$1" in\n'
        '  --list-elf) echo sm_70 ;;\n'
        f'  --list-ptx) echo "{ptx}" ;;\n'
        'esac\n', encoding="utf8",
    )
    stub.chmod(0o755)
    result = subprocess.run(
        [bash, "-c", 'export PATH="$(cd "$2" && pwd):$PATH"; bash setup/check_isaacgym.sh "$1" --arch sm_89',
         "test", package.as_posix(), bin_dir.as_posix()],
        cwd=ROOT, capture_output=True, encoding="utf8", timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "Python: 3.8" in result.stdout
    assert "下一步" in result.stdout
    if ptx:
        assert "compute_80，可能支持 JIT" in result.stdout


def load_smoke(filename="smoke_test.py"):
    spec = importlib.util.spec_from_file_location("pgmt_smoke_test", ROOT / "setup" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failed_phase", [0, 1, 2, 3])
def test_smoke_phases_run_in_fresh_interpreters_and_stop_on_failure(monkeypatch, failed_phase):
    smoke = load_smoke()
    calls = []

    def child(command, check):
        calls.append(command)
        phase = int(command[-1])
        return SimpleNamespace(returncode=1 if phase == failed_phase else 0)

    def forbidden(*args):
        pytest.fail("The orchestrator must not import torch/Isaac Gym in its own process")

    for name in ("phase1", "phase2", "phase3"):
        monkeypatch.setattr(smoke, name, forbidden)
    monkeypatch.setattr(smoke.subprocess, "run", child)
    monkeypatch.setattr(sys, "argv", [str(ROOT / "setup/smoke_test.py"), "--num-envs", "2"])
    with pytest.raises(SystemExit) as result:
        smoke.main()
    assert result.value.code == failed_phase
    assert [int(c[-1]) for c in calls] == list(range(1, (failed_phase or 3) + 1))
    assert all(c[0] == sys.executable and c[-2] == "--phase" for c in calls)


@pytest.mark.parametrize("phase", [2, 3])
def test_gpu_phases_import_isaacgym_before_torch(monkeypatch, phase):
    smoke = load_smoke()
    real_import = builtins.__import__
    imports = []

    def checked_import(name, *args, **kwargs):
        if name == "isaacgym":
            imports.append(name)
            return SimpleNamespace(gymapi=object(), gymtorch=object())
        if name == "torch":
            imports.append(name)
            raise ImportError("Stop before GPU initialization")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    try:
        getattr(smoke, f"phase{phase}")(1, 1)
    except ImportError:
        pass
    assert "torch" in imports
    assert imports.index("isaacgym") < imports.index("torch")


@pytest.mark.parametrize("option", ["--num-envs", "--steps", "--iters"])
def test_smoke_rejects_empty_workloads(monkeypatch, option):
    smoke = load_smoke()
    monkeypatch.setattr(sys, "argv", ["smoke_test.py", option, "0"])
    with pytest.raises(SystemExit) as result:
        smoke.main()
    assert result.value.code == 2


def test_isaaclab_falling_box_has_collision_and_releases_resources(monkeypatch):
    import torch

    smoke = load_smoke("smoke_test_isaaclab.py")
    closed = []
    app = SimpleNamespace(close=lambda: closed.append("app"))
    sim = SimpleNamespace(reset=lambda: None, step=lambda: None,
                          close=lambda: closed.append("sim"))

    class RigidConfig(SimpleNamespace):
        InitialStateCfg = SimpleNamespace

    def rigid_object(cfg):
        # Without CollisionPropertiesCfg the box has no collision geometry.
        assert getattr(cfg.spawn, "collision_props", None) is not None
        return SimpleNamespace(
            write_data_to_sim=lambda: None, update=lambda **kwargs: None,
            data=SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.25]])),
        )

    sim_utils = SimpleNamespace(
        SimulationCfg=SimpleNamespace, SimulationContext=lambda cfg: sim,
        GroundPlaneCfg=SimpleNamespace, spawn_ground_plane=lambda *args: None,
        CuboidCfg=SimpleNamespace, RigidBodyPropertiesCfg=SimpleNamespace,
        MassPropertiesCfg=SimpleNamespace, CollisionPropertiesCfg=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace(sim=sim_utils))
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=lambda **kw: SimpleNamespace(app=app)))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", SimpleNamespace(RigidObject=rigid_object, RigidObjectCfg=RigidConfig))
    assert smoke.phase2(1)
    assert closed == ["sim", "app"]
