import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "setup" / "check_g1_asset.py"
_SPEC = importlib.util.spec_from_file_location("check_g1_asset", _SCRIPT)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_repo_mjcf_mapping_is_29_dof_and_one_to_one():
    report = _MOD.analyze_mjcf(Path(__file__).parents[1] / "data/raw/g1/g1_29dof.xml")
    assert report["joint_count"] == 29
    assert report["checks"]["expected_joint_order"]
    assert report["checks"]["30_one_to_one_actuators"]
    # The checked-in XML intentionally does not vendor Unitree meshes.
    assert len(report["missing_meshes"]) == 36
    assert not report["checks"]["mesh_files_resolve"]


def test_external_protomotions_candidate_is_complete_when_available():
    root = Path("/data/jxc/projects/ProtoMotions-v2.3/protomotions/data/assets")
    urdf = root / "urdf/g1.urdf"
    mjcf = root / "mjcf/g1.xml"
    if not (urdf.is_file() and mjcf.is_file()):
        return
    urdf_report = _MOD.analyze_urdf(urdf)
    mjcf_report = _MOD.analyze_mjcf(mjcf)
    assert urdf_report["checks"]["29_revolute_joints"]
    assert urdf_report["checks"]["expected_joint_order"]
    assert urdf_report["checks"]["mesh_files_resolve"]
    assert mjcf_report["checks"]["30_one_to_one_actuators"]
    assert mjcf_report["checks"]["mesh_files_resolve"]


def test_usd_probe_is_conservative(tmp_path):
    usd = tmp_path / "minimal.usd"
    usd.write_bytes(b"PXR-USDC\x00test")
    report = _MOD.analyze_usd(usd)
    assert report["checks"] == {"usd_file_present": True, "joint_mapping_inspected": False}
