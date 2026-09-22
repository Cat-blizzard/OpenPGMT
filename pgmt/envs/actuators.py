"""G1 effort limits verified against the local ProtoMotions URDF and USD.

These are asset specifications, not controller parameters disclosed by PGMT.
Only effort changes relative to v4: keep kp=80, kd=2, zero reset, and the
USD's existing velocity limits. See docs/actuator_diagnostics_20260922.md.
"""
from data.retarget_lafan1 import G1_JOINT_NAMES


def _effort(name):
    if name.endswith(("wrist_pitch", "wrist_yaw")):
        return 5.
    if any(part in name for part in ("shoulder", "elbow", "wrist_roll")):
        return 25.
    if "knee" in name:
        return 139.
    if "ankle" in name or name in ("waist_roll", "waist_pitch"):
        return 50.
    if "hip" in name or name == "waist_yaw":
        return 88.
    raise ValueError(f"unrecognized G1 joint: {name}")


G1_EFFORT_LIMITS = tuple(_effort(name) for name in G1_JOINT_NAMES)
ACTUATOR_PROFILES = ("asset_effort_v1", "legacy_uniform120")


def actuator_options(profile="asset_effort_v1"):
    if profile not in ACTUATOR_PROFILES:
        raise ValueError(f"unknown actuator profile: {profile}")
    return {"torque_limit": G1_EFFORT_LIMITS if profile == "asset_effort_v1" else (120.,) * 29}


def validate_actuator_resume(checkpoint, cfg):
    """A resume is not an actuator ablation. Require the saved controller."""
    saved = checkpoint.get("environment_config")
    if saved is None:
        raise ValueError("resume lacks environment_config; cannot verify actuator settings")
    if saved.get("reset_mode", "nominal") != cfg.reset_mode:
        raise ValueError("resume reset_mode differs; start a fresh reset ablation")
    for key in ("torque_limit", "stiffness", "damping", "default_joint_pos"):
        value = saved.get(key)
        if value is None or tuple(value) != tuple(getattr(cfg, key)):
            raise ValueError(f"resume actuator setting {key} differs; select the saved profile or start a fresh run")
