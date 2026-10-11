import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROBOT_RADIUS = 0.18518136649782
EXPECTED_WHEEL_RADIUS = 0.076


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _extract_value(content: str, key: str) -> float:
    match = re.search(rf"{key}:\s*([0-9.]+)", content)
    assert match, f"Could not find '{key}' in file"
    return float(match.group(1))


def test_sim_and_isaac_robot_radius_match_real_controller():
    real = _extract_value(
        _read("config/ros2_controller/real_controllers.yaml"), "robot_radius"
    )
    sim = _extract_value(
        _read("config/ros2_controller/sim_controllers.yaml"), "robot_radius"
    )
    isaac = _extract_value(
        _read("config/ros2_controller/isaac_controllers.yaml"), "robot_radius"
    )

    assert real == EXPECTED_ROBOT_RADIUS
    assert sim == EXPECTED_ROBOT_RADIUS
    assert isaac == EXPECTED_ROBOT_RADIUS


def test_gz_omni_wheel_uses_wheel_radius_property():
    content = _read("urdf/walkie/omni_wheel/gz_omni_wheel.urdf.xacro")

    radius_value = _extract_value(content, "wheel_radius")
    assert radius_value == EXPECTED_WHEEL_RADIUS
    assert "wheel_radian" not in content
    assert content.count("${wheel_radius}") == 2
