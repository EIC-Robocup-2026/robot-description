"""
Structural and contract tests for the walkie robot description.

Every test expands robots/gz_walkie.urdf.xacro from the installed walkie_description
share directory, so the package must be built (and its workspace sourced) first.
Expansions are cached per configuration, so each one is processed only once per run.
"""

import functools
import math
import os
import re
from xml.etree import ElementTree

from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError
import pytest
import xacro
import yaml


PACKAGE = 'walkie_description'

# Named xacro configurations. Values are passed to xacro as string mappings, exactly as the
# launch files do. 'use_arm' is declared by gz_walkie.urdf.xacro and passed by the launch
# files, so it is expanded here even though the arm is currently always included.
CONFIGURATIONS = {
    'mock': {'ros2_control': 'mock'},
    'gazebo': {'ros2_control': 'gazebo'},
    'topic_base': {'ros2_control': 'topic_base'},
    'real_robot': {'ros2_control': 'real_robot'},
    'mujoco': {'ros2_control': 'mujoco'},
    'gazebo_single_lidar': {'ros2_control': 'gazebo', 'dual_lidar': 'False'},
    'gazebo_no_arm': {'ros2_control': 'gazebo', 'use_arm': 'False'},
    # real_omnibot.launch.py defaults to driving the arms through the OpenArm CAN plugin.
    'real_robot_arm_hardware': {'ros2_control': 'real_robot', 'use_fake_arm_hardware': 'False'},
    'gazebo_right_joint2_fixed': {'ros2_control': 'gazebo', 'right_joint2_fixed': 'True'},
}

# Which controller parameter file is loaded together with which ros2_control mode, as wired
# by the launch files:
#   gazebo     robots/gz_walkie.urdf.xacro (sim_controllers_yaml default, read by the
#              gz_ros2_control plugin); simulation/gazebo/launch/gzsim_omnibot.launch.py
#   mujoco     mujoco_controllers.yaml, for the mujoco_ros2_control controller manager
#   topic_base bringup/launch/isaac_omnibot.launch.py
#   real_robot bringup/launch/real_omnibot.launch.py
#   mock       bringup/launch/test_zed.launch.py
CONTROLLER_PAIRINGS = [
    ('gazebo', 'sim_controllers.yaml'),
    ('mujoco', 'mujoco_controllers.yaml'),
    ('topic_base', 'isaac_controllers.yaml'),
    ('real_robot', 'real_controllers.yaml'),
    ('mock', 'real_controllers.yaml'),
]
CONTROLLER_FILES = sorted({controllers_file for _, controllers_file in CONTROLLER_PAIRINGS})

# Known mismatches between a controller file and the mode it is loaded with. Each one needs a
# decision outside this test (a controller file per mode, or different hardware interfaces), so
# it is marked as a strict expected failure: the test turns red once the mismatch is resolved,
# and the entry must then be removed.
KNOWN_PAIRING_MISMATCHES = {
    ('mock', 'real_controllers.yaml'):
        'joint_broad lists effort state interfaces for the wheels and velocity/effort for '
        'head_servo_joint, which the mock branch does not declare, so test_zed.launch.py '
        'cannot activate it',
}

# Controller parameters that name hardware joints.
JOINT_PARAMETERS = ('joints', 'joint', 'wheel_names')

# Frame parameters whose frames are published outside the URDF (odometry, localization).
EXTERNAL_FRAME_PARAMETERS = frozenset({'odom_frame_id', 'map_frame_id', 'world_frame_id'})

JOINT_STATE_BROADCASTER = 'joint_state_broadcaster/JointStateBroadcaster'
JOINT_TRAJECTORY_CONTROLLER = 'joint_trajectory_controller/JointTrajectoryController'
FORWARD_COMMAND_CONTROLLER = 'forward_command_controller/ForwardCommandController'
JOINT_GROUP_CONTROLLERS = {
    'position_controllers/JointGroupPositionController': 'position',
    'velocity_controllers/JointGroupVelocityController': 'velocity',
    'effort_controllers/JointGroupEffortController': 'effort',
}
GRIPPER_ACTION_CONTROLLERS = {
    'position_controllers/GripperActionController': 'position',
    'effort_controllers/GripperActionController': 'effort',
}
OMNI_CONTROLLER = 'omni_wheel_drive_controller/OmniWheelDriveController'
IMU_SENSOR_BROADCASTER = 'imu_sensor_broadcaster/IMUSensorBroadcaster'
# State interfaces semantic_components::IMUSensor claims on its sensor.
IMU_STATE_INTERFACES = tuple(
    f'{group}.{axis}'
    for group, axes in (
        ('orientation', 'xyzw'), ('angular_velocity', 'xyz'), ('linear_acceleration', 'xyz'))
    for axis in axes)

# Shared contract for the MuJoCo ros2_control branch: wheels are velocity-commanded,
# every other actuated joint is position-commanded, mimic joints are not declared.
WHEEL_JOINTS = ('fl_wheel_joint', 'fr_wheel_joint', 'bl_wheel_joint', 'br_wheel_joint')
MUJOCO_REQUIRED_STATE = frozenset({'position', 'velocity'})
MUJOCO_WHEEL_STATE = frozenset({'position', 'velocity'})
MUJOCO_JOINT_STATE = frozenset({'position', 'velocity', 'effort'})
MUJOCO_HARDWARE_PLUGIN = 'mujoco_ros2_control/MujocoSystemInterface'

ROBOT_RADIUS_TOLERANCE = 0.001  # m
WHEEL_RADIUS_TOLERANCE = 0.002  # m
WHEEL_ANGLE_TOLERANCE = math.radians(1.0)

UNRESOLVED_XACRO = re.compile(r'\$[{(]')

# Rigid transforms are (rotation, translation) pairs of nested lists.
IDENTITY = ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])


def _share_dir():
    return get_package_share_directory(PACKAGE)


def _rpy_rotation(roll, pitch, yaw):
    """Return the URDF rotation Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _origin_pose(element):
    origin = element.find('origin')
    xyz = origin.get('xyz', '0 0 0') if origin is not None else '0 0 0'
    rpy = origin.get('rpy', '0 0 0') if origin is not None else '0 0 0'
    return _rpy_rotation(*map(float, rpy.split())), [float(value) for value in xyz.split()]


def _compose(first, second):
    rotation_a, translation_a = first
    rotation_b, translation_b = second
    rotation = [
        [sum(rotation_a[i][k] * rotation_b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    translation = [
        sum(rotation_a[i][k] * translation_b[k] for k in range(3)) + translation_a[i]
        for i in range(3)
    ]
    return rotation, translation


def _inverse(pose):
    rotation, translation = pose
    transposed = [[rotation[j][i] for j in range(3)] for i in range(3)]
    return transposed, [
        -sum(transposed[i][k] * translation[k] for k in range(3)) for i in range(3)
    ]


class RobotModel:
    """Indexed view of an expanded URDF."""

    def __init__(self, root):
        """Index the links and joints of a parsed <robot> element."""
        self.root = root
        self.link_elements = root.findall('link')
        self.joint_elements = root.findall('joint')
        self.links = {link.get('name'): link for link in self.link_elements}
        self.joints = {joint.get('name'): joint for joint in self.joint_elements}
        self.parent_joint = {
            joint.find('child').get('link'): joint
            for joint in self.joint_elements
            if joint.find('child') is not None
        }

    def hardware_interfaces(self):
        """Map each ros2_control joint to the union of its declared interfaces."""
        interfaces = {}
        for block in self.root.findall('ros2_control'):
            for joint in block.findall('joint'):
                declared = interfaces.setdefault(
                    joint.get('name'), {'command': set(), 'state': set()})
                declared['command'].update(
                    item.get('name') for item in joint.findall('command_interface'))
                declared['state'].update(
                    item.get('name') for item in joint.findall('state_interface'))
        return interfaces

    def sensor_interfaces(self):
        """Map each ros2_control sensor to the union of its declared state interfaces."""
        interfaces = {}
        for block in self.root.findall('ros2_control'):
            for sensor in block.findall('sensor'):
                interfaces.setdefault(sensor.get('name'), set()).update(
                    item.get('name') for item in sensor.findall('state_interface'))
        return interfaces

    def link_pose(self, link):
        """Return the pose of a link in the root link frame."""
        pose = IDENTITY
        visited = set()
        while link in self.parent_joint and link not in visited:
            visited.add(link)
            joint = self.parent_joint[link]
            pose = _compose(_origin_pose(joint), pose)
            link = joint.find('parent').get('link')
        return pose

    def joint_pose(self, joint_name):
        """Return the pose of a joint frame in the root link frame."""
        joint = self.joints[joint_name]
        return _compose(self.link_pose(joint.find('parent').get('link')), _origin_pose(joint))


@functools.lru_cache(maxsize=None)
def _robot(config_id):
    """Expand and index one named configuration, once per test session."""
    xacro_file = os.path.join(_share_dir(), 'robots', 'gz_walkie.urdf.xacro')
    # xacro stores argument defaults in the mappings dict it receives, so pass a copy.
    mappings = dict(CONFIGURATIONS[config_id])
    document = xacro.process_file(xacro_file, mappings=mappings)
    # ElementTree drops XML comments, so commented-out xacro snippets are ignored.
    return RobotModel(ElementTree.fromstring(document.toxml()))


@functools.lru_cache(maxsize=None)
def _controller_config(controllers_file):
    path = os.path.join(_share_dir(), 'config', 'ros2_controller', controllers_file)
    with open(path) as stream:
        return yaml.safe_load(stream)


def _controllers(controllers_file):
    """Yield (name, type, parameters) for every controller the controller manager declares."""
    config = _controller_config(controllers_file)
    manager = config.get('controller_manager', {}).get('ros__parameters', {})
    for name, entry in manager.items():
        if isinstance(entry, dict) and 'type' in entry:
            params = (config.get(name) or {}).get('ros__parameters') or {}
            yield name, entry['type'], params


def _claimed_interfaces(controller_type, params):
    """
    Return the (joint, kind, interface) triples a controller claims when it activates.

    kind is 'command' or 'state'. The rules mirror what each controller type requests from
    the controller manager, so parameters a type ignores (for example command_interfaces
    on a JointGroupPositionController) are not treated as claims. Return None for
    controller types this test does not model.
    """
    joints = params.get('joints') or []
    claims = []
    if controller_type == JOINT_STATE_BROADCASTER:
        # Without both lists the broadcaster reads whatever state interfaces exist.
        names = params.get('interfaces') or []
        if joints and names:
            claims += [(joint, 'state', name) for joint in joints for name in names]
    elif controller_type == JOINT_TRAJECTORY_CONTROLLER:
        for joint in joints:
            claims += [
                (joint, 'command', name) for name in params.get('command_interfaces') or []]
            claims += [(joint, 'state', name) for name in params.get('state_interfaces') or []]
    elif controller_type == FORWARD_COMMAND_CONTROLLER:
        claims += [(joint, 'command', params.get('interface_name')) for joint in joints]
    elif controller_type in JOINT_GROUP_CONTROLLERS:
        interface = JOINT_GROUP_CONTROLLERS[controller_type]
        claims += [(joint, 'command', interface) for joint in joints]
    elif controller_type in GRIPPER_ACTION_CONTROLLERS:
        joint = params.get('joint')
        claims += [
            (joint, 'command', GRIPPER_ACTION_CONTROLLERS[controller_type]),
            (joint, 'state', 'position'),
            (joint, 'state', 'velocity'),
        ]
    elif controller_type == OMNI_CONTROLLER:
        feedback = 'position' if params.get('position_feedback', True) else 'velocity'
        for wheel in params.get('wheel_names') or []:
            claims += [(wheel, 'command', 'velocity'), (wheel, 'state', feedback)]
    elif controller_type == IMU_SENSOR_BROADCASTER:
        pass  # Claims sensor interfaces only; see _claimed_sensor_interfaces.
    else:
        return None
    return list(dict.fromkeys(claims))


def _claimed_sensor_interfaces(controller_type, params):
    """Return the (sensor, interface) pairs a controller claims when it activates."""
    if controller_type == IMU_SENSOR_BROADCASTER:
        return [(params.get('sensor_name'), name) for name in IMU_STATE_INTERFACES]
    return []


def _referenced_joints(params, claims):
    referenced = {joint for joint, _, _ in claims}
    for key in JOINT_PARAMETERS:
        value = params.get(key)
        if isinstance(value, str):
            referenced.add(value)
        elif isinstance(value, list):
            referenced.update(value)
    return referenced


def _frame_parameters(params, prefix=''):
    """Yield (parameter, frame) for every *frame_id parameter that must name a URDF link."""
    for key, value in params.items():
        if isinstance(value, dict):
            yield from _frame_parameters(value, f'{prefix}{key}.')
        elif key.endswith('frame_id') and key not in EXTERNAL_FRAME_PARAMETERS and value:
            yield prefix + key, value


def _resolve_uri(uri):
    """Return the filesystem path of a package:// or file:// URI, or None."""
    if uri.startswith('package://'):
        package, _, relative = uri[len('package://'):].partition('/')
        try:
            return os.path.join(get_package_share_directory(package), relative)
        except PackageNotFoundError:
            return None
    if uri.startswith('file://'):
        return uri[len('file://'):]
    return None


def _wheel_collision_radius(link):
    """Return the largest cylinder or sphere collision radius of a link, or None."""
    radii = [
        float(shape.get('radius'))
        for collision in link.findall('collision')
        for shape in collision.findall('geometry/*')
        if shape.tag in ('cylinder', 'sphere')
    ]
    return max(radii) if radii else None


def _report(header, problems):
    return header + ':\n  ' + '\n  '.join(problems)


@pytest.mark.parametrize('config_id', list(CONFIGURATIONS))
def test_xacro_expands(config_id):
    """The configuration expands without xacro errors."""
    robot = _robot(config_id)
    assert robot.root.tag == 'robot'
    assert robot.link_elements, f'{config_id}: expansion contains no links'


@pytest.mark.parametrize('config_id', list(CONFIGURATIONS))
def test_link_and_joint_names_are_unique(config_id):
    """No two links and no two joints share a name."""
    robot = _robot(config_id)
    for kind, elements in (('link', robot.link_elements), ('joint', robot.joint_elements)):
        names = [element.get('name') for element in elements]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        assert not duplicates, f'{config_id}: duplicate {kind} names {duplicates}'


@pytest.mark.parametrize('config_id', list(CONFIGURATIONS))
def test_kinematic_tree_is_connected(config_id):
    """Joints connect existing links into a single tree with one root link."""
    robot = _robot(config_id)
    problems = []
    parents = {}
    children = {}
    for joint in robot.joint_elements:
        name = joint.get('name')
        parent = joint.find('parent')
        child = joint.find('child')
        if parent is None or child is None:
            problems.append(f'joint {name} lacks a parent or child element')
            continue
        for role, link in (('parent', parent.get('link')), ('child', child.get('link'))):
            if link not in robot.links:
                problems.append(f'joint {name} {role} link {link} does not exist')
        parents.setdefault(child.get('link'), []).append(name)
        children.setdefault(parent.get('link'), []).append(child.get('link'))
    for link, joints in sorted(parents.items()):
        if len(joints) > 1:
            problems.append(f'link {link} is the child of several joints {joints}')
    roots = sorted(name for name in robot.links if name not in parents)
    if len(roots) == 1:
        reachable = set(roots)
        frontier = list(roots)
        while frontier:
            for child in children.get(frontier.pop(), []):
                if child not in reachable:
                    reachable.add(child)
                    frontier.append(child)
        unreachable = sorted(set(robot.links) - reachable)
        if unreachable:
            problems.append(f'links not connected to root {roots[0]}: {unreachable}')
    else:
        problems.append(f'expected exactly one root link, found {roots}')
    assert not problems, _report(config_id, problems)


@pytest.mark.parametrize('config_id', list(CONFIGURATIONS))
def test_mesh_files_exist(config_id):
    """Every mesh URI resolves to an installed file."""
    robot = _robot(config_id)
    problems = set()
    for mesh in robot.root.iter('mesh'):
        uri = mesh.get('filename', '')
        path = _resolve_uri(uri)
        if path is None:
            problems.add(f'{uri!r} is not a resolvable package:// or file:// URI')
        elif not os.path.isfile(path):
            problems.add(f'{uri} -> {path} does not exist')
    assert not problems, _report(config_id, sorted(problems))


@pytest.mark.parametrize('config_id', list(CONFIGURATIONS))
def test_no_unresolved_xacro_expressions(config_id):
    """No ${...} or $(...) survives expansion in attribute values or element text."""
    robot = _robot(config_id)
    problems = []
    for element in robot.root.iter():
        for attribute, value in element.attrib.items():
            if UNRESOLVED_XACRO.search(value):
                problems.append(f'<{element.tag} {attribute}="{value}">')
        for text in (element.text, element.tail):
            if text and UNRESOLVED_XACRO.search(text):
                problems.append(f'<{element.tag}> text {text.strip()!r}')
    assert not problems, _report(config_id, problems)


@pytest.mark.parametrize('config_id', list(CONFIGURATIONS))
def test_gazebo_references_exist(config_id):
    """Every <gazebo reference> names an existing link or joint."""
    robot = _robot(config_id)
    references = {gazebo.get('reference') for gazebo in robot.root.findall('gazebo')}
    missing = sorted(
        reference for reference in references - {None}
        if reference not in robot.links and reference not in robot.joints)
    assert not missing, f'{config_id}: <gazebo reference> names unknown frames {missing}'


@pytest.mark.parametrize('config_id', list(CONFIGURATIONS))
def test_ros2_control_joints_are_movable_urdf_joints(config_id):
    """Every ros2_control joint is a non-fixed URDF joint owned by exactly one block."""
    robot = _robot(config_id)
    problems = []
    owners = {}
    for block in robot.root.findall('ros2_control'):
        for joint in block.findall('joint'):
            owners.setdefault(joint.get('name'), []).append(block.get('name'))
    for name, blocks in sorted(owners.items()):
        if len(blocks) > 1:
            problems.append(f'{name} is declared in several ros2_control blocks {blocks}')
        if name not in robot.joints:
            problems.append(f'{name} is not a joint in the URDF')
        elif robot.joints[name].get('type') == 'fixed':
            problems.append(f'{name} is a fixed joint in the URDF')
    assert not problems, _report(config_id, problems)


def test_every_controller_config_is_paired():
    """Each controller parameter file is listed in CONTROLLER_PAIRINGS."""
    directory = os.path.join(_share_dir(), 'config', 'ros2_controller')
    present = {name for name in os.listdir(directory) if name.endswith('.yaml')}
    unpaired = sorted(present - set(CONTROLLER_FILES))
    missing = sorted(set(CONTROLLER_FILES) - present)
    assert not unpaired, f'add a CONTROLLER_PAIRINGS entry for {unpaired}'
    assert not missing, f'CONTROLLER_PAIRINGS names files that do not exist: {missing}'


@pytest.mark.parametrize(
    ('config_id', 'controllers_file'),
    [
        pytest.param(
            config, name,
            id=f'{config}-{os.path.splitext(name)[0]}',
            marks=[pytest.mark.xfail(strict=True, reason=KNOWN_PAIRING_MISMATCHES[config, name])]
            if (config, name) in KNOWN_PAIRING_MISMATCHES else [],
        )
        for config, name in CONTROLLER_PAIRINGS
    ],
)
def test_controller_config_matches_ros2_control(config_id, controllers_file):
    """Controllers only claim joints, interfaces and frames the paired mode provides."""
    robot = _robot(config_id)
    declared = robot.hardware_interfaces()
    declared_sensors = robot.sensor_interfaces()
    problems = []
    for name, controller_type, params in _controllers(controllers_file):
        claims = _claimed_interfaces(controller_type, params)
        if claims is None:
            problems.append(
                f'{name}: controller type {controller_type} is not modelled; '
                'teach _claimed_interfaces which interfaces it claims')
            continue
        for joint in sorted(_referenced_joints(params, claims), key=str):
            if joint not in robot.joints:
                problems.append(f'{name}: {joint} is not a joint in the URDF')
            elif joint not in declared:
                problems.append(f'{name}: {joint} is not declared in any ros2_control block')
        for joint, kind, interface in claims:
            if joint in declared and interface not in declared[joint][kind]:
                problems.append(
                    f'{name}: claims {kind} interface {joint}/{interface}, which is not '
                    f'declared (declared: {sorted(declared[joint][kind])})')
        for sensor, interface in _claimed_sensor_interfaces(controller_type, params):
            if sensor not in declared_sensors:
                problems.append(f'{name}: sensor {sensor!r} is not declared in any ros2_control '
                                'block')
                break
            if interface not in declared_sensors[sensor]:
                problems.append(
                    f'{name}: claims state interface {sensor}/{interface}, which is not declared')
        for parameter, frame in _frame_parameters(params):
            if frame not in robot.links:
                problems.append(f'{name}: {parameter} {frame!r} is not a URDF link')
    assert not problems, _report(
        f'{controllers_file} against ros2_control:={config_id}', problems)


@pytest.mark.parametrize('controllers_file', CONTROLLER_FILES)
def test_omni_wheel_geometry_matches_urdf(controllers_file):
    """Omni drive kinematic parameters match the wheel placement and size in the URDF."""
    omni_controllers = [
        (name, params)
        for name, controller_type, params in _controllers(controllers_file)
        if controller_type == OMNI_CONTROLLER
    ]
    if not omni_controllers:
        pytest.skip(f'{controllers_file} configures no omni_wheel_drive_controller')
    # Wheel geometry does not depend on the mode; use the mode this file is loaded with.
    config_id = next(
        config for config, name in CONTROLLER_PAIRINGS if name == controllers_file)
    robot = _robot(config_id)
    problems = []
    for name, params in omni_controllers:
        base_frame = params.get('base_frame_id', 'base_link')
        wheels = params.get('wheel_names') or []
        required = [key for key in ('robot_radius', 'wheel_radius', 'wheel_offset')
                    if key not in params]
        if required:
            problems.append(f'{name}: missing parameters {required}')
            continue
        if base_frame not in robot.links:
            problems.append(f'{name}: base_frame_id {base_frame!r} is not a URDF link')
            continue
        if not wheels:
            problems.append(f'{name}: wheel_names is empty')
            continue
        base_inverse = _inverse(robot.link_pose(base_frame))
        for index, wheel in enumerate(wheels):
            if wheel not in robot.joints:
                problems.append(f'{name}: wheel {wheel} is not a URDF joint')
                continue
            x, y, _ = _compose(base_inverse, robot.joint_pose(wheel))[1]
            distance = math.hypot(x, y)
            if abs(distance - params['robot_radius']) > ROBOT_RADIUS_TOLERANCE:
                problems.append(
                    f'{name}: robot_radius {params["robot_radius"]} differs from the '
                    f'{wheel} distance {distance:.6f} m in {base_frame}')
            # Wheels are listed anti-clockwise, starting wheel_offset from the x axis.
            angle = math.atan2(y, x)
            expected = params['wheel_offset'] + index * 2.0 * math.pi / len(wheels)
            if abs(math.remainder(angle - expected, 2.0 * math.pi)) > WHEEL_ANGLE_TOLERANCE:
                problems.append(
                    f'{name}: {wheel} sits at {math.degrees(angle):.2f} deg in {base_frame}, '
                    f'wheel_names order and wheel_offset place it at '
                    f'{math.degrees(expected):.2f} deg')
            child = robot.joints[wheel].find('child').get('link')
            radius = _wheel_collision_radius(robot.links[child])
            if radius is None:
                problems.append(f'{name}: {child} has no cylinder or sphere collision')
            elif abs(radius - params['wheel_radius']) > WHEEL_RADIUS_TOLERANCE:
                problems.append(
                    f'{name}: wheel_radius {params["wheel_radius"]} differs from the '
                    f'{child} collision radius {radius} m')
    assert not problems, _report(controllers_file, problems)


def test_mujoco_ros2_control_interfaces():
    """In MuJoCo mode wheels are velocity-only and other joints are position-only."""
    robot = _robot('mujoco')
    declared = robot.hardware_interfaces()
    assert declared, 'ros2_control:=mujoco declares no ros2_control joints'
    problems = [f'{wheel} is not declared' for wheel in WHEEL_JOINTS if wheel not in declared]
    for name, interfaces in sorted(declared.items()):
        joint = robot.joints.get(name)
        if joint is not None and joint.find('mimic') is not None:
            problems.append(f'{name} is a mimic joint and must not be declared')
            continue
        if name in WHEEL_JOINTS:
            command, allowed_state = {'velocity'}, MUJOCO_WHEEL_STATE
        else:
            command, allowed_state = {'position'}, MUJOCO_JOINT_STATE
        if interfaces['command'] != command:
            problems.append(
                f'{name}: command interfaces {sorted(interfaces["command"])}, '
                f'expected {sorted(command)}')
        state = interfaces['state']
        if not MUJOCO_REQUIRED_STATE <= state <= allowed_state:
            problems.append(
                f'{name}: state interfaces {sorted(state)}, expected '
                f'{sorted(MUJOCO_REQUIRED_STATE)} and nothing outside {sorted(allowed_state)}')
    assert not problems, _report('ros2_control:=mujoco', problems)


def test_mujoco_hardware_plugin():
    """The MuJoCo ros2_control block loads the mujoco_ros2_control system plugin."""
    robot = _robot('mujoco')
    plugins = [
        (plugin.text or '').strip()
        for plugin in robot.root.iterfind('ros2_control/hardware/plugin')
    ]
    assert plugins == [MUJOCO_HARDWARE_PLUGIN], (
        f'expected one hardware plugin {MUJOCO_HARDWARE_PLUGIN}, found {plugins}')


def test_mujoco_model_parameter_points_to_installed_scene():
    """The MuJoCo hardware loads mujoco/scene.xml from the installed share directory."""
    robot = _robot('mujoco')
    values = [
        os.path.normpath((param.text or '').strip())
        for param in robot.root.iterfind('ros2_control/hardware/param')
        if param.get('name') == 'mujoco_model'
    ]
    expected = os.path.normpath(os.path.join(_share_dir(), 'mujoco', 'scene.xml'))
    assert values == [expected], (
        f'expected one mujoco_model parameter equal to {expected}, found {values}')
    assert os.path.isfile(expected), f'{expected} is not installed'
