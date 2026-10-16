#!/usr/bin/env python3
"""
Generate the MuJoCo model of walkie (description/mujoco/walkie.xml) from the URDF description.

Pipeline:
  1. Expand robots/gz_walkie.urdf.xacro with ros2_control:=mujoco. The xacro files are always
     resolved from the source tree this script lives in, never from an installed share
     directory, so a stale install cannot leak into the generated model.
  2. Prepare the URDF for MuJoCo's importer: drop <gazebo>, <ros2_control> and <transmission>,
     make mesh paths relative to the package root, and replace visual meshes MuJoCo cannot
     load (COLLADA, oversized STL) with the collision geometry of the same link.
  3. Import with mujoco.MjSpec and add what URDF cannot express: the floating base joint,
     omni-wheel contact capsules, actuators, IMU sensors, cameras, emulated 2D lidars, the head
     3D lidar, collision filtering, self-collision excludes, inertia estimates and a 'home'
     keyframe. The 3D lidar needs the MuJoCo extension of ros-jazzy-mujoco-3d-lidar.
  4. Write the result deterministically with MjSpec.to_xml().

The generated file is checked in. Regenerate it after any change to the robot description:

    pixi run mjcf
    python description/scripts/build_mjcf.py [OUTPUT]

description/mujoco/scene.xml includes the generated model and adds the floor and the
wheel-floor contact pairs; see that file for the requirements on custom worlds.
"""

import argparse
import contextlib
import copy
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from xml.etree import ElementTree

import mujoco
import numpy as np
import xacro


PACKAGE = 'walkie_description'
DESCRIPTION_DIR = Path(__file__).resolve().parent.parent
XACRO_FILE = DESCRIPTION_DIR / 'robots' / 'gz_walkie.urdf.xacro'
XACRO_MAPPINGS = {'ros2_control': 'mujoco'}
MUJOCO_DIR = DESCRIPTION_DIR / 'mujoco'
DEFAULT_OUTPUT = MUJOCO_DIR / 'walkie.xml'
# Mesh directory written into the model, relative to MUJOCO_DIR. The same layout exists in the
# installed share directory, so the model loads from either location.
RELATIVE_MESHDIR = '..'
PACKAGE_URI_PREFIX = f'package://{PACKAGE}/'

MODEL_NAME = 'walkie'
ROOT_LINK = 'base_footprint'
FREE_JOINT = 'floating_base_joint'
LIFT_JOINT = 'lift_joint'
MUJOCO_HARDWARE_PLUGIN = 'mujoco_ros2_control/MujocoSystemInterface'

# ----------------------------------------------------------------------------------------------
# Simulation options
# ----------------------------------------------------------------------------------------------
TIMESTEP = 0.002
INTEGRATOR = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
CONE = mujoco.mjtCone.mjCONE_ELLIPTIC

# ----------------------------------------------------------------------------------------------
# Collision filtering. Every collision geom carries exactly one class bit as contype and the
# union of the classes it may touch as conaffinity. The matrix is symmetric, so two geoms
# collide exactly when their classes are paired below. Wheels are not in any class: their
# capsules have contype = conaffinity = 0 and touch the floor only through explicit <pair>
# elements in the world file.
# ----------------------------------------------------------------------------------------------
FLOOR = 1 << 0  # The plane named 'floor' of the world.
BASE = 1 << 1   # Mobile base and everything rigidly attached to it.
BODY = 1 << 2   # Lift, head and arm mounts.
ARM = 1 << 3    # Arm links 1 to 7.
GRIP = 1 << 4   # Hands, fingers and wrist camera brackets.
OBJ = 1 << 5    # Movable world objects (graspable items).
FURN = 1 << 6   # Static world geometry other than the floor (furniture, walls).

COLLISION_CLASSES = {
    'FLOOR': FLOOR, 'BASE': BASE, 'BODY': BODY, 'ARM': ARM, 'GRIP': GRIP, 'OBJ': OBJ,
    'FURN': FURN,
}
COLLIDES_WITH = {
    FLOOR: OBJ,
    BASE: ARM | GRIP,
    BODY: ARM | GRIP,
    ARM: BASE | BODY | ARM | GRIP | OBJ | FURN,
    GRIP: BASE | BODY | ARM | GRIP | OBJ | FURN,
    OBJ: FLOOR | FURN | OBJ | ARM | GRIP,
    FURN: OBJ | ARM | GRIP,
}

# Collision class of each link with collision geometry (full-match regular expressions).
# Generation fails for a link with collision geometry that matches none of them.
LINK_COLLISION_CLASSES = (
    (r'base_link|bottom_link', BASE),
    (r'lift_link|head_servo_link|zed_head_camera_center|openarm_(left|right)_link0', BODY),
    (r'openarm_(left|right)_link[1-7]', ARM),
    (r'openarm_(left|right)_(hand|left_finger|right_finger|wristcam_bracket)', GRIP),
)

# Geom and site groups (MuJoCo shows groups 0 to 2 by default, cameras render the same set).
VISUAL_GROUP = 2
COLLISION_GROUP = 3
FRAME_SITE_GROUP = 4
LIDAR_SITE_GROUP = 5
FRAME_SITE_SIZE = 0.01
LIDAR_SITE_SIZE = 0.002

# ----------------------------------------------------------------------------------------------
# Omni wheels. Each wheel touches the floor through a thin capsule whose axis is the axle:
# MuJoCo aligns the first tangent of a capsule-plane contact with the capsule axis, which is
# what makes the anisotropic friction of the wheel-floor pairs in scene.xml work.
# ----------------------------------------------------------------------------------------------
WHEELS = ('fl', 'fr', 'bl', 'br')
WHEEL_CAPSULE_HALF_LENGTH = 0.005
WHEEL_JOINT_DAMPING = 0.001
WHEEL_JOINT_ARMATURE = 0.01
# Velocity actuator gain, N m per rad/s of wheel speed error.
WHEEL_KV = 20.0


def wheel_joint(wheel):
    return f'{wheel}_wheel_joint'


def wheel_link(wheel):
    return f'{wheel}_wheel_link'


def wheel_contact_geom(wheel):
    """Name of the capsule a world <pair> must reference for this wheel."""
    return f'{wheel}_wheel_contact'


# ----------------------------------------------------------------------------------------------
# Position actuators. The stiffness is chosen so the actuator reaches the URDF effort limit at
# POSITION_SATURATION_ERROR of tracking error; damping is critical (dampratio 1).
# ----------------------------------------------------------------------------------------------
POSITION_SATURATION_ERROR = 0.05  # rad for revolute joints, m for prismatic joints
POSITION_DAMPRATIO = 1.0
# Stiffness for joints whose URDF effort limit is zero (unspecified), e.g. the head servo.
POSITION_KP_FALLBACK = 20.0
# Reflected rotor inertia added to every position-controlled joint; it keeps the light distal
# links stable at TIMESTEP with the stiffness above.
POSITION_JOINT_ARMATURE = 0.01

# Lift drive. The URDF effort limit of lift_joint (10000 N) is a placeholder: an actuator
# derived from it throws the robot off the floor on a setpoint step and crushes the arms into
# the base when the lift is lowered onto them. The values below are ESTIMATES PENDING the lift
# drive data and replace the derived actuator of LIFT_JOINT only:
#   - the force limit is about four times the static load of the lift with both arms (154 N),
#   - the stiffness is independent of the force limit and keeps the static sag near 3 mm,
#   - the joint damping bounds the lift speed: the force limit over the damping is the URDF
#     velocity limit of lift_joint (0.1 m/s), so steps move the lift at a drive-like speed.
LIFT_FORCE_LIMIT_ESTIMATE = 600.0     # N
LIFT_POSITION_KP_ESTIMATE = 5.0e4     # N/m
LIFT_JOINT_DAMPING_ESTIMATE = 6000.0  # N s/m

# ----------------------------------------------------------------------------------------------
# Inertia estimates. The URDF inertials of these links are placeholders (base_link has a
# negligible inertia tensor at the floor, lift_link has none). The values below are
# ESTIMATES PENDING CAD and only replace the placeholders in the MuJoCo model; the URDF is
# left untouched.
# ----------------------------------------------------------------------------------------------
# base_link: URDF mass as a solid cylinder matching its collision cylinder (base_link frame).
BASE_LINK_INERTIA_ESTIMATE_RADIUS = 0.275
BASE_LINK_INERTIA_ESTIMATE_HEIGHT = 0.33
BASE_LINK_INERTIA_ESTIMATE_CENTER = (0.0, 0.0, 0.165)
# lift_link: estimated mass as a solid box matching its 'top_bar' collision box.
LIFT_LINK_MASS_ESTIMATE = 3.0
LIFT_LINK_INERTIA_ESTIMATE_SIZE = (0.346, 0.174, 0.132)
LIFT_LINK_INERTIA_ESTIMATE_CENTER = (0.0, 0.0, 0.109)

# ----------------------------------------------------------------------------------------------
# Sensors
# ----------------------------------------------------------------------------------------------
IMU_LINK = 'top_imu_link'
IMU_NAME = 'top_imu'

# Cameras sit at the URDF optical frames. A MuJoCo camera looks along -Z with +Y up while a
# ROS optical frame looks along +Z with +Y down: a rotation of pi about X maps one onto the
# other. Field of view and resolution come from the named Gazebo sensor.
OPTICAL_TO_MUJOCO_CAMERA_QUAT = (0.0, 1.0, 0.0, 0.0)
CAMERAS = (
    # (MJCF camera name, optical frame link, Gazebo sensor providing fov and resolution)
    ('head_camera', 'zed_head_left_camera_frame_optical', 'head_rgbd_sensor'),
    ('bottom_camera', 'bottom_color_optical_frame', 'bottom_depth_sensor'),
)

# 2D lidars are emulated with one site-based rangefinder per beam, compatible with the
# RangefinderLidarPlugin of mujoco_ros2_control_plugins (sensor names '<lidar>-<index>').
# Beams run from the URDF min_angle to max_angle inclusive, with the beam count whose spacing
# is closest to LIDAR_RESOLUTION_DEG, so min_angle + (count - 1) * increment == max_angle as a
# LaserScan consumer expects. The beam sites live on LIDAR_RAY_BODY, the body that owns the
# base collision geometry around the scanners, because a rangefinder ignores the geoms of its
# own body.
LIDAR_RESOLUTION_DEG = 1.0  # Nominal beam spacing.
# Digits of the angle increment written to the model header. The value is rounded down, so a
# consumer computing int((max_angle - min_angle) / increment) + 1 gets the generated count.
LIDAR_INCREMENT_DIGITS = 12
LIDAR_RAY_BODY = 'base_link'
LIDAR_SENSOR_TYPES = ('gpu_lidar', 'lidar', 'gpu_ray', 'ray')
# MuJoCo draws every rangefinder beam that hits something, in the viewer and in camera images.
# A transparent colour keeps the beams out of rendered colour images; depth images still
# contain them unless the renderer disables mjVIS_RANGEFINDER.
RANGEFINDER_RGBA = (1.0, 1.0, 0.1, 0.0)

# The head 3D lidar (Unitree 4D L2) is one sensor of the MuJoCo extension 'mujoco.plugin.lidar'
# (conda package ros-jazzy-mujoco-3d-lidar), published by Mujoco3dLidarPlugin of
# mujoco_ros2_control_plugins. It sits on the URDF frame the real driver stamps its cloud with.
# The L2 sees the hemisphere z >= 0 of that frame (360 x 90 deg) from 0.05 m to 30 m and sends
# 12 clouds per second (216 scan lines per second, 18 lines per cloud). The extension casts a
# fixed grid of rays, azimuth about the Z axis of the site and elevation from its XY plane. The
# rays sit at the centres of an even grid of cells covering the hemisphere, so no ray repeats at
# the azimuth seam or at the pole. Every ray is cast synchronously in mj_step and the extension
# cannot exclude geoms (it also hits the robot), so the resolution trades density for
# simulation speed.
LIDAR_3D_NAME = 'unitree_l2'  # MJCF plugin instance and sensor name.
LIDAR_3D_SITE = 'unitree4d_l2_imu_initial'
LIDAR_3D_PLUGIN = 'mujoco.plugin.lidar'
LIDAR_3D_PACKAGE = 'ros-jazzy-mujoco-3d-lidar'
LIDAR_3D_RESOLUTION = (180, 32)  # Rays in azimuth and elevation (2 x 2.8 deg).
LIDAR_3D_RANGE = (0.05, 30.0)    # m
LIDAR_3D_UPDATE_RATE = 12.0      # Hz, in simulated time.

# MuJoCo's STL decoder only reads binary files with at most this many faces.
STL_MAX_FACES = 200000
MESH_CONTENT_TYPES = {'.stl': 'model/stl', '.obj': 'model/obj', '.msh': 'model/vnd.mujoco.msh'}
LOADABLE_MESH_SUFFIXES = tuple(MESH_CONTENT_TYPES)

KEYFRAME_NAME = 'home'

HEADER_TEMPLATE = """
  GENERATED FILE - DO NOT EDIT BY HAND.

  Generated by description/scripts/build_mjcf.py from robots/gz_walkie.urdf.xacro
  (ros2_control:=mujoco). Regenerate with 'pixi run mjcf' after changing the description.

  Load description/mujoco/scene.xml (or a world that includes this file); this file has no
  floor, and the wheels touch the floor only through the <pair> elements of the world.

  Collision classes (contype = class bit, conaffinity = classes it collides with):
{collision_table}
  Wheel capsules {wheel_geoms} have contype = conaffinity = 0.

  Lidars (rangefinder sensors '<lidar>-<index>', beam 0 at min_angle and the last beam at
  max_angle, -1 when nothing is hit; beams are drawn transparent so they stay out of camera
  images):
{lidar_table}
  Head 3D lidar (sensor of the MuJoCo extension {lidar_3d_plugin}; azimuth about the site Z
  axis, elevation from its XY plane, -1 when nothing is hit; the rays also hit the robot):
{lidar_3d_row}
  Cameras (MJCF camera -> optical frame): {camera_table}
  IMU: site {imu}, sensors {imu}_quat, {imu}_gyro, {imu}_accel.

  Estimates pending CAD or drive data (the *_ESTIMATE constants of build_mjcf.py):
    base_link inertia; lift_link mass ({lift_mass:g} kg) and inertia;
    lift_joint actuator force limit {lift_force:g} N and stiffness {lift_kp:g} N/m;
    lift_joint damping {lift_damping:g} N s/m.

  Expanded with right_joint2_fixed:=false. With right_joint2_fixed:=True the ros2_control block
  omits openarm_right_joint2; its actuator then keeps control 0, which holds the joint at the
  pose of the fixed URDF joint.
"""


# ----------------------------------------------------------------------------------------------
# Small geometry helpers (quaternions are w, x, y, z)
# ----------------------------------------------------------------------------------------------
def _axis_angle_quat(axis, angle):
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, np.asarray(axis, dtype=float), angle)
    return quat


def _quat_mul(first, second):
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    return result


def _rotate(quat, vector):
    result = np.zeros(3)
    mujoco.mju_rotVecQuat(result, np.asarray(vector, dtype=float), np.asarray(quat, dtype=float))
    return result


def _compose(first, second):
    """Pose of 'second' (given relative to 'first') in the frame 'first' is expressed in."""
    pos_a, quat_a = first
    pos_b, quat_b = second
    return pos_a + _rotate(quat_a, pos_b), _quat_mul(quat_a, quat_b)


def _rounded(values, digits=12):
    """Round computed values so the written model does not depend on the last float bits."""
    return [round(float(value), digits) + 0.0 for value in values]


# ----------------------------------------------------------------------------------------------
# URDF expansion and inspection
# ----------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _source_tree_package_index():
    """
    Make $(find walkie_description) resolve to the source tree of this script.

    A throw-away ament prefix containing only walkie_description (a symlink to the source
    directory) is prepended to AMENT_PREFIX_PATH for the duration of the expansion.
    """
    previous = os.environ.get('AMENT_PREFIX_PATH')
    with tempfile.TemporaryDirectory(prefix='walkie_mjcf_') as prefix:
        index = Path(prefix, 'share', 'ament_index', 'resource_index', 'packages')
        index.mkdir(parents=True)
        (index / PACKAGE).touch()
        os.symlink(DESCRIPTION_DIR, Path(prefix, 'share', PACKAGE))
        os.environ['AMENT_PREFIX_PATH'] = os.pathsep.join(filter(None, [prefix, previous]))
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop('AMENT_PREFIX_PATH', None)
            else:
                os.environ['AMENT_PREFIX_PATH'] = previous


def expand_urdf(mappings=None):
    """Expand the robot xacro (ros2_control:=mujoco by default) and return the URDF root."""
    with _source_tree_package_index():
        document = xacro.process_file(str(XACRO_FILE), mappings=dict(mappings or XACRO_MAPPINGS))
        return ElementTree.fromstring(document.toxml())


class UrdfModel:
    """Read-only view of the parts of the expanded URDF the generator needs."""

    def __init__(self, root):
        self.root = root
        self.links = {link.get('name'): link for link in root.findall('link')}
        self.joints = {joint.get('name'): joint for joint in root.findall('joint')}

    def movable_joints(self):
        """Non-fixed, non-mimic joints in document order."""
        return [joint for joint in self.root.findall('joint')
                if joint.get('type') != 'fixed' and joint.find('mimic') is None]

    def gazebo_sensors(self):
        """Yield (reference link, <sensor> element) for every Gazebo sensor."""
        for gazebo in self.root.findall('gazebo'):
            for sensor in gazebo.iter('sensor'):
                yield gazebo.get('reference'), sensor

    def gazebo_sensor(self, name):
        for reference, sensor in self.gazebo_sensors():
            if sensor.get('name') == name:
                return reference, sensor
        raise ValueError(f"Gazebo sensor '{name}' not found in the URDF")

    def mujoco_ros2_control(self):
        """Return the <ros2_control> element using the MuJoCo hardware plugin, or None."""
        for element in self.root.findall('ros2_control'):
            if (element.findtext('hardware/plugin') or '').strip() == MUJOCO_HARDWARE_PLUGIN:
                return element
        return None


def ros2_control_joints(ros2_control):
    """{joint: (command interfaces, initial position or None)} of a <ros2_control> block."""
    joints = {}
    for joint in ros2_control.findall('joint'):
        commands = [interface.get('name') for interface in joint.findall('command_interface')]
        initial = None
        for state in joint.findall('state_interface'):
            if state.get('name') != 'position':
                continue
            for param in state.findall('param'):
                if param.get('name') == 'initial_value':
                    initial = float(param.text)
        joints[joint.get('name')] = (commands, initial)
    return joints


def _lidar_parameters(sensor):
    """Scan parameters of a Gazebo lidar <sensor> element."""
    horizontal = sensor.find('.//scan/horizontal')
    scan_range = sensor.find('.//range')
    return {
        'update_rate': float(sensor.findtext('update_rate')),
        'min_angle': float(horizontal.findtext('min_angle')),
        'max_angle': float(horizontal.findtext('max_angle')),
        'range_min': float(scan_range.findtext('min')),
        'range_max': float(scan_range.findtext('max')),
    }


def lidar_angle_increment(min_angle, max_angle):
    """Beam spacing of lidar_beam_angles(): the even spacing closest to LIDAR_RESOLUTION_DEG."""
    intervals = max(1, round((max_angle - min_angle) / math.radians(LIDAR_RESOLUTION_DEG)))
    return (max_angle - min_angle) / intervals


def lidar_beam_angles(min_angle, max_angle):
    """Beam angles in the lidar frame, from min_angle to max_angle inclusive."""
    increment = lidar_angle_increment(min_angle, max_angle)
    count = round((max_angle - min_angle) / increment) + 1
    return [min_angle + index * increment for index in range(count)]


def lidar_3d_angle_ranges():
    """(azimuth, elevation) ranges of the head 3D lidar: first and last ray, inclusive."""
    azimuth_count, elevation_count = LIDAR_3D_RESOLUTION
    azimuth_step = 2 * math.pi / azimuth_count
    elevation_step = math.pi / 2 / elevation_count
    return ((-math.pi + azimuth_step / 2, math.pi - azimuth_step / 2),
            (elevation_step / 2, math.pi / 2 - elevation_step / 2))


def lidar_3d_config():
    """Plugin configuration of the head 3D lidar (synchronous: the 'async' key is unset)."""
    azimuth, elevation = lidar_3d_angle_ranges()
    return {
        'resolution': ' '.join(str(count) for count in LIDAR_3D_RESOLUTION),
        'azimuth_range': ' '.join(f'{angle:.9f}' for angle in azimuth),
        'elevation_range': ' '.join(f'{angle:.9f}' for angle in elevation),
        'min_range': f'{LIDAR_3D_RANGE[0]:g}',
        'max_range': f'{LIDAR_3D_RANGE[1]:g}',
        'update_rate': f'{LIDAR_3D_UPDATE_RATE:g}',
    }


# ----------------------------------------------------------------------------------------------
# URDF preparation for MuJoCo's importer
# ----------------------------------------------------------------------------------------------
def _mesh_is_loadable(relative_path):
    path = DESCRIPTION_DIR / relative_path
    if path.suffix.lower() not in LOADABLE_MESH_SUFFIXES:
        return False
    if path.suffix.lower() != '.stl':
        return True
    with open(path, 'rb') as stream:
        header = stream.read(84)
    if len(header) < 84:
        return False
    faces = int.from_bytes(header[80:84], 'little')
    # ASCII STL files fail the size check of the binary layout.
    return path.stat().st_size == 84 + 50 * faces and 0 < faces <= STL_MAX_FACES


def prepare_urdf(root):
    """Return a copy of the URDF ready for mujoco.MjSpec.from_string()."""
    root = copy.deepcopy(root)
    for tag in ('gazebo', 'ros2_control', 'transmission'):
        for element in root.findall(tag):
            root.remove(element)

    for mesh in root.iter('mesh'):
        filename = mesh.get('filename')
        if not filename.startswith(PACKAGE_URI_PREFIX):
            raise ValueError(f'mesh outside {PACKAGE}: {filename}')
        mesh.set('filename', filename[len(PACKAGE_URI_PREFIX):])

    # MuJoCo cannot read COLLADA, ASCII STL or STL files above STL_MAX_FACES. Such visuals are
    # replaced by the collision geometry of the same link, so no extra mesh files are needed.
    # Negative mesh scales (mirrored arm meshes) are supported by MuJoCo as they are.
    for link in root.findall('link'):
        collision = link.find('collision')
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None or _mesh_is_loadable(mesh.get('filename')):
                continue
            if collision is None:
                raise ValueError(f"link '{link.get('name')}': unloadable visual mesh "
                                 f"'{mesh.get('filename')}' and no collision geometry")
            for child in ('origin', 'geometry'):
                if visual.find(child) is not None:
                    visual.remove(visual.find(child))
            if collision.find('origin') is not None:
                visual.insert(0, copy.deepcopy(collision.find('origin')))
            visual.append(copy.deepcopy(collision.find('geometry')))

    extension = ElementTree.SubElement(root, 'mujoco')
    ElementTree.SubElement(
        extension, 'compiler', meshdir=str(DESCRIPTION_DIR), strippath='false',
        discardvisual='false', fusestatic='false')
    return ElementTree.tostring(root, encoding='unicode')


# ----------------------------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------------------------
def _link_collision_class(link):
    for pattern, collision_class in LINK_COLLISION_CLASSES:
        if re.fullmatch(pattern, link):
            return collision_class
    return None


def _class_name(collision_class):
    return next(name for name, bit in COLLISION_CLASSES.items() if bit == collision_class)


def _set_inertial(body, mass, center, diagonal):
    body.explicitinertial = True
    body.mass = mass
    body.ipos = list(center)
    body.iquat = [1, 0, 0, 0]
    body.inertia = _rounded(diagonal)
    body.fullinertia = [math.nan, 0, 0, 0, 0, 0]  # Unset, the diagonal above applies.


def _set_inertia_estimates(spec):
    base = spec.body('base_link')
    mass = base.mass  # URDF mass is kept, only its distribution is estimated.
    radius = BASE_LINK_INERTIA_ESTIMATE_RADIUS
    height = BASE_LINK_INERTIA_ESTIMATE_HEIGHT
    radial = mass * (3 * radius ** 2 + height ** 2) / 12
    _set_inertial(base, mass, BASE_LINK_INERTIA_ESTIMATE_CENTER,
                  [radial, radial, mass * radius ** 2 / 2])

    size_x, size_y, size_z = LIFT_LINK_INERTIA_ESTIMATE_SIZE
    mass = LIFT_LINK_MASS_ESTIMATE
    _set_inertial(spec.body('lift_link'), mass, LIFT_LINK_INERTIA_ESTIMATE_CENTER,
                  [mass * (size_y ** 2 + size_z ** 2) / 12,
                   mass * (size_x ** 2 + size_z ** 2) / 12,
                   mass * (size_x ** 2 + size_y ** 2) / 12])


def _configure_geoms(spec):
    """Name every geom, assign groups and collision classes, turn wheels into capsules."""
    wheel_links = {wheel_link(wheel): wheel for wheel in WHEELS}
    for body in spec.bodies:
        if body.name == 'world':
            continue
        # The URDF importer marks visual geoms with contype = conaffinity = 0.
        visuals = [geom for geom in body.geoms if geom.contype == 0 and geom.conaffinity == 0]
        collisions = [geom for geom in body.geoms if geom not in visuals]
        for kind, geoms in (('visual', visuals), ('collision', collisions)):
            for index, geom in enumerate(geoms):
                if not geom.name:
                    suffix = f'_{index}' if len(geoms) > 1 else ''
                    geom.name = f'{body.name}_{kind}{suffix}'
        for geom in visuals:
            geom.group = VISUAL_GROUP
        if not body.explicitinertial:
            # A URDF link without <inertial> is massless. MJCF infers the inertia of such a
            # body from its geoms (inertiafromgeom="auto"), so they are given zero density.
            for geom in body.geoms:
                geom.density = 0
        if body.name in wheel_links:
            if len(collisions) != 1 or collisions[0].type != mujoco.mjtGeom.mjGEOM_CYLINDER:
                raise ValueError(f"wheel link '{body.name}' needs one collision cylinder")
            capsule = collisions[0]
            capsule.name = wheel_contact_geom(wheel_links[body.name])
            capsule.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            capsule.size = [capsule.size[0], WHEEL_CAPSULE_HALF_LENGTH, 0]
            capsule.contype = 0
            capsule.conaffinity = 0
            capsule.group = COLLISION_GROUP
            continue
        if not collisions:
            continue
        collision_class = _link_collision_class(body.name)
        if collision_class is None:
            raise ValueError(f"link '{body.name}' has collision geometry but no collision "
                             'class; add it to LINK_COLLISION_CLASSES')
        for geom in collisions:
            geom.contype = collision_class
            geom.conaffinity = COLLIDES_WITH[collision_class]
            geom.group = COLLISION_GROUP


def _configure_joints(spec, urdf):
    for wheel in WHEELS:
        joint = spec.joint(wheel_joint(wheel))
        # Linear damping coefficient only (MjSpec stores a damping polynomial).
        joint.damping = [WHEEL_JOINT_DAMPING, 0, 0]
        joint.frictionloss = 0
        joint.armature = WHEEL_JOINT_ARMATURE
    wheel_joints = {wheel_joint(wheel) for wheel in WHEELS}
    for urdf_joint in urdf.movable_joints():
        if urdf_joint.get('name') not in wheel_joints:
            spec.joint(urdf_joint.get('name')).armature = POSITION_JOINT_ARMATURE
    lift = spec.joint(LIFT_JOINT)
    lift.damping = [LIFT_JOINT_DAMPING_ESTIMATE, 0, 0]
    # The importer copies the placeholder URDF effort limit here; keep it equal to the actuator.
    lift.actfrcrange = [-LIFT_FORCE_LIMIT_ESTIMATE, LIFT_FORCE_LIMIT_ESTIMATE]
    for equality in spec.equalities:
        if equality.type == mujoco.mjtEq.mjEQ_JOINT and not equality.name:
            # URDF <mimic> joints are converted to joint equalities (follower = name1).
            equality.name = f'{equality.name1}_mimic'


def _add_actuators(spec, urdf):
    """One actuator per movable non-mimic joint, named after the joint."""
    wheel_joints = {wheel_joint(wheel) for wheel in WHEELS}
    for urdf_joint in urdf.movable_joints():
        name = urdf_joint.get('name')
        limit = urdf_joint.find('limit')
        effort = float(limit.get('effort', 0)) if limit is not None else 0.0
        actuator = spec.add_actuator(name=name, target=name,
                                     trntype=mujoco.mjtTrn.mjTRN_JOINT)
        if name in wheel_joints:
            velocity = float(limit.get('velocity', 0)) if limit is not None else 0.0
            actuator.set_to_velocity(kv=WHEEL_KV)
            if velocity > 0:
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                actuator.ctrlrange = [-velocity, velocity]
        else:
            if name == LIFT_JOINT:
                kp, effort = LIFT_POSITION_KP_ESTIMATE, LIFT_FORCE_LIMIT_ESTIMATE
            elif effort > 0:
                kp = effort / POSITION_SATURATION_ERROR
            else:
                kp = POSITION_KP_FALLBACK
            actuator.set_to_position(kp=kp, dampratio=POSITION_DAMPRATIO)
            if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                actuator.ctrlrange = [float(limit.get('lower')), float(limit.get('upper'))]
        if effort > 0:
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange = [-effort, effort]


def _add_frame_sites(spec, urdf):
    """Keep every pure frame link (no geometry, no inertia) as a site of the same name."""
    for name, link in urdf.links.items():
        if name == ROOT_LINK or link.find('visual') is not None:
            continue
        if link.find('collision') is not None or link.find('inertial') is not None:
            continue
        spec.body(name).add_site(name=name, group=FRAME_SITE_GROUP, size=[FRAME_SITE_SIZE] * 3)


def _add_imu(spec):
    spec.body(IMU_LINK).add_site(name=IMU_NAME, group=FRAME_SITE_GROUP,
                                 size=[FRAME_SITE_SIZE] * 3)
    spec.add_sensor(name=f'{IMU_NAME}_quat', type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=IMU_NAME)
    spec.add_sensor(name=f'{IMU_NAME}_gyro', type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=IMU_NAME)
    spec.add_sensor(name=f'{IMU_NAME}_accel', type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=IMU_NAME)


def _add_cameras(spec, urdf):
    cameras = []
    for name, frame, sensor_name in CAMERAS:
        _, sensor = urdf.gazebo_sensor(sensor_name)
        hfov = float(sensor.findtext('.//camera/horizontal_fov'))
        width = int(sensor.findtext('.//camera/image/width'))
        height = int(sensor.findtext('.//camera/image/height'))
        fovy = math.degrees(2 * math.atan(math.tan(hfov / 2) * height / width))
        spec.body(frame).add_camera(name=name, quat=list(OPTICAL_TO_MUJOCO_CAMERA_QUAT),
                                    fovy=round(fovy, 6), resolution=[width, height])
        cameras.append((name, frame, width, height))
    return cameras


def _relative_body_pose(model, data, body, reference):
    """Pose of 'body' in the frame of 'reference' (both must be rigidly attached)."""
    if model.body(body).weldid[0] != model.body(reference).weldid[0]:
        raise ValueError(f"body '{body}' is not rigidly attached to '{reference}'")
    reference_quat = data.body(reference).xquat
    inverse = np.zeros(4)
    mujoco.mju_negQuat(inverse, reference_quat)
    return (_rotate(inverse, data.body(body).xpos - data.body(reference).xpos),
            _quat_mul(inverse, data.body(body).xquat))


def _add_lidars(spec, urdf):
    # Frame poses are taken from the compiled model so the beams match the lidar frames
    # exactly as MuJoCo imported them.
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    lidars = []
    for reference, sensor in urdf.gazebo_sensors():
        if sensor.get('type') not in LIDAR_SENSOR_TYPES:
            continue
        name = sensor.get('name')
        params = _lidar_parameters(sensor)
        frame_pose = _relative_body_pose(model, data, reference, LIDAR_RAY_BODY)
        # A rangefinder measures along the +Z axis of its site; rotating by +pi/2 about Y maps
        # +Z onto +X of the beam frame.
        beam_to_site = _axis_angle_quat([0, 1, 0], math.pi / 2)
        body = spec.body(LIDAR_RAY_BODY)
        angles = lidar_beam_angles(params['min_angle'], params['max_angle'])
        for index, angle in enumerate(angles):
            pos, quat = _compose(frame_pose, (np.zeros(3), _quat_mul(
                _axis_angle_quat([0, 0, 1], angle), beam_to_site)))
            site = f'{name}_ray_{index:03d}'
            body.add_site(name=site, pos=_rounded(pos), quat=_rounded(quat),
                          group=LIDAR_SITE_GROUP, size=[LIDAR_SITE_SIZE] * 3)
            rangefinder = spec.add_sensor(name=f'{name}-{index}',
                                          type=mujoco.mjtSensor.mjSENS_RANGEFINDER,
                                          objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)
            rangefinder.intprm = [1, 0, 0]  # Report the distance (the MJCF default).
            # Evaluate at the scanner rate only. MuJoCo honours 'interval' for sensors with a
            # history buffer, hence nsample = 1; this cuts the cost of the beams roughly by
            # the ratio of scan period to TIMESTEP.
            rangefinder.interval = [round(1.0 / params['update_rate'], 9), 0]
            rangefinder.nsample = 1
        lidars.append(dict(params, name=name, frame=reference, count=len(angles),
                           increment=lidar_angle_increment(params['min_angle'],
                                                           params['max_angle'])))
    if not lidars:
        raise ValueError('no Gazebo lidar sensors found in the URDF')
    return lidars


def _add_lidar_3d(spec):
    """Add the head 3D lidar as a sensor of the MuJoCo lidar extension on its frame site."""
    # Parsing an <extension> element fails when no loaded library registers the plugin. The
    # mujoco Python package loads every library in <prefix>/bin/mujoco_plugin on import.
    try:
        mujoco.MjSpec.from_string(
            f'<mujoco><extension><plugin plugin="{LIDAR_3D_PLUGIN}"/></extension></mujoco>')
    except ValueError as error:
        raise RuntimeError(f"MuJoCo extension '{LIDAR_3D_PLUGIN}' is not available; install "
                           f'{LIDAR_3D_PACKAGE} (pixi install)') from error
    if spec.site(LIDAR_3D_SITE) is None:
        raise ValueError(f"3D lidar frame '{LIDAR_3D_SITE}' is not a site of the model")
    spec.activate_plugin(LIDAR_3D_PLUGIN)
    plugin = spec.add_plugin(name=LIDAR_3D_NAME, plugin_name=LIDAR_3D_PLUGIN, active=True)
    plugin.config = lidar_3d_config()
    sensor = spec.add_sensor(name=LIDAR_3D_NAME, type=mujoco.mjtSensor.mjSENS_PLUGIN,
                             objtype=mujoco.mjtObj.mjOBJ_SITE, objname=LIDAR_3D_SITE)
    sensor.plugin = plugin


def _set_rest_height(spec):
    """Place the root so the wheel capsules touch the z = 0 plane at qpos0."""
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    lowest = min(data.geom(wheel_contact_geom(wheel)).xpos[2] - model.geom(
        wheel_contact_geom(wheel)).size[0] for wheel in WHEELS)
    root = spec.body(ROOT_LINK)
    root.pos = _rounded([root.pos[0], root.pos[1], root.pos[2] - lowest], 9)


def _clamped_qpos0(model):
    qpos = model.qpos0.copy()
    for joint in range(model.njnt):
        if model.jnt_limited[joint] and model.jnt_type[joint] in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            address = model.jnt_qposadr[joint]
            qpos[address] = np.clip(qpos[address], *model.jnt_range[joint])
    return qpos


def robot_contacts(model, qpos):
    """Sorted (body1, body2) pairs of robot geoms in contact at the given configuration."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    pairs = set()
    for contact in data.contact[:data.ncon]:
        bodies = sorted(model.body(model.geom_bodyid[geom]).name for geom in contact.geom)
        pairs.add(tuple(bodies))
    return sorted(pairs)


def _add_self_collision_excludes(spec):
    """
    Exclude body pairs whose meshes interpenetrate at the zero pose.

    The zero pose is evaluated at both limits of LIFT_JOINT and only pairs touching at both are
    excluded: those overlaps come from the meshes themselves. Contacts that depend on the lift
    height (arms reaching into the base when the lift is low) are real self-collisions and
    stay active.
    """
    model = spec.compile()
    lift = model.joint(LIFT_JOINT)
    pairs = None
    for height in lift.range:
        qpos = _clamped_qpos0(model)
        qpos[lift.qposadr[0]] = height
        contacts = set(robot_contacts(model, qpos))
        pairs = contacts if pairs is None else pairs & contacts
    pairs = sorted(pairs)
    for body1, body2 in pairs:
        spec.add_exclude(bodyname1=body1, bodyname2=body2)
    return pairs


def _add_home_keyframe(spec, urdf):
    """'home' keyframe: ros2_control initial positions, zero elsewhere, matching controls."""
    initial = {}
    ros2_control = urdf.mujoco_ros2_control()
    if ros2_control is not None:
        initial = {name: value for name, (_, value) in ros2_control_joints(ros2_control).items()
                   if value is not None}
    model = spec.compile()
    qpos = _clamped_qpos0(model)
    for name, value in sorted(initial.items()):
        joint = model.joint(name)
        qpos[joint.qposadr[0]] = value
    # Mimic followers start where their equality puts them.
    for equality in spec.equalities:
        if equality.type == mujoco.mjtEq.mjEQ_JOINT:
            leader = model.joint(equality.name2).qposadr[0]
            follower = model.joint(equality.name1).qposadr[0]
            poly = equality.data
            value = qpos[leader] - model.qpos0[leader]
            qpos[follower] = model.qpos0[follower] + poly[0] + poly[1] * value
    ctrl = np.zeros(model.nu)
    for actuator in range(model.nu):
        joint = model.actuator_trnid[actuator, 0]
        if model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_AFFINE and \
                model.actuator_biasprm[actuator, 1] != 0:
            ctrl[actuator] = qpos[model.jnt_qposadr[joint]]
    spec.add_key(name=KEYFRAME_NAME, qpos=_rounded(qpos, 9), ctrl=_rounded(ctrl, 9))


def _header(lidars, cameras, excludes):
    width = max(len(name) for name in COLLISION_CLASSES)
    rows = []
    for name, bit in COLLISION_CLASSES.items():
        partners = '|'.join(other for other, other_bit in COLLISION_CLASSES.items()
                            if COLLIDES_WITH[bit] & other_bit)
        rows.append(f'    {name:<{width}} contype {bit:>3}  conaffinity {COLLIDES_WITH[bit]:>3}'
                    f'  ({partners})')
    scale = 10 ** LIDAR_INCREMENT_DIGITS
    lidar_rows = [
        f"    {lidar['name']}: frame {lidar['frame']}, {lidar['count']} beams, "
        f"min_angle {lidar['min_angle']:.8f}, max_angle {lidar['max_angle']:.8f}, "
        'angle_increment '
        f"{math.floor(lidar['increment'] * scale) / scale:.{LIDAR_INCREMENT_DIGITS}f}, "
        f"range {lidar['range_min']:g} to {lidar['range_max']:g} m, "
        f"{lidar['update_rate']:g} Hz"
        for lidar in lidars]
    lidar_3d_row = (
        f'    {LIDAR_3D_NAME}: site {LIDAR_3D_SITE}, '
        f'{LIDAR_3D_RESOLUTION[0]} x {LIDAR_3D_RESOLUTION[1]} rays, '
        f'range {LIDAR_3D_RANGE[0]:g} to {LIDAR_3D_RANGE[1]:g} m, {LIDAR_3D_UPDATE_RATE:g} Hz')
    camera_rows = ', '.join(f'{name} -> {frame} ({width}x{height})'
                            for name, frame, width, height in cameras)
    del excludes  # Listed in the model itself.
    return HEADER_TEMPLATE.format(
        collision_table='\n'.join(rows),
        wheel_geoms=', '.join(wheel_contact_geom(wheel) for wheel in WHEELS),
        lidar_table='\n'.join(lidar_rows), camera_table=camera_rows, imu=IMU_NAME,
        lidar_3d_plugin=LIDAR_3D_PLUGIN, lidar_3d_row=lidar_3d_row,
        lift_mass=LIFT_LINK_MASS_ESTIMATE, lift_force=LIFT_FORCE_LIMIT_ESTIMATE,
        lift_kp=LIFT_POSITION_KP_ESTIMATE, lift_damping=LIFT_JOINT_DAMPING_ESTIMATE)


def build_spec(urdf_root=None):
    """Build the MjSpec of walkie. Returns (spec, report)."""
    urdf = UrdfModel(urdf_root if urdf_root is not None else expand_urdf())
    spec = mujoco.MjSpec.from_string(prepare_urdf(urdf.root))
    spec.modelname = MODEL_NAME
    # Static bodies are not fused: every URDF link stays a body of the same name. Rigidly
    # attached bodies already behave as one for contact filtering (MuJoCo weld groups), and
    # fusing in place would rewrite the spec, which then no longer recompiles.
    # The MJCF writer omits these compiler options, so they must keep their MJCF defaults for
    # the written file to compile to the same model (checked by to_xml).
    spec.compiler.fusestatic = False
    spec.compiler.discardvisual = False
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO
    spec.option.timestep = TIMESTEP
    spec.option.integrator = INTEGRATOR
    spec.option.cone = CONE

    spec.body(ROOT_LINK).add_freejoint(name=FREE_JOINT)
    _set_inertia_estimates(spec)
    _configure_geoms(spec)
    _configure_joints(spec, urdf)
    _add_actuators(spec, urdf)
    _add_frame_sites(spec, urdf)
    _add_imu(spec)
    cameras = _add_cameras(spec, urdf)
    lidars = _add_lidars(spec, urdf)
    spec.visual.global_.offwidth = max(width for _, _, width, _ in cameras)
    spec.visual.global_.offheight = max(height for _, _, _, height in cameras)
    spec.visual.rgba.rangefinder = list(RANGEFINDER_RGBA)
    _set_rest_height(spec)
    excludes = _add_self_collision_excludes(spec)
    _add_home_keyframe(spec, urdf)
    # Added last, so the forward passes of the steps above do not cast its rays.
    _add_lidar_3d(spec)
    spec.comment = _header(lidars, cameras, excludes)
    return spec, {'lidars': lidars, 'cameras': cameras, 'excludes': excludes, 'urdf': urdf}


def to_xml(spec):
    """
    MJCF text with mesh paths relative to MUJOCO_DIR.

    The text is compiled again and compared with the spec, so a setting the MJCF writer drops
    cannot silently change the model.
    """
    spec.meshdir = RELATIVE_MESHDIR
    spec.modelfiledir = str(MUJOCO_DIR)
    # MuJoCo fills in the content type of some meshes while compiling, depending on its asset
    # cache; setting it for every mesh keeps the output independent of what ran before.
    for mesh in spec.meshes:
        mesh.content_type = MESH_CONTENT_TYPES[Path(mesh.file).suffix.lower()]
    xml = spec.to_xml()
    expected = spec.compile()
    reloaded_spec = mujoco.MjSpec.from_string(xml)
    reloaded_spec.modelfiledir = str(MUJOCO_DIR)
    reloaded = reloaded_spec.compile()
    for field in ('nq', 'nv', 'nu', 'nbody', 'ngeom', 'nsite', 'nsensor', 'nplugin', 'neq',
                  'nexclude', 'nkey'):
        if getattr(expected, field) != getattr(reloaded, field):
            raise RuntimeError(f'written model differs from the spec in {field}')
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'geom_size', 'geom_contype',
                  'geom_conaffinity', 'dof_armature', 'dof_damping', 'actuator_gainprm',
                  'actuator_biasprm', 'site_pos', 'key_qpos'):
        if not np.allclose(getattr(expected, field), getattr(reloaded, field), atol=1e-9):
            raise RuntimeError(f'written model differs from the spec in {field}')
    return xml


def check_ros2_control(urdf, model):
    """Problems with the MuJoCo ros2_control block: missing, or disagreeing with the actuators."""
    ros2_control = urdf.mujoco_ros2_control()
    if ros2_control is None:
        return ['the expanded URDF has no MuJoCo <ros2_control> block']
    warnings = []
    joints = ros2_control_joints(ros2_control)
    actuators = {model.actuator(index).name for index in range(model.nu)}
    for name in sorted(set(joints) - actuators):
        warnings.append(f"ros2_control joint '{name}' has no actuator")
    for name in sorted(actuators - set(joints)):
        warnings.append(f"actuator '{name}' has no ros2_control joint")
    return warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('output', nargs='?', type=Path, default=DEFAULT_OUTPUT,
                        help=f'output MJCF file (default: {DEFAULT_OUTPUT})')
    args = parser.parse_args(argv)

    spec, report = build_spec()
    xml = to_xml(spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml)

    model = spec.compile()
    problems = check_ros2_control(report['urdf'], model)
    for problem in problems:
        print(f'error: {problem}', file=sys.stderr)
    mass = sum(model.body_mass)
    print(f'wrote {args.output}: nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} '
          f'ngeom={model.ngeom} nmesh={model.nmesh} nsensor={model.nsensor} '
          f'mass={mass:.3f} kg, {len(report["excludes"])} self-collision excludes')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
