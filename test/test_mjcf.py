"""
Tests of the generated MuJoCo model (mujoco/walkie.xml) and its default world (mujoco/scene.xml).

They cover the physics of the default world (rest, omni-wheel drive, lidar emulation), the
contract between the model and the MuJoCo ros2_control block of the URDF, the collision
classes, and that the checked-in model matches what scripts/build_mjcf.py generates.

All files are located relative to this test file, so the tests run from a source checkout
without building the package, e.g.

    pixi run python -m pytest description/test/test_mjcf.py
"""

import importlib.util
import math
from pathlib import Path
import re
import sys

import pytest
import yaml

mujoco = pytest.importorskip('mujoco')
np = pytest.importorskip('numpy')


DESCRIPTION_DIR = Path(__file__).resolve().parent.parent
MUJOCO_DIR = DESCRIPTION_DIR / 'mujoco'
MODEL_FILE = MUJOCO_DIR / 'walkie.xml'
SCENE_FILE = MUJOCO_DIR / 'scene.xml'
GENERATOR_FILE = DESCRIPTION_DIR / 'scripts' / 'build_mjcf.py'
CONTROLLERS_FILE = DESCRIPTION_DIR / 'config' / 'ros2_controller' / 'mujoco_controllers.yaml'

ROOT_BODY = 'base_footprint'
FREE_JOINT = 'floating_base_joint'
WHEEL_JOINTS = ('fl_wheel_joint', 'fr_wheel_joint', 'bl_wheel_joint', 'br_wheel_joint')
LIDARS = ('front_lidar', 'back_lidar')
IMU = 'top_imu'
KEYFRAME = 'home'

# Plausible total mass of the robot in kg.
MASS_RANGE = (25.0, 60.0)
REST_DURATION = 2.0
REST_HEIGHT_TOLERANCE = 0.003
REST_DRIFT_TOLERANCE = 0.01

DRIVE_COMMANDS = {
    'forward': (0.3, 0.0, 0.0),
    'sideways': (0.0, 0.3, 0.0),
    'rotation': (0.0, 0.0, 0.5),
    'combined': (0.2, -0.15, 0.4),
}
DRIVE_SETTLE_TIME = 1.5
DRIVE_MEASURE_TIME = 0.5
# Tight enough to catch a few percent of error in wheel_radius or robot_radius. The current
# shortfall (about 1.5 % linear, 2.4 % angular) comes from the controller wheel_radius 0.076
# against the 0.075 collision radius of the wheels.
DRIVE_LINEAR_RELATIVE_TOLERANCE = 0.03
DRIVE_ANGULAR_RELATIVE_TOLERANCE = 0.04
DRIVE_LINEAR_FLOOR = 0.002   # m/s, absolute slack for small components
DRIVE_ANGULAR_FLOOR = 0.005  # rad/s

LIFT_JOINT = 'lift_joint'
LIFT_STEP = 0.2              # m, setpoint step from the home keyframe
LIFT_STEP_DURATION = 3.0
LIFT_LIFTOFF_TOLERANCE = 0.005  # m, largest allowed change of the root height
LIFT_SPEED_BOUND = 0.2       # m/s, twice the URDF velocity limit of lift_joint

LIDAR_TOLERANCE = 0.02
WALL_DISTANCE = 1.0          # m from the robot origin to the wall face, along +x
WALL_HALF_SIZE = (0.05, 3.0, 0.5)

LIDAR_3D = 'unitree_l2'
# Self-filter box of the head lidar cloud (bringup/config/lidar/unitree_self_filter.yaml), in
# the root body frame. Rays that hit the robot must end inside it.
SELF_FILTER_BOX = ((-0.45, -0.45, 0.0), (0.45, 0.45, 1.9))
# Shares of the rays in the empty scene at the home keyframe (measured: 67 % floor, 21 % robot,
# 12 % upward rays with no hit).
LIDAR_3D_FLOOR_SHARE = 0.6
LIDAR_3D_MAX_SELF_SHARE = 0.3
LIDAR_3D_TOLERANCE = 1e-6
# A box beside the robot (+y), where the head lidar sees down from its horizon.
LIDAR_3D_BOX_CENTER = (0.0, 1.5, 1.0)
LIDAR_3D_BOX_HALF_SIZE = (0.3, 0.1, 1.0)


# ----------------------------------------------------------------------------------------------
# Fixtures and helpers
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def generator():
    """Load the scripts/build_mjcf.py module (needs xacro to expand the description)."""
    pytest.importorskip('xacro')
    spec = importlib.util.spec_from_file_location('build_mjcf', GENERATOR_FILE)
    module = importlib.util.module_from_spec(spec)
    # Keep the source tree free of __pycache__ directories.
    dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
    return module


@pytest.fixture(scope='module')
def urdf(generator):
    """Expand the URDF (ros2_control:=mujoco) into a generator UrdfModel."""
    return generator.UrdfModel(generator.expand_urdf())


@pytest.fixture(scope='module')
def scene_model():
    return mujoco.MjModel.from_xml_path(str(SCENE_FILE))


def _scene_with(add_bodies):
    """Compile the default scene after letting add_bodies(spec) add world bodies."""
    spec = mujoco.MjSpec.from_file(str(SCENE_FILE))
    add_bodies(spec)
    return spec.compile()


def _home_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(KEYFRAME).id)
    mujoco.mj_forward(model, data)
    return data


def _step(model, data, duration):
    for _ in range(int(round(duration / model.opt.timestep))):
        mujoco.mj_step(model, data)


def _base_twist(model, data):
    """(vx, vy, wz) of the root body, expressed in the root body frame."""
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(ROOT_BODY).id,
                             velocity, 1)
    return np.array([velocity[3], velocity[4], velocity[2]])


def _lidar_parameters(urdf, generator, name):
    for _, sensor in urdf.gazebo_sensors():
        if sensor.get('name') == name:
            return generator._lidar_parameters(sensor)
    raise AssertionError(f"lidar '{name}' not found in the URDF")


def _lidar_readings(model, data, lidar):
    """Beam origins, unit directions and readings of a lidar, in beam order."""
    origins, directions, readings = [], [], []
    index = 0
    while True:
        try:
            sensor = model.sensor(f'{lidar}-{index}')
        except KeyError:
            break
        site = sensor.objid[0]
        origins.append(data.site_xpos[site].copy())
        directions.append(data.site_xmat[site].reshape(3, 3)[:, 2].copy())
        readings.append(data.sensordata[sensor.adr[0]])
        index += 1
    return np.array(origins), np.array(directions), np.array(readings)


def _ray_box_distance(origin, direction, center, half_size):
    """Distance along a ray to an axis-aligned box, or None (slab method)."""
    near, far = -math.inf, math.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if abs(origin[axis] - center[axis]) > half_size[axis]:
                return None
            continue
        first = (center[axis] - half_size[axis] - origin[axis]) / direction[axis]
        second = (center[axis] + half_size[axis] - origin[axis]) / direction[axis]
        near, far = max(near, min(first, second)), min(far, max(first, second))
    if far < max(near, 0.0) or near < 0.0:
        return None
    return near


def _lidar_3d_rays(model, data, generator):
    """Origin, unit directions (world frame) and readings of the head 3D lidar, in data order."""
    config = generator.lidar_3d_config()
    azimuth_count, elevation_count = (int(value) for value in config['resolution'].split())
    azimuth = np.linspace(*(float(value) for value in config['azimuth_range'].split()),
                          azimuth_count)
    elevation = np.linspace(*(float(value) for value in config['elevation_range'].split()),
                            elevation_count)
    # The extension stores elevation rows of azimuth columns; a ray is
    # (cos az cos el, sin az cos el, sin el) in the site frame.
    elevations, azimuths = np.meshgrid(elevation, azimuth, indexing='ij')
    local = np.stack([np.cos(azimuths) * np.cos(elevations),
                      np.sin(azimuths) * np.cos(elevations),
                      np.sin(elevations)], axis=-1).reshape(-1, 3)
    sensor = model.sensor(LIDAR_3D)
    site = sensor.objid[0]
    directions = local @ data.site_xmat[site].reshape(3, 3).T
    readings = data.sensordata[sensor.adr[0]:sensor.adr[0] + sensor.dim[0]].copy()
    return data.site_xpos[site].copy(), directions, readings


def _floor_distances(model, origin, directions):
    """Distance along each ray to the floor plane, inf for rays that do not point down."""
    height = origin[2] - model.geom('floor').pos[2]
    distances = np.full(len(directions), np.inf)
    down = directions[:, 2] < 0
    distances[down] = height / -directions[down, 2]
    return distances


def _contact_bodies(model, data):
    return {tuple(sorted(model.body(model.geom_bodyid[geom]).name for geom in contact.geom))
            for contact in data.contact[:data.ncon]}


# ----------------------------------------------------------------------------------------------
# (a) The default world compiles and the robot rests stably
# ----------------------------------------------------------------------------------------------
def test_scene_compiles_with_plausible_mass(scene_model):
    robot_mass = scene_model.body_subtreemass[scene_model.body(ROOT_BODY).id]
    assert MASS_RANGE[0] < robot_mass < MASS_RANGE[1], f'robot mass {robot_mass:.2f} kg'
    free_joint = scene_model.joint(FREE_JOINT)
    assert free_joint.type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert scene_model.body(free_joint.bodyid[0]).name == ROOT_BODY
    assert scene_model.opt.timestep == pytest.approx(0.002)
    assert scene_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert scene_model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC


def test_model_loads_without_scene():
    model = mujoco.MjModel.from_xml_path(str(MODEL_FILE))
    assert all(model.geom(index).name != 'floor' for index in range(model.ngeom))


def test_rest_is_stable(scene_model):
    data = _home_data(scene_model)
    start = data.qpos[:3].copy()
    heights = []
    for _ in range(int(round(REST_DURATION / scene_model.opt.timestep))):
        mujoco.mj_step(scene_model, data)
        heights.append(data.qpos[2])
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    warnings = {mujoco.mjtWarning(index).name: warning.number
                for index, warning in enumerate(data.warning) if warning.number}
    assert not warnings
    assert max(abs(height - start[2]) for height in heights) < REST_HEIGHT_TOLERANCE
    assert np.linalg.norm(data.qpos[:2] - start[:2]) < REST_DRIFT_TOLERANCE


def test_imu_at_rest(scene_model):
    data = _home_data(scene_model)
    _step(scene_model, data, 0.5)
    site = scene_model.site(IMU)
    assert scene_model.body(site.bodyid[0]).name == 'top_imu_link'
    accel = data.sensor(f'{IMU}_accel').data
    gyro = data.sensor(f'{IMU}_gyro').data
    quat = data.sensor(f'{IMU}_quat').data
    assert scene_model.sensor(f'{IMU}_quat').type[0] == mujoco.mjtSensor.mjSENS_FRAMEQUAT
    site_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, data.site_xmat[site.id])
    assert accel == pytest.approx([0.0, 0.0, 9.81], abs=0.05)
    assert gyro == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
    assert abs(np.dot(quat, site_quat)) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('step', (LIFT_STEP, -LIFT_STEP), ids=('up', 'down'))
def test_lift_step_keeps_base_on_floor(scene_model, step):
    """
    A lift setpoint step moves the lift at drive speed and leaves the base in place.

    Lowering by LIFT_STEP from home brings the hands onto the base, so the down step also
    checks that the lift stalls there instead of pushing the robot around.
    """
    data = _home_data(scene_model)
    actuator = scene_model.actuator(LIFT_JOINT).id
    dof = scene_model.joint(LIFT_JOINT).dofadr[0]
    start = data.qpos[:3].copy()
    data.ctrl[actuator] += step
    height_change, peak_speed = 0.0, 0.0
    for _ in range(int(round(LIFT_STEP_DURATION / scene_model.opt.timestep))):
        mujoco.mj_step(scene_model, data)
        height_change = max(height_change, abs(data.qpos[2] - start[2]))
        peak_speed = max(peak_speed, abs(data.qvel[dof]))
    assert np.all(np.isfinite(data.qpos))
    assert height_change < LIFT_LIFTOFF_TOLERANCE
    assert np.linalg.norm(data.qpos[:2] - start[:2]) < REST_DRIFT_TOLERANCE
    assert peak_speed < LIFT_SPEED_BOUND


# ----------------------------------------------------------------------------------------------
# (b) Omni-wheel drive follows the inverse kinematics of omni_wheel_drive_controller
# ----------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def omni_controller():
    with open(CONTROLLERS_FILE) as stream:
        return yaml.safe_load(stream)['omni_wheel_drive_controller']['ros__parameters']


def test_wheel_layout_matches_controller(scene_model, omni_controller):
    data = _home_data(scene_model)
    root = scene_model.body(ROOT_BODY).id
    rotation = data.xmat[root].reshape(3, 3)
    for index, name in enumerate(omni_controller['wheel_names']):
        joint = scene_model.joint(name).id
        position = rotation.T @ (data.xanchor[joint] - data.xpos[root])
        axis = rotation.T @ data.xaxis[joint]
        angle = index * math.pi / 2 + omni_controller['wheel_offset']
        assert math.atan2(position[1], position[0]) == pytest.approx(
            math.remainder(angle, 2 * math.pi), abs=1e-3), name
        assert math.hypot(position[0], position[1]) == pytest.approx(
            omni_controller['robot_radius'], abs=0.005), name
        # The controller assumes positive wheel speed turns the axle outward-pointing.
        assert axis == pytest.approx([math.cos(angle), math.sin(angle), 0.0], abs=1e-3), name


@pytest.mark.parametrize('command', DRIVE_COMMANDS.values(), ids=DRIVE_COMMANDS.keys())
def test_drive_follows_omni_kinematics(scene_model, omni_controller, command):
    data = _home_data(scene_model)
    root = scene_model.body(ROOT_BODY).id
    rotation = data.xmat[root].reshape(3, 3)
    vx, vy, wz = command
    for name in omni_controller['wheel_names']:
        # Wheel angle from the compiled model, radii from the controller configuration.
        position = rotation.T @ (data.xanchor[scene_model.joint(name).id] - data.xpos[root])
        angle = math.atan2(position[1], position[0])
        data.ctrl[scene_model.actuator(name).id] = (
            math.sin(angle) * vx - math.cos(angle) * vy
            - wz * omni_controller['robot_radius']) / omni_controller['wheel_radius']
    _step(scene_model, data, DRIVE_SETTLE_TIME)
    twists = []
    for _ in range(int(round(DRIVE_MEASURE_TIME / scene_model.opt.timestep))):
        mujoco.mj_step(scene_model, data)
        twists.append(_base_twist(scene_model, data))
    measured = np.mean(twists, axis=0)

    linear_error = np.linalg.norm(measured[:2] - [vx, vy])
    linear_tolerance = DRIVE_LINEAR_RELATIVE_TOLERANCE * math.hypot(vx, vy) + DRIVE_LINEAR_FLOOR
    angular_tolerance = DRIVE_ANGULAR_RELATIVE_TOLERANCE * abs(wz) + DRIVE_ANGULAR_FLOOR
    assert linear_error < linear_tolerance, f'measured {measured}, commanded {command}'
    assert abs(measured[2] - wz) < angular_tolerance, f'measured {measured}, commanded {command}'


# ----------------------------------------------------------------------------------------------
# (c) Lidar emulation
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lidar', LIDARS)
def test_lidar_beam_layout(scene_model, urdf, generator, lidar):
    params = _lidar_parameters(urdf, generator, lidar)
    data = _home_data(scene_model)
    origins, directions, _ = _lidar_readings(scene_model, data, lidar)
    expected_angles = generator.lidar_beam_angles(params['min_angle'], params['max_angle'])
    assert len(directions) == len(expected_angles)
    # Beams span the advertised field of view exactly, at about the nominal resolution.
    assert expected_angles[0] == pytest.approx(params['min_angle'], abs=1e-12)
    assert expected_angles[-1] == pytest.approx(params['max_angle'], abs=1e-12)
    assert np.degrees(np.diff(expected_angles)) == pytest.approx(
        generator.LIDAR_RESOLUTION_DEG, rel=0.01)

    # A LaserScan publisher configured from the model header (as RangefinderLidarPlugin
    # computes it) reads exactly the generated beams.
    header = re.search(
        rf'{lidar}: frame \S+, (\d+) beams, min_angle (\S+), max_angle (\S+), '
        r'angle_increment ([0-9.]+),', MODEL_FILE.read_text())
    assert header is not None, f'{lidar} is not documented in the {MODEL_FILE.name} header'
    count, min_angle, max_angle, increment = (int(header[1]), float(header[2]),
                                              float(header[3]), float(header[4]))
    assert count == len(expected_angles)
    assert int((max_angle - min_angle) / increment) + 1 == count

    frame = scene_model.site(f'{lidar}_link')
    frame_rotation = data.site_xmat[frame.id].reshape(3, 3)
    assert np.allclose(origins, data.site_xpos[frame.id], atol=1e-9)
    # MuJoCo's XML writer stores orientations with six significant digits, so beam and frame
    # orientations each carry an error of up to about 1e-6.
    for direction, angle in zip(directions, expected_angles):
        local = frame_rotation.T @ direction
        assert local == pytest.approx([math.cos(angle), math.sin(angle), 0.0], abs=1e-5)
    for index in range(len(expected_angles)):
        sensor = scene_model.sensor(f'{lidar}-{index}').id
        assert scene_model.sensor_interval[sensor][0] == pytest.approx(1.0 / params['update_rate'])
        assert scene_model.sensor_history[sensor][0] == 1  # 'interval' needs a history buffer


@pytest.mark.parametrize('lidar', LIDARS)
def test_lidar_sees_nothing_in_empty_scene(scene_model, urdf, generator, lidar):
    """No beam hits the robot itself or the floor within the scanner range."""
    params = _lidar_parameters(urdf, generator, lidar)
    data = _home_data(scene_model)
    _step(scene_model, data, 0.5)
    _, _, readings = _lidar_readings(scene_model, data, lidar)
    assert readings.size == len(
        generator.lidar_beam_angles(params['min_angle'], params['max_angle']))
    in_range = readings[(readings >= 0) & (readings <= params['range_max'])]
    assert in_range.size == 0, f'unexpected hits at {in_range}'


def test_lidar_ranges_to_wall(urdf, generator):
    """Readings match the analytic distance to a wall in front of the robot."""
    center = np.array([WALL_DISTANCE + WALL_HALF_SIZE[0], 0.0, WALL_HALF_SIZE[2]])

    def add_wall(spec):
        body = spec.worldbody.add_body(name='test_wall', pos=center.tolist())
        body.add_geom(name='test_wall', type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=list(WALL_HALF_SIZE), contype=generator.FURN,
                      conaffinity=generator.COLLIDES_WITH[generator.FURN])

    model = _scene_with(add_wall)
    data = _home_data(model)
    _step(model, data, 0.1)
    hits = {}
    for lidar in LIDARS:
        params = _lidar_parameters(urdf, generator, lidar)
        origins, directions, readings = _lidar_readings(model, data, lidar)
        hits[lidar] = 0
        for index, (origin, direction, reading) in enumerate(
                zip(origins, directions, readings)):
            expected = _ray_box_distance(origin, direction, center, WALL_HALF_SIZE)
            if expected is not None and expected <= params['range_max']:
                hits[lidar] += 1
                assert reading == pytest.approx(expected, abs=LIDAR_TOLERANCE), \
                    f'{lidar}-{index}'
            else:
                assert reading < 0 or reading > params['range_max'], f'{lidar}-{index}'
    # The front scanner faces the wall; the back scanner only sees it with its outermost beams.
    assert hits['front_lidar'] > 100
    assert 0 < hits['back_lidar'] < hits['front_lidar']


def test_lidar_3d_layout(scene_model, generator):
    """The head 3D lidar is one extension sensor on the URDF frame of the real cloud."""
    sensor = scene_model.sensor(LIDAR_3D)
    assert sensor.type[0] == mujoco.mjtSensor.mjSENS_PLUGIN
    assert sensor.objtype[0] == mujoco.mjtObj.mjOBJ_SITE
    site = scene_model.site(sensor.objid[0])
    assert site.name == generator.LIDAR_3D_SITE
    assert scene_model.body(site.bodyid[0]).name == generator.LIDAR_3D_SITE
    assert sensor.dim[0] == math.prod(generator.LIDAR_3D_RESOLUTION)
    plugins = mujoco.MjSpec.from_file(str(MODEL_FILE)).plugins
    assert [(plugin.name, plugin.plugin_name, plugin.config) for plugin in plugins] == [
        (LIDAR_3D, generator.LIDAR_3D_PLUGIN, generator.lidar_3d_config())]

    # The rays sit at the centres of an even grid of cells covering the hemisphere z >= 0 of
    # the site, within the [-pi, pi] the extension accepts.
    (azimuth_min, azimuth_max), (elevation_min, elevation_max) = \
        generator.lidar_3d_angle_ranges()
    azimuth_count, elevation_count = generator.LIDAR_3D_RESOLUTION
    azimuth_step = (azimuth_max - azimuth_min) / (azimuth_count - 1)
    elevation_step = (elevation_max - elevation_min) / (elevation_count - 1)
    assert azimuth_min - azimuth_step / 2 == pytest.approx(-math.pi)
    assert azimuth_max + azimuth_step / 2 == pytest.approx(math.pi)
    assert elevation_min - elevation_step / 2 == pytest.approx(0.0, abs=1e-12)
    assert elevation_max + elevation_step / 2 == pytest.approx(math.pi / 2)
    assert -math.pi < azimuth_min and azimuth_max < math.pi

    header = re.search(rf'{LIDAR_3D}: site (\S+), (\d+) x (\d+) rays', MODEL_FILE.read_text())
    assert header is not None, f'{LIDAR_3D} is not documented in the {MODEL_FILE.name} header'
    assert (header[1], int(header[2]), int(header[3])) == (
        generator.LIDAR_3D_SITE, azimuth_count, elevation_count)


def test_lidar_3d_sees_floor(scene_model, generator):
    """In the empty scene the rays end on the floor or on the robot, inside the self-filter box."""
    data = _home_data(scene_model)
    origin, directions, readings = _lidar_3d_rays(scene_model, data, generator)
    floor = _floor_distances(scene_model, origin, directions)
    min_range, max_range = generator.LIDAR_3D_RANGE
    hit = readings >= 0
    assert np.all((readings[hit] >= min_range) & (readings[hit] <= max_range))
    # Nothing is seen beyond the floor, and every ray that reaches the floor in range hits.
    assert np.all(readings[hit] <= floor[hit] + LIDAR_3D_TOLERANCE)
    assert np.all(hit[floor <= max_range])
    on_floor = hit & (np.abs(readings - floor) < LIDAR_3D_TOLERANCE)
    on_robot = hit & ~on_floor
    assert on_floor.mean() > LIDAR_3D_FLOOR_SHARE
    assert on_robot.mean() < LIDAR_3D_MAX_SELF_SHARE

    root = scene_model.body(ROOT_BODY).id
    points = origin + directions[on_robot] * readings[on_robot, None]
    local = (points - data.xpos[root]) @ data.xmat[root].reshape(3, 3)
    lower, upper = SELF_FILTER_BOX
    assert np.all((local >= lower) & (local <= upper)), 'robot hits outside the self-filter box'


def test_lidar_3d_ranges_to_box(generator):
    """Readings match the analytic distance to a box beside the robot."""
    center = np.array(LIDAR_3D_BOX_CENTER)

    def add_box(spec):
        body = spec.worldbody.add_body(name='test_box', pos=center.tolist())
        body.add_geom(name='test_box', type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=list(LIDAR_3D_BOX_HALF_SIZE), contype=generator.FURN,
                      conaffinity=generator.COLLIDES_WITH[generator.FURN])

    model = _scene_with(add_box)
    data = _home_data(model)
    origin, directions, readings = _lidar_3d_rays(model, data, generator)
    floor = _floor_distances(model, origin, directions)
    box = np.array([_ray_box_distance(origin, direction, center, LIDAR_3D_BOX_HALF_SIZE)
                    or math.inf for direction in directions])
    nearest = np.minimum(floor, box)
    hit = readings >= 0
    # Rays end on the nearest surface, or earlier on the robot itself.
    assert np.all(readings[hit] <= nearest[hit] + LIDAR_3D_TOLERANCE)
    towards_box = box < floor
    on_box = hit & towards_box & (np.abs(readings - box) < LIDAR_3D_TOLERANCE)
    on_robot = hit & towards_box & (readings < box - LIDAR_3D_TOLERANCE)
    assert np.all(on_box | on_robot | ~towards_box)
    assert on_box.sum() > 100


# ----------------------------------------------------------------------------------------------
# (d) Contract with the MuJoCo ros2_control block of the URDF
# ----------------------------------------------------------------------------------------------
def _actuator_kind(model, actuator):
    """'position', 'velocity' or 'other' from the actuator gain and bias parameters."""
    gain = model.actuator_gainprm[actuator, 0]
    bias = model.actuator_biasprm[actuator, :3]
    if model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED or \
            model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE:
        return 'other'
    if bias[0] == 0 and bias[1] == pytest.approx(-gain) and gain > 0:
        return 'position'
    if bias[0] == 0 and bias[1] == 0 and bias[2] == pytest.approx(-gain) and gain > 0:
        return 'velocity'
    return 'other'


def test_ros2_control_contract(scene_model, urdf, generator):
    ros2_control = urdf.mujoco_ros2_control()
    assert ros2_control is not None, (
        'the expanded URDF (ros2_control:=mujoco) has no <ros2_control> block using '
        f'{generator.MUJOCO_HARDWARE_PLUGIN}')

    joints = {joint.get('name'): joint for joint in ros2_control.findall('joint')}
    actuators = {scene_model.actuator(index).name: index for index in range(scene_model.nu)}
    assert set(actuators) == set(joints), 'actuators and ros2_control joints differ'

    for name, joint in joints.items():
        commands = [interface.get('name') for interface in joint.findall('command_interface')]
        states = {interface.get('name') for interface in joint.findall('state_interface')}
        expected = 'velocity' if name in WHEEL_JOINTS else 'position'
        assert commands == [expected], f'{name} command interfaces {commands}'
        assert {'position', 'velocity'} <= states, f'{name} state interfaces {states}'
        actuator = actuators[name]
        assert scene_model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
        assert scene_model.joint(scene_model.actuator_trnid[actuator, 0]).name == name
        assert _actuator_kind(scene_model, actuator) == expected, name

    for joint in urdf.root.findall('joint'):
        if joint.find('mimic') is not None:
            name = joint.get('name')
            assert name not in joints and name not in actuators, name
            equality = scene_model.equality(f'{name}_mimic')
            assert scene_model.joint(equality.obj1id[0]).name == name

    for sensor in ros2_control.findall('sensor'):
        params = {param.get('name'): param.text for param in sensor.findall('param')}
        if params.get('mujoco_type') == 'imu':
            prefix = params.get('mujoco_sensor_name', sensor.get('name'))
            for suffix in ('quat', 'gyro', 'accel'):
                scene_model.sensor(f'{prefix}_{suffix}')

    model_param = ros2_control.find("hardware/param[@name='mujoco_model']")
    assert model_param is not None
    assert model_param.text.strip().endswith('walkie_description/mujoco/scene.xml')


def test_home_keyframe_matches_initial_positions(scene_model, urdf):
    ros2_control = urdf.mujoco_ros2_control()
    assert ros2_control is not None, 'the expanded URDF has no MuJoCo <ros2_control> block'
    key = scene_model.key(KEYFRAME)
    checked = 0
    for joint in ros2_control.findall('joint'):
        for param in joint.findall("state_interface[@name='position']/param"):
            if param.get('name') == 'initial_value':
                address = scene_model.joint(joint.get('name')).qposadr[0]
                assert key.qpos[address] == pytest.approx(float(param.text)), joint.get('name')
                checked += 1
    assert checked, 'the MuJoCo <ros2_control> block declares no initial positions'


def test_cameras_at_optical_frames(scene_model, urdf, generator):
    data = _home_data(scene_model)
    for name, frame, sensor_name in generator.CAMERAS:
        camera = scene_model.camera(name)
        assert scene_model.body(camera.bodyid[0]).name == frame
        camera_rotation = data.cam_xmat[camera.id].reshape(3, 3)
        frame_rotation = data.xmat[scene_model.body(frame).id].reshape(3, 3)
        # MuJoCo cameras look along -Z with +Y up, ROS optical frames along +Z with +Y down.
        assert -camera_rotation[:, 2] == pytest.approx(frame_rotation[:, 2], abs=1e-9)
        assert camera_rotation[:, 1] == pytest.approx(-frame_rotation[:, 1], abs=1e-9)
        assert data.cam_xpos[camera.id] == pytest.approx(
            data.xpos[scene_model.body(frame).id], abs=1e-9)
        _, sensor = urdf.gazebo_sensor(sensor_name)
        hfov = float(sensor.findtext('.//camera/horizontal_fov'))
        width, height = scene_model.cam_resolution[camera.id]
        horizontal = 2 * math.atan(math.tan(math.radians(camera.fovy[0]) / 2) * width / height)
        assert horizontal == pytest.approx(hfov, abs=1e-4), name


# ----------------------------------------------------------------------------------------------
# (e) The checked-in model is up to date
# ----------------------------------------------------------------------------------------------
def test_generated_model_is_up_to_date(generator, tmp_path):
    output = tmp_path / 'walkie.xml'
    generator.main([str(output)])
    assert output.read_text() == MODEL_FILE.read_text(), (
        'mujoco/walkie.xml is out of date; regenerate it with "pixi run mjcf"')


# ----------------------------------------------------------------------------------------------
# (f) Collision classes
# ----------------------------------------------------------------------------------------------
def _collides(model, geom, contype, conaffinity):
    return bool(model.geom_contype[geom] & conaffinity or contype & model.geom_conaffinity[geom])


def _robot_geoms(model, collision_class):
    root = model.body(ROOT_BODY).id
    return [geom for geom in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[geom]] == root
            and model.geom_contype[geom] == collision_class]


def test_floor_and_wheel_pairs(scene_model, generator):
    floor = scene_model.geom('floor')
    assert floor.type[0] == mujoco.mjtGeom.mjGEOM_PLANE
    assert floor.contype[0] == generator.FLOOR
    assert floor.conaffinity[0] == generator.COLLIDES_WITH[generator.FLOOR]
    pairs = {}
    for index in range(scene_model.npair):
        geoms = {scene_model.geom(scene_model.pair_geom1[index]).name,
                 scene_model.geom(scene_model.pair_geom2[index]).name}
        pairs[frozenset(geoms)] = index
    for wheel in generator.WHEELS:
        capsule = scene_model.geom(generator.wheel_contact_geom(wheel))
        assert capsule.type[0] == mujoco.mjtGeom.mjGEOM_CAPSULE
        assert capsule.contype[0] == 0 and capsule.conaffinity[0] == 0
        pair = pairs[frozenset({capsule.name, 'floor'})]
        friction = scene_model.pair_friction[pair]
        assert scene_model.pair_dim[pair] == 3
        assert friction[0] < 0.1 * friction[1], 'axle friction must be far below rolling'


def test_collision_classes_by_bitmask(scene_model, generator):
    furniture = (generator.FURN, generator.COLLIDES_WITH[generator.FURN])
    objects = (generator.OBJ, generator.COLLIDES_WITH[generator.OBJ])
    floor = (generator.FLOOR, generator.COLLIDES_WITH[generator.FLOOR])
    for collision_class in (generator.BASE, generator.BODY):
        geoms = _robot_geoms(scene_model, collision_class)
        assert geoms
        for geom in geoms:
            for other in (furniture, objects, floor):
                assert not _collides(scene_model, geom, *other), scene_model.geom(geom).name
    for collision_class in (generator.ARM, generator.GRIP):
        geoms = _robot_geoms(scene_model, collision_class)
        assert geoms
        for geom in geoms:
            name = scene_model.geom(geom).name
            assert _collides(scene_model, geom, *furniture), name
            assert _collides(scene_model, geom, *objects), name
            for other_class in (generator.BASE, generator.BODY, generator.ARM, generator.GRIP):
                assert _collides(scene_model, geom, other_class,
                                 generator.COLLIDES_WITH[other_class]), name
    # Every robot geom that collides at all is in exactly one class.
    root = scene_model.body(ROOT_BODY).id
    for geom in range(scene_model.ngeom):
        if scene_model.body_rootid[scene_model.geom_bodyid[geom]] == root:
            contype = int(scene_model.geom_contype[geom])
            assert contype & (contype - 1) == 0, scene_model.geom(geom).name


def test_collision_classes_in_contact(generator):
    """Furniture overlapping the base is ignored, furniture and objects at the hands are not."""
    model = mujoco.MjModel.from_xml_path(str(SCENE_FILE))
    data = _home_data(model)
    left_tcp = data.site_xpos[model.site('openarm_left_hand_tcp').id].copy()
    right_tcp = data.site_xpos[model.site('openarm_right_hand_tcp').id].copy()

    def add_props(spec):
        for name, collision_class, pos, size in (
                ('furniture_in_base', generator.FURN, [0.25, 0.0, 0.15], [0.1, 0.1, 0.1]),
                ('furniture_at_left_hand', generator.FURN, left_tcp, [0.03, 0.03, 0.03]),
                ('object_at_right_hand', generator.OBJ, right_tcp, [0.03, 0.03, 0.03])):
            body = spec.worldbody.add_body(name=name, pos=list(pos))
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, contype=collision_class,
                          conaffinity=generator.COLLIDES_WITH[collision_class])
        body = spec.worldbody.add_body(name='object_on_floor', pos=[2.0, 0.0, 0.049])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05] * 3, contype=generator.OBJ,
                      conaffinity=generator.COLLIDES_WITH[generator.OBJ])

    model = _scene_with(add_props)
    data = _home_data(model)
    pairs = _contact_bodies(model, data)
    touching = {name for pair in pairs for name in pair}
    assert not any('furniture_in_base' in pair for pair in pairs), pairs
    assert 'furniture_at_left_hand' in touching
    assert 'object_at_right_hand' in touching
    assert ('object_on_floor', 'world') in pairs
