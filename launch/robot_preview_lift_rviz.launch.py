#!/usr/bin/env python3
#
# Preview RViz launch that lets an external publisher (e.g. lift_homing_node)
# drive /joint_states directly. No joint_state_publisher_gui is started, so
# non-lift joints sit at URDF zero pose.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'walkie_description'

    package_dir = os.path.join(get_package_share_directory(package_name))

    robot_model = LaunchConfiguration('robot_model', default=os.path.join(get_package_share_directory(package_name), 'robots', 'gz_walkie.urdf.xacro'))
    ros2_control = LaunchConfiguration('ros2_control', default='gazebo')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    rviz_config_file = LaunchConfiguration('rviz_config_file')
    use_rviz = LaunchConfiguration('use_rviz')

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config_file',
        default_value=os.path.join(package_dir, 'rviz', 'robot_preview.rviz'),
        description='Full path to the RVIZ config file to use')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Whether to start RVIZ')

    rviz_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_dir, 'launch', 'rviz.launch.py')),
        condition=IfCondition(use_rviz),
        launch_arguments={'rviz_config': rviz_config_file}.items())

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(package_dir, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'robot_model': robot_model,
            'ros2_control': ros2_control,
        }.items()
    )

    joint_state_publisher_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'source_list': ['/lift/joint_states'],
            'rate': 30,
            # Default lift to the top of its travel (URDF upper limit) so the
            # preview shows the homed pose before any /lift/joint_states msg.
            'zeros': {'lift_joint': 0.7435},
        }],
    )

    ld = LaunchDescription()
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(joint_state_publisher_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(rviz_cmd)

    return ld
