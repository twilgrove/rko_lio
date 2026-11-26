from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=PathJoinSubstitution([FindPackageShare("rko_lio"), "config", "rko_lio.yaml"]),
        ),
        Node(
            package="rko_lio",
            executable="online_node",
            name="rko_lio_node",
            output="screen",
            parameters=[LaunchConfiguration("config_file")],
        ),
    ])
