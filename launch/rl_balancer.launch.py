from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/home/radxa/projects/robot_ws/src/rl_balancer_node/policy/brob27_05_2026.onnx',
        description='Absolute path to the ONNX policy model file'
    )

    return LaunchDescription([
        model_path_arg,
        Node(
            package='rl_balancer_node',
            executable='rl_balancer_node',
            name='rl_balancer_node',
            output='screen',
            parameters=[
                {'model_path': LaunchConfiguration('model_path')}
            ]
        )
    ])
