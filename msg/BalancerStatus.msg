std_msgs/Header header

# Velocity command
float32 cmd_lin_x
float32 cmd_ang_z

# Projected gravity in base_link (from EKF quat + imu→base TF)
float32 grav_x
float32 grav_y
float32 grav_z

# Angular velocity in base_link (rad/s)
float32 omega_x
float32 omega_y
float32 omega_z

# Body-frame velocity from odometry (m/s)
float32 odom_vx

# Wheel velocities (rad/s)
float32 wheel_left_rad_s
float32 wheel_right_rad_s

# Previous filtered action feedback (rad/s)
float32 act_fb_left_rad_s
float32 act_fb_right_rad_s

# Velocity error integrals
float32 integ_lin
float32 integ_ang

# Linear acceleration finite-diff (m/s²)
float32 lin_accel_x

# Raw policy output [-1, 1]
float32 policy_raw_left
float32 policy_raw_right

# Filtered wheel commands
float32 filtered_left_rad_s
float32 filtered_right_rad_s
float32 filtered_left_rpm
float32 filtered_right_rpm
