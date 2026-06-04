/**
 * rl_balancer_node — RL policy inference for a 2-wheel self-balancing robot.
 *
 * Observation vector: 105-dim = 21 dims × 5 history frames, newest-first.
 * Per-frame layout (UPDATING_ROS2_NODE.md §1):
 *   [0]      cmd_lin_x         /cmd_vel linear.x
 *   [1]      cmd_lin_y         always 0.0
 *   [2]      cmd_ang_z         /cmd_vel angular.z
 *   [3..5]   projected_gravity computed from /imu quaternion, rotated imu→base_link
 *   [6..8]   angular_velocity  /imu angular_velocity rotated imu→base_link
 *   [9..11]  linear_velocity   /odom twist.linear (body-frame, y/z=0)
 *   [12..13] wheel_vel*0.1     /raw_odom RPM→rad/s×0.1
 *   [14..15] prev_filt÷12      post-filter action÷WHEEL_SCALE
 *   [16..17] vel_error_integ   leaky integrals [lin_x_I, ang_z_I], decay 0.93
 *   [18..20] lin_accel*0.1     finite-diff body-frame vel×0.1
 *
 * Action: raw∈[-1,1] → deadband→slew→clip→LPF → rad/s → RPM → /wheel_vel
 */

#include <rclcpp/rclcpp.hpp>
#include <onnxruntime_cxx_api.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include "rl_balancer_node/msg/balancer_status.hpp"
#include "ddsm115_driver/msg/raw_odom.hpp"
#include "ddsm115_driver/msg/wheel_vel.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

using namespace std::chrono_literals;

static constexpr float RPM_TO_RAD_S = 2.0f * static_cast<float>(M_PI) / 60.0f;
static constexpr float RAD_S_TO_RPM = 60.0f / (2.0f * static_cast<float>(M_PI));

// Policy contract constants — must match training (UPDATING_ROS2_NODE.md §0)
static constexpr int   PER_FRAME_OBS  = 21;
static constexpr int   HISTORY_LEN    = 5;
static constexpr int   OBS_DIM        = PER_FRAME_OBS * HISTORY_LEN;  // 105
static constexpr int   ACT_DIM        = 2;
static constexpr float DT             = 0.01f;
static constexpr float WHEEL_SCALE    = 12.0f;
static constexpr float INTEGRAL_DECAY = 0.93f;
static constexpr float DEADBAND       = 0.05f;
static constexpr float MAX_SLEW       = 1.5f;
static constexpr float CLIP           = 12.0f;
static constexpr float LPF_ALPHA      = 1.0f / 3.0f;

class RLBalancerNode : public rclcpp::Node {
public:
  explicit RLBalancerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("rl_balancer_node", options)
  {
    this->declare_parameter("model_path", "");
    std::string model_path = this->get_parameter("model_path").as_string();
    if (model_path.empty()) {
      RCLCPP_ERROR(get_logger(), "model_path parameter is required!");
      throw std::runtime_error("model_path not set");
    }
    RCLCPP_INFO(get_logger(), "Loading ONNX model: %s", model_path.c_str());

    // ONNX Runtime
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rl_balancer");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);

    // Shape contract check
    {
      auto in_shape  = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
      if (in_shape.back() != OBS_DIM)
        throw std::runtime_error("obs dim mismatch: model=" + std::to_string(in_shape.back())
                                 + " expected=" + std::to_string(OBS_DIM));
      if (out_shape.back() != ACT_DIM)
        throw std::runtime_error("act dim mismatch: model=" + std::to_string(out_shape.back())
                                 + " expected=" + std::to_string(ACT_DIM));
    }
    RCLCPP_INFO(get_logger(), "ONNX session ready: [1,%d]→[1,%d]", OBS_DIM, ACT_DIM);

    obs_buf_.fill(0.0f);
    act_buf_.fill(0.0f);
    mem_info_ = std::make_unique<Ort::MemoryInfo>(
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    // Subscribers
    sub_cmd_vel_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          cmd_vel_ = *msg;
        });

    // Single /imu subscription replaces /imu/gravity + /imu/filtered.
    // sensor_msgs/Imu publishes at full EKF rate (~200 Hz) so gravity and
    // gyro are always fresh relative to the 100 Hz control tick.
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Imu::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          imu_quat_ = {static_cast<float>(msg->orientation.w),
                       static_cast<float>(msg->orientation.x),
                       static_cast<float>(msg->orientation.y),
                       static_cast<float>(msg->orientation.z)};
          imu_gyro_ = {static_cast<float>(msg->angular_velocity.x),
                       static_cast<float>(msg->angular_velocity.y),
                       static_cast<float>(msg->angular_velocity.z)};
          has_imu_ = true;
        });

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odom", rclcpp::SensorDataQoS(),
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          // odometry_node publishes body-frame forward vel in twist.linear.x
          odom_lin_x_ = static_cast<float>(msg->twist.twist.linear.x);
          has_odom_ = true;
        });

    sub_raw_odom_ = create_subscription<ddsm115_driver::msg::RawOdom>(
        "/raw_odom", rclcpp::SensorDataQoS(),
        [this](ddsm115_driver::msg::RawOdom::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          wheel_rad_s_[0] = static_cast<float>(msg->left_actual_rpm)  * RPM_TO_RAD_S;
          wheel_rad_s_[1] = static_cast<float>(msg->right_actual_rpm) * RPM_TO_RAD_S;
          has_raw_odom_ = true;
        });

    pub_wheel_vel_ = create_publisher<ddsm115_driver::msg::WheelVel>("wheel_vel", 10);

    // TF: one-time lookup for the static imu→base_link mount rotation
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    this->declare_parameter("tf_timeout_sec", 10.0);
    double tf_timeout = this->get_parameter("tf_timeout_sec").as_double();
    RCLCPP_INFO(get_logger(), "Waiting up to %.1fs for imu→base_link TF...", tf_timeout);
    try {
      auto t = tf_buffer_->lookupTransform("base_link", "imu",
                                            tf2::TimePointZero,
                                            tf2::durationFromSec(tf_timeout));
      const auto & r = t.transform.rotation;
      q_imu_to_base_ = tf2::Quaternion(r.x, r.y, r.z, r.w);
      RCLCPP_INFO(get_logger(), "imu→base_link TF acquired.");
    } catch (const tf2::TransformException & e) {
      RCLCPP_FATAL(get_logger(), "imu→base_link TF not available: %s", e.what());
      throw std::runtime_error("Required TF not available");
    }

    timer_ = create_wall_timer(10ms, std::bind(&RLBalancerNode::control_loop, this));

    pub_status_ = create_publisher<rl_balancer_node::msg::BalancerStatus>("rl_balancer/status", 10);
    status_timer_ = create_wall_timer(200ms, std::bind(&RLBalancerNode::status_loop, this));

    RCLCPP_INFO(get_logger(), "rl_balancer_node ready (100 Hz, obs_dim=%d).", OBS_DIM);
  }

private:
  // Action filter chain per §3 of UPDATING_ROS2_NODE.md.
  // a[2]:    raw policy output ∈ [-1,1]
  // prev[2]: post-filter rad/s from previous tick
  // out[2]:  filtered rad/s to send to motors
  void apply_action_chain(const float a[ACT_DIM], const float prev[ACT_DIM],
                          float out[ACT_DIM]) {
    for (int i = 0; i < ACT_DIM; ++i) {
      float v = a[i] * WHEEL_SCALE;
      // 1. Direction-change deadband
      if ((prev[i] * v) < 0.0f && std::fabs(v) < DEADBAND) v = 0.0f;
      // 2. Slew rate limit
      float delta = v - prev[i];
      if (delta >  MAX_SLEW) delta =  MAX_SLEW;
      if (delta < -MAX_SLEW) delta = -MAX_SLEW;
      v = prev[i] + delta;
      // 3. Clip to motor limits
      if (v >  CLIP) v =  CLIP;
      if (v < -CLIP) v = -CLIP;
      // 4. 1st-order IIR LPF
      out[i] = LPF_ALPHA * prev[i] + (1.0f - LPF_ALPHA) * v;
    }
  }

  void status_loop() {
    if (!has_imu_) return;
    const float* f = obs_buf_.data();  // slot 0 = most recent frame

    rl_balancer_node::msg::BalancerStatus s;
    s.header.stamp = now();

    s.cmd_lin_x = f[0];
    s.cmd_ang_z = f[2];

    s.grav_x = f[3];
    s.grav_y = f[4];
    s.grav_z = f[5];

    s.omega_x = f[6];
    s.omega_y = f[7];
    s.omega_z = f[8];

    s.odom_vx = f[9];

    s.wheel_left_rad_s  =  f[12] / 0.1f;   // policy convention (positive = forward)
    s.wheel_right_rad_s =  f[13] / 0.1f;   // already negated in frame build

    s.act_fb_left_rad_s  = f[14] * WHEEL_SCALE;
    s.act_fb_right_rad_s = f[15] * WHEEL_SCALE;

    s.integ_lin = f[16];
    s.integ_ang = f[17];

    s.lin_accel_x = f[18] / 0.1f;

    s.policy_raw_left  = act_buf_[0];
    s.policy_raw_right = act_buf_[1];

    s.filtered_left_rad_s  = prev_filt_[0];
    s.filtered_right_rad_s = prev_filt_[1];
    s.filtered_left_rpm  = prev_filt_[0] * RAD_S_TO_RPM;
    s.filtered_right_rpm = prev_filt_[1] * RAD_S_TO_RPM;

    pub_status_->publish(s);
  }

  void control_loop() {
    if (!has_imu_) return;

    // Snapshot shared sensor data under lock; all policy state is private
    float cmd_lin_x, cmd_ang_z;
    std::array<float, 4> imu_quat;
    std::array<float, 3> imu_gyro;
    float odom_vx = 0.0f;
    std::array<float, 2> wrad = {};
    {
      std::lock_guard<std::mutex> lk(mtx_);
      cmd_lin_x = static_cast<float>(cmd_vel_.linear.x);
      cmd_ang_z = static_cast<float>(cmd_vel_.angular.z);
      imu_quat  = imu_quat_;
      imu_gyro  = imu_gyro_;
      if (has_odom_)     odom_vx = odom_lin_x_;
      if (has_raw_odom_) wrad    = wheel_rad_s_;
    }

    // ── Build per-frame observation (21 dims) ─────────────────────────────
    float frame[PER_FRAME_OBS] = {};

    // [0..2] velocity command
    frame[0] = cmd_lin_x;
    frame[1] = 0.0f;
    frame[2] = cmd_ang_z;

    // [3..5] projected gravity in base_link frame (§2.1).
    // Formula rotates world-frame [0,0,-1] into the imu frame using the EKF
    // quaternion, then the static q_imu_to_base_ rotates to base_link.
    {
      const float qw = imu_quat[0], qx = imu_quat[1],
                  qy = imu_quat[2], qz = imu_quat[3];
      const float gx = 2.0f*(qx*qz - qw*qy);
      const float gy = 2.0f*(qy*qz + qw*qx);
      const float gz = qw*qw - qx*qx - qy*qy + qz*qz;
      const tf2::Vector3 g_imu(-gx, -gy, -gz);
      const tf2::Vector3 g_base = tf2::quatRotate(q_imu_to_base_, g_imu);
      frame[3] = static_cast<float>(g_base.x());
      frame[4] = static_cast<float>(g_base.y());
      frame[5] = static_cast<float>(g_base.z());
    }

    // [6..8] angular velocity rotated imu→base_link
    float gyro_z_base;
    {
      const tf2::Vector3 omega_imu(imu_gyro[0], imu_gyro[1], imu_gyro[2]);
      const tf2::Vector3 omega_base = tf2::quatRotate(q_imu_to_base_, omega_imu);
      frame[6] = static_cast<float>(omega_base.x());
      frame[7] = static_cast<float>(omega_base.y());
      frame[8] = static_cast<float>(omega_base.z());
      gyro_z_base = frame[8];
    }

    // [9..11] body-frame linear velocity (odom twist is already body-frame)
    frame[9]  = odom_vx;
    frame[10] = 0.0f;
    frame[11] = 0.0f;

    // [12..13] wheel velocity × 0.1
    frame[12] =  wrad[0] * 0.1f;
    frame[13] = -wrad[1] * 0.1f;  // right motor is physically inverted; negate to match policy convention

    // [14..15] post-filter action feedback ÷ WHEEL_SCALE (§3)
    // Right motor is hardware-inverted: prev_filt_[1] negative = hardware forward.
    // Negate to match policy convention (positive = right wheel forward).
    frame[14] =  prev_filt_[0] / WHEEL_SCALE;
    frame[15] = -prev_filt_[1] / WHEEL_SCALE;

    // [16..17] leaky velocity-error integrals (§2.3); seeded on first tick
    {
      const float lin_err = cmd_lin_x - odom_vx;
      const float ang_err = cmd_ang_z - gyro_z_base;
      if (!integ_init_) {
        lin_I_ = lin_err;
        ang_I_ = ang_err;
        integ_init_ = true;
      } else {
        lin_I_ = INTEGRAL_DECAY * lin_I_ + lin_err * DT;
        ang_I_ = INTEGRAL_DECAY * ang_I_ + ang_err * DT;
      }
    }
    frame[16] = lin_I_;
    frame[17] = ang_I_;

    // [18..20] body-frame linear accel × 0.1 (finite-diff, §2.4); zero on first tick
    if (!accel_init_) {
      accel_init_ = true;
    } else {
      frame[18] = (odom_vx - prev_v_body_x_) / DT * 0.1f;
    }
    prev_v_body_x_ = odom_vx;

    // ── Shift history newest-first and write current frame into slot 0 ────
    // On first tick, replicate the real frame into all history slots so the
    // policy never sees physically impossible zero-gravity history.
    if (!history_init_) {
      for (int h = 0; h < HISTORY_LEN; ++h)
        std::memcpy(obs_buf_.data() + h * PER_FRAME_OBS, frame, sizeof(float) * PER_FRAME_OBS);
      history_init_ = true;
    } else {
      std::memmove(obs_buf_.data() + PER_FRAME_OBS,
                   obs_buf_.data(),
                   sizeof(float) * PER_FRAME_OBS * (HISTORY_LEN - 1));
      std::memcpy(obs_buf_.data(), frame, sizeof(float) * PER_FRAME_OBS);
    }

    // ── ONNX inference (zero-copy tensor wrappers) ────────────────────────
    std::array<int64_t, 2> in_shape  = {1, OBS_DIM};
    std::array<int64_t, 2> out_shape = {1, ACT_DIM};
    auto in_tensor = Ort::Value::CreateTensor<float>(
        *mem_info_, obs_buf_.data(), obs_buf_.size(),
        in_shape.data(), in_shape.size());
    auto out_tensor = Ort::Value::CreateTensor<float>(
        *mem_info_, act_buf_.data(), act_buf_.size(),
        out_shape.data(), out_shape.size());

    const char* in_name  = "obs";
    const char* out_name = "actions";
    try {
      session_->Run(Ort::RunOptions{nullptr},
                    &in_name,  &in_tensor,  1,
                    &out_name, &out_tensor, 1);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "ONNX inference failed: %s", e.what());
      return;
    }

    // ── Action filter chain → publish ─────────────────────────────────────
    float filtered[ACT_DIM];
    apply_action_chain(act_buf_.data(), prev_filt_.data(), filtered);
    prev_filt_[0] = filtered[0];
    prev_filt_[1] = filtered[1];

    ddsm115_driver::msg::WheelVel cmd;
    cmd.left_rpm  = static_cast<int16_t>(std::round(filtered[0] * RAD_S_TO_RPM));
    cmd.right_rpm = static_cast<int16_t>(std::round(filtered[1] * RAD_S_TO_RPM));
    pub_wheel_vel_->publish(cmd);
  }

  // ── TF ──────────────────────────────────────────────────────────────────
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  tf2::Quaternion                             q_imu_to_base_;

  // ── ONNX ────────────────────────────────────────────────────────────────
  std::unique_ptr<Ort::Env>        env_;
  std::unique_ptr<Ort::Session>    session_;
  std::unique_ptr<Ort::MemoryInfo> mem_info_;
  std::array<float, OBS_DIM>       obs_buf_{};   // flat [105], newest-first
  std::array<float, ACT_DIM>       act_buf_{};

  // ── ROS I/O ─────────────────────────────────────────────────────────────
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr      sub_cmd_vel_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr          sub_imu_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        sub_odom_;
  rclcpp::Subscription<ddsm115_driver::msg::RawOdom>::SharedPtr   sub_raw_odom_;
  rclcpp::Publisher<ddsm115_driver::msg::WheelVel>::SharedPtr     pub_wheel_vel_;
  rclcpp::Publisher<rl_balancer_node::msg::BalancerStatus>::SharedPtr pub_status_;
  rclcpp::TimerBase::SharedPtr                                    timer_;
  rclcpp::TimerBase::SharedPtr                                    status_timer_;

  // ── Shared state (guarded by mtx_) ──────────────────────────────────────
  std::mutex             mtx_;
  geometry_msgs::msg::Twist  cmd_vel_{};
  std::array<float, 4>       imu_quat_ = {1.0f, 0.0f, 0.0f, 0.0f};  // w,x,y,z
  std::array<float, 3>       imu_gyro_ = {};
  float                      odom_lin_x_ = 0.0f;
  std::array<float, 2>       wheel_rad_s_ = {};
  bool has_imu_ = false, has_odom_ = false, has_raw_odom_ = false;

  // ── Policy-private state (control loop only, no lock needed) ────────────
  std::array<float, ACT_DIM> prev_filt_    = {};    // post-filter rad/s
  float                      prev_v_body_x_ = 0.0f;
  float                      lin_I_         = 0.0f;
  float                      ang_I_         = 0.0f;
  bool                       integ_init_    = false;
  bool                       accel_init_    = false;
  bool                       history_init_  = false;
};

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(RLBalancerNode)

#ifndef COMPONENT_ONLY
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RLBalancerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
#endif
