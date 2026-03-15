/**
 * rl_balancer_node — RL policy inference for a 2-wheel self-balancing robot.
 *
 * Observation vector (16-dim, order must match training):
 *   [0..2]   velocity_cmd        — from /cmd_vel (Twist)
 *   [3..5]   projected_gravity   — from /imu/gravity (ImuGravity)
 *   [6..8]   angular_velocity    — from /imu/filtered (ImuFiltered.gx,gy,gz)
 *   [9..11]  linear_velocity     — from /odom (Odometry twist.linear)
 *   [12..13] wheel_velocity      — from /raw_odom (RPM → rad/s × 0.1)
 *   [14..15] previous_actions    — buffered (raw action / 12.0)
 *
 * Action output (2-dim):
 *   Raw actions in [-1, 1] → scaled to [-12, 12] rad/s
 *   Then converted to RPM: rpm = rad_s * 60 / (2π)  ≈ rad_s * 9.5493
 *   Published to /wheel_vel (WheelVel int16 left_rpm, right_rpm).
 */

#include <rclcpp/rclcpp.hpp>
#include <onnxruntime_cxx_api.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "imu_driver_node/msg/imu_gravity.hpp"
#include "imu_driver_node/msg/imu_filtered.hpp"
#include "ddsm115_driver/msg/raw_odom.hpp"
#include "ddsm115_driver/msg/wheel_vel.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono_literals;

// RPM <-> rad/s helpers
static constexpr double RPM_TO_RAD_S = 2.0 * M_PI / 60.0;
static constexpr double RAD_S_TO_RPM = 60.0 / (2.0 * M_PI);
// Policy constants
static constexpr double ACTION_SCALE = 12.0;          // maps [-1,1] → [-12,12] rad/s
static constexpr double WHEEL_VEL_OBS_SCALE = 0.1;    // policy sees rad/s * 0.1
static constexpr double PREV_ACTION_OBS_SCALE = 1.0 / 12.0;
static constexpr int    OBS_DIM = 16;
static constexpr int    ACT_DIM = 2;

class RLBalancerNode : public rclcpp::Node {
public:
  RLBalancerNode()
  : Node("rl_balancer_node")
  {
    // ---------- parameters ----------
    this->declare_parameter("model_path", "");
    std::string model_path = this->get_parameter("model_path").as_string();
    if (model_path.empty()) {
      RCLCPP_ERROR(get_logger(), "model_path parameter is required!");
      throw std::runtime_error("model_path not set");
    }
    RCLCPP_INFO(get_logger(), "Loading ONNX model: %s", model_path.c_str());

    // ---------- ONNX Runtime ----------
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rl_balancer");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);
    RCLCPP_INFO(get_logger(), "ONNX session created successfully");

    // ---------- subscribers ----------
    sub_cmd_vel_ = create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10,
        [this](geometry_msgs::msg::Twist::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          cmd_vel_ = *msg;
        });

    sub_gravity_ = create_subscription<imu_driver_node::msg::ImuGravity>(
        "/imu/gravity", 10,
        [this](imu_driver_node::msg::ImuGravity::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          gravity_ = *msg;
          has_gravity_ = true;
        });

    sub_imu_filtered_ = create_subscription<imu_driver_node::msg::ImuFiltered>(
        "/imu/filtered", 10,
        [this](imu_driver_node::msg::ImuFiltered::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          imu_filt_ = *msg;
          has_imu_ = true;
        });

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        [this](nav_msgs::msg::Odometry::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          odom_ = *msg;
          has_odom_ = true;
        });

    sub_raw_odom_ = create_subscription<ddsm115_driver::msg::RawOdom>(
        "/raw_odom", 10,
        [this](ddsm115_driver::msg::RawOdom::SharedPtr msg) {
          std::lock_guard<std::mutex> lk(mtx_);
          raw_odom_ = *msg;
          has_raw_odom_ = true;
        });

    // ---------- publisher ----------
    pub_wheel_vel_ = create_publisher<ddsm115_driver::msg::WheelVel>("wheel_vel", 10);

    // ---------- TF: require imu→base_link before starting ----------
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    this->declare_parameter("tf_timeout_sec", 10.0);
    double tf_timeout = this->get_parameter("tf_timeout_sec").as_double();
    RCLCPP_INFO(get_logger(), "Waiting up to %.1fs for imu→base_link TF...", tf_timeout);
    try {
      imu_to_base_ = tf_buffer_->lookupTransform(
          "base_link", "imu",
          tf2::TimePointZero,
          tf2::durationFromSec(tf_timeout));
      RCLCPP_INFO(get_logger(), "imu→base_link TF acquired.");
    } catch (const tf2::TransformException & e) {
      RCLCPP_FATAL(get_logger(), "imu→base_link TF not available within %.1fs: %s", tf_timeout, e.what());
      throw std::runtime_error("Required TF transform imu→base_link not available");
    }

    // ---------- 100 Hz control timer ----------
    timer_ = create_wall_timer(10ms, std::bind(&RLBalancerNode::control_loop, this));

    RCLCPP_INFO(get_logger(), "rl_balancer_node ready (100 Hz).");
  }

private:
  void control_loop() {
    // Wait until we have at least gravity & IMU data
    if (!has_gravity_ || !has_imu_) {
      return;
    }

    // ---- Build observation vector ----
    std::array<float, OBS_DIM> obs{};

    {
      std::lock_guard<std::mutex> lk(mtx_);

      // [0..2] velocity_cmd
      obs[0] = static_cast<float>(cmd_vel_.linear.x);
      obs[1] = static_cast<float>(cmd_vel_.linear.y);
      obs[2] = static_cast<float>(cmd_vel_.angular.z);

      // [3..5] projected_gravity — rotated from imu frame to base_link via TF
      {
        const auto & r = imu_to_base_.transform.rotation;
        const tf2::Quaternion q(r.x, r.y, r.z, r.w);
        const tf2::Vector3 g_imu(gravity_.x, gravity_.y, gravity_.z);
        const tf2::Vector3 g_base = tf2::quatRotate(q, g_imu);
        obs[3] = static_cast<float>(g_base.x());
        obs[4] = static_cast<float>(g_base.y());
        obs[5] = static_cast<float>(g_base.z());
      }

      // [6..8] angular_velocity (gyro, filtered)
      obs[6] = imu_filt_.gx;
      obs[7] = imu_filt_.gy;
      obs[8] = imu_filt_.gz;

      // [9..11] linear_velocity from odometry
      if (has_odom_) {
        obs[9]  = static_cast<float>(odom_.twist.twist.linear.x);
        obs[10] = static_cast<float>(odom_.twist.twist.linear.y);
        obs[11] = static_cast<float>(odom_.twist.twist.linear.z);
      }

      // [12..13] wheel_velocity (RPM → rad/s → ×0.1)
      if (has_raw_odom_) {
        obs[12] = static_cast<float>(raw_odom_.left_actual_rpm  * RPM_TO_RAD_S * WHEEL_VEL_OBS_SCALE);
        obs[13] = static_cast<float>(raw_odom_.right_actual_rpm * RPM_TO_RAD_S * WHEEL_VEL_OBS_SCALE);
      }

      // [14..15] previous actions (normalised by 1/12)
      obs[14] = prev_actions_[0];
      obs[15] = prev_actions_[1];
    }

    // ---- Run ONNX inference ----
    std::array<int64_t, 2> input_shape = {1, OBS_DIM};
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, obs.data(), obs.size(), input_shape.data(), input_shape.size());

    const char* input_name  = "obs";
    const char* output_name = "actions";
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        &input_name,  &input_tensor,  1,
        &output_name, 1);

    // ---- Read actions ----
    float* action_data = output_tensors[0].GetTensorMutableData<float>();
    float act_left  = std::clamp(action_data[0], -1.0f, 1.0f);
    float act_right = std::clamp(action_data[1], -1.0f, 1.0f);

    // Store normalised previous actions for next frame
    prev_actions_[0] = act_left  * static_cast<float>(PREV_ACTION_OBS_SCALE * ACTION_SCALE);
    prev_actions_[1] = act_right * static_cast<float>(PREV_ACTION_OBS_SCALE * ACTION_SCALE);

    // Scale to rad/s, then to RPM
    double left_rad_s  = act_left  * ACTION_SCALE;
    double right_rad_s = act_right * ACTION_SCALE;
    int16_t left_rpm   = static_cast<int16_t>(std::round(left_rad_s  * RAD_S_TO_RPM));
    int16_t right_rpm  = static_cast<int16_t>(std::round(right_rad_s * RAD_S_TO_RPM));

    // ---- Publish ----
    ddsm115_driver::msg::WheelVel cmd;
    cmd.left_rpm  = left_rpm;
    cmd.right_rpm = right_rpm;
    pub_wheel_vel_->publish(cmd);
  }

  // ---- TF ----
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  geometry_msgs::msg::TransformStamped        imu_to_base_;

  // ---- ONNX ----
  std::unique_ptr<Ort::Env>     env_;
  std::unique_ptr<Ort::Session> session_;

  // ---- ROS I/O ----
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr           sub_cmd_vel_;
  rclcpp::Subscription<imu_driver_node::msg::ImuGravity>::SharedPtr   sub_gravity_;
  rclcpp::Subscription<imu_driver_node::msg::ImuFiltered>::SharedPtr  sub_imu_filtered_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr            sub_odom_;
  rclcpp::Subscription<ddsm115_driver::msg::RawOdom>::SharedPtr       sub_raw_odom_;
  rclcpp::Publisher<ddsm115_driver::msg::WheelVel>::SharedPtr         pub_wheel_vel_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---- Latest data (guarded by mtx_) ----
  std::mutex mtx_;
  geometry_msgs::msg::Twist          cmd_vel_{};
  imu_driver_node::msg::ImuGravity   gravity_{};
  imu_driver_node::msg::ImuFiltered  imu_filt_{};
  nav_msgs::msg::Odometry            odom_{};
  ddsm115_driver::msg::RawOdom       raw_odom_{};

  bool has_gravity_  = false;
  bool has_imu_      = false;
  bool has_odom_     = false;
  bool has_raw_odom_ = false;

  // ---- State ----
  std::array<float, 2> prev_actions_ = {0.0f, 0.0f};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RLBalancerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
