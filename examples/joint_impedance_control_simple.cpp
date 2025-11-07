// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

#include "examples_common.h"

namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}

/**
 * @example joint_impedance_control_simple.cpp
 * A simplified example showing pure joint-space impedance control without model compensation
 * or Cartesian trajectory generation. The robot moves smoothly between two joint configurations
 * (all 7 joints) using a sinusoidal interpolation. The controller applies only PD control 
 * without coriolis or gravity compensation.
 */

// ========================================
// CONFIGURATION PARAMETERS
// ========================================

// Initial robot configuration (goal position to move to first)
constexpr std::array<double, 7> kInitialPosition = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};

// Joint offsets from initial position for the second configuration
// The robot will move between initial_position and initial_position + offsets
constexpr std::array<double, 7> kConfigurationOffsets = {{0.3, -0.2, 0.25, -0.3, 0.2, 0.15, 0.2}};  // radians

// Trajectory parameters
constexpr double kRunTime = 20.0;  // seconds
constexpr double kFrequency = 0.25;  // Hz (0.25 Hz = 4 seconds per cycle)

// Control gains - Joint impedance control (PD controller)
// Stiffness [Nm/rad] for each joint
constexpr std::array<double, 7> kStiffness = {{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0}};
// Damping [Nm*s/rad] for each joint
constexpr std::array<double, 7> kDamping = {{50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0}};

// Collision behavior thresholds
// Lower torque thresholds [Nm] - contact detection
constexpr std::array<double, 7> kLowerTorqueThresholds = {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}};
// Upper torque thresholds [Nm] - contact detection
constexpr std::array<double, 7> kUpperTorqueThresholds = {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}};
// Lower force thresholds [N] - contact detection
constexpr std::array<double, 6> kLowerForceThresholds = {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}};
// Upper force thresholds [N] - contact detection
constexpr std::array<double, 6> kUpperForceThresholds = {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}};

// Print rate for console output
constexpr double kPrintRate = 10.0;  // Hz

// ========================================
// MAIN PROGRAM
// ========================================

int main(int argc, char** argv) {
  // Check whether the required arguments were passed.
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  double time = 0.0;

  // Initialize data fields for the print thread.
  struct {
    std::mutex mutex;
    bool has_data;
    std::array<double, 7> tau_d_last;
    std::array<double, 7> q_ref;
    franka::RobotState robot_state;
    uint64_t loop_counter;
    double elapsed_time;
  } print_data{};
  std::atomic_bool running{true};

  // Start print thread.
  std::thread print_thread([&print_data, &running]() {
    while (running) {
      // Sleep to achieve the desired print rate.
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int>((1.0 / kPrintRate * 1000.0))));

      // Try to lock data to avoid read write collisions.
      if (print_data.mutex.try_lock()) {
        if (print_data.has_data) {
          std::array<double, 7> q_error{};
          double error_rms(0.0);
          for (size_t i = 0; i < 7; ++i) {
            q_error[i] = print_data.q_ref[i] - print_data.robot_state.q[i];
            error_rms += std::pow(q_error[i], 2.0) / q_error.size();
          }
          error_rms = std::sqrt(error_rms);

          // Calculate actual control frequency
          double control_frequency = 0.0;
          if (print_data.elapsed_time > 0.0) {
            control_frequency = print_data.loop_counter / print_data.elapsed_time;
          }

          // Print data to console
          std::cout << "Control frequency [Hz]: " << control_frequency << std::endl
                    << "Command success rate: " << print_data.robot_state.control_command_success_rate << std::endl
                    << "Robot mode: " << static_cast<int>(print_data.robot_state.robot_mode) << std::endl
                    << "Time [s]: " << print_data.elapsed_time << std::endl
                    << "Loop count: " << print_data.loop_counter << std::endl
                    << "q_error [rad]: " << q_error << std::endl
                    << "tau_commanded [Nm]: " << print_data.tau_d_last << std::endl
                    << "tau_measured [Nm]: " << print_data.robot_state.tau_J << std::endl
                    << "root mean square of q_error [rad]: " << error_rms << std::endl
                    << "-----------------------" << std::endl;
          print_data.has_data = false;
        }
        print_data.mutex.unlock();
      }
    }
  });

  try {
    // Connect to robot.
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    setDefaultBehavior(robot);

    // First move the robot to a suitable joint configuration
    MotionGenerator motion_generator(0.5, kInitialPosition);
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);
    std::cout << "Finished moving to initial joint configuration." << std::endl;

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior(
        kLowerTorqueThresholds, kUpperTorqueThresholds,
        kLowerTorqueThresholds, kUpperTorqueThresholds,
        kLowerForceThresholds, kUpperForceThresholds,
        kLowerForceThresholds, kUpperForceThresholds);

    // Store initial joint configuration as the center of our trajectory
    std::array<double, 7> q_initial;
    bool first_iteration = true;
    uint64_t loop_counter = 0;

    // Define callback for the joint torque control loop.
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
        impedance_control_callback =
            [&print_data, &time, &running, &q_initial, &first_iteration, &loop_counter](
                const franka::RobotState& state, franka::Duration period) -> franka::Torques {
      // Update time and loop counter
      time += period.toSec();
      loop_counter++;

      // Initialize on first iteration
      if (first_iteration) {
        q_initial = state.q;
        first_iteration = false;
      }

      // Generate joint-space reference trajectory that moves between two configurations
      // Use smooth sinusoidal interpolation for all 7 joints
      std::array<double, 7> q_ref;
      if (time < kRunTime) {
        // Interpolation factor: 0.0 to 1.0 and back, creates smooth back-and-forth motion
        double interpolation_factor = 0.5 * (1.0 - std::cos(2.0 * M_PI * kFrequency * time));
        
        // Interpolate all joints between initial configuration and target configuration
        for (size_t i = 0; i < 7; i++) {
          q_ref[i] = q_initial[i] + kConfigurationOffsets[i] * interpolation_factor;
        }
      } else {
        // After run_time, return to initial configuration
        q_ref = q_initial;
      }

      // Compute torque command from pure joint impedance control law (PD control).
      // No model compensation terms (no coriolis, no gravity compensation).
      // Uses our own q_ref instead of state.q_d from inverse kinematics.
      std::array<double, 7> tau_d_calculated;
      for (size_t i = 0; i < 7; i++) {
        tau_d_calculated[i] = kStiffness[i] * (q_ref[i] - state.q[i]) - kDamping[i] * state.dq[i];
      }

      // Apply rate limiting for safety
      std::array<double, 7> tau_d_rate_limited =
          franka::limitRate(franka::kMaxTorqueRate, tau_d_calculated, state.tau_J_d);

      // Update data to print.
      if (print_data.mutex.try_lock()) {
        print_data.has_data = true;
        print_data.robot_state = state;
        print_data.tau_d_last = tau_d_rate_limited;
        print_data.q_ref = q_ref;
        print_data.loop_counter = loop_counter;
        print_data.elapsed_time = time;
        print_data.mutex.unlock();
      }

      // Check if motion is finished
      if (time >= kRunTime) {
        running = false;
        return franka::MotionFinished(franka::Torques(tau_d_rate_limited));
      }

      // Send torque command.
      return franka::Torques(tau_d_rate_limited);
    };

    // Start real-time control loop with only torque callback (no Cartesian pose callback).
    robot.control(impedance_control_callback);

  } catch (const franka::Exception& ex) {
    running = false;
    std::cerr << ex.what() << std::endl;
  }

  if (print_thread.joinable()) {
    print_thread.join();
  }
  return 0;
}

