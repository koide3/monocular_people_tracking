#include <monocular_people_tracking/track_system.hpp>

namespace monocular_people_tracking {

template<>
Eigen::Vector2f TrackSystem::h(const Eigen::VectorXf& state) const {
  Eigen::Vector4f neck_pos_3d = odom2camera * footprint2base * Eigen::Vector4f(state[0], state[1], state[2], 1.0f);
  Eigen::Vector3f neck_pos_2d = camera_matrix * neck_pos_3d.head<3>();

  Eigen::Vector2f observation;
  observation.head<2>() = neck_pos_2d.head<2>() / neck_pos_2d.z();
  return observation;
}

template<>
Eigen::MatrixXf TrackSystem::measurementNoiseCov<Eigen::Vector2f>() const {
  return measurement_noise.block<2, 2>(0, 0);
}

template<>
Eigen::Vector4f TrackSystem::h(const Eigen::VectorXf& state) const {
   Eigen::Vector4f neck_pos_3d = odom2camera * footprint2base * Eigen::Vector4f(state[0], state[1], state[2], 1.0f);
   Eigen::Vector4f ankle_pos_3d = odom2camera * footprint2base * Eigen::Vector4f(state[0], state[1], 0.0f, 1.0f);

   Eigen::Vector3f neck_pos_2d = camera_matrix * neck_pos_3d.head<3>();
   Eigen::Vector3f ankle_pos_2d = camera_matrix * ankle_pos_3d.head<3>();

   Eigen::Vector4f observation;
   observation.head<2>() = neck_pos_2d.head<2>() / neck_pos_2d.z();
   observation.tail<2>() = ankle_pos_2d.head<2>() / ankle_pos_2d.z();

   return observation;
}

template<>
Eigen::MatrixXf TrackSystem::measurementNoiseCov<Eigen::Vector4f>() const {
  return measurement_noise;
}

}
