#ifndef MONOCULAR_PEOPLE_TRACKING_TRACKSYSTEM_HPP
#define MONOCULAR_PEOPLE_TRACKING_TRACKSYSTEM_HPP

#include <Eigen/Dense>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/CameraInfo.h>

namespace monocular_people_tracking {

class TrackSystem {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TrackSystem(const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
    : camera_frame_id(camera_frame_id),
      tf_listener(tf_listener)
  {
    dt = 0.1;

    measurement_noise = Eigen::Matrix4f::Identity() * 10 * 10;

    camera_matrix = Eigen::Map<const Eigen::Matrix3d>(camera_info_msg->K.data()).transpose().cast<float>();
    update_matrices(ros::Time(0));
  }

  void update_matrices(const ros::Time& stamp) {
    odom2camera = lookup_eigen(camera_frame_id, "odom", stamp);
    odom2footprint = lookup_eigen("base_footprint", "odom", stamp);
    footprint2base = lookup_eigen("base_link", "base_footprint", stamp);
    footprint2camera = lookup_eigen(camera_frame_id, "base_footprint", stamp);
  }

  Eigen::Isometry3f lookup_eigen(const std::string& to, const std::string& from, const ros::Time& stamp) {
    tf::StampedTransform transform;
    try{
      tf_listener->waitForTransform(to, from, stamp, ros::Duration(1.0));
      tf_listener->lookupTransform(to, from, stamp, transform);
    } catch (tf::ExtrapolationException& e) {
      tf_listener->waitForTransform(to, from, ros::Time(0), ros::Duration(5.0));
      tf_listener->lookupTransform(to, from, ros::Time(0), transform);
    }

    Eigen::Isometry3d iso;
    tf::transformTFToEigen(transform, iso);
    return iso.cast<float>();
  }

  Eigen::Vector3f transform_odom2camera(const Eigen::Vector3f& pos_in_odom) const {
    return (odom2camera * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
  }

  Eigen::Vector3f transform_odom2footprint(const Eigen::Vector3f& pos_in_odom) const {
    return (odom2footprint * Eigen::Vector4f(pos_in_odom.x(), pos_in_odom.y(), pos_in_odom.z(), 1.0f)).head<3>();
  }

  Eigen::Vector3f transform_footprint2odom(const Eigen::Vector3f& pos_in_footprint) const {
    return (odom2footprint.inverse() * Eigen::Vector4f(pos_in_footprint.x(), pos_in_footprint.y(), pos_in_footprint.z(), 1.0f)).head<3>();
  }

  void set_dt(double d) {
    dt = std::max(d, 1e-9);
  }


  // interface for UKF
  Eigen::VectorXf f(const Eigen::VectorXf& state, const Eigen::VectorXf& control) const {
    Eigen::VectorXf next_state = state;
    next_state.middleRows(0, 2) += dt * state.middleRows(3, 2);
    return next_state;
  }

  Eigen::MatrixXf processNoiseCov() const {
    Eigen::MatrixXf process_noise = Eigen::MatrixXf::Identity(5, 5);
    process_noise.middleRows(0, 2) *= 1e-1;
    process_noise(2, 2) = 1e-10;
    process_noise.middleRows(3, 2) *= 1e-1;

    return process_noise;
  }

  template<typename Measurement>
  Measurement h(const Eigen::VectorXf& state) const;

  template<typename Measurement>
  Eigen::MatrixXf measurementNoiseCov() const;

public:
  double dt;

  Eigen::Isometry3f odom2camera;
  Eigen::Isometry3f odom2footprint;
  Eigen::Isometry3f footprint2base;
  Eigen::Isometry3f footprint2camera;
  Eigen::Matrix3f camera_matrix;

  std::string camera_frame_id;
  std::shared_ptr<tf::TransformListener> tf_listener;

  Eigen::Matrix4f measurement_noise;
};

}

#endif // TRACKSYSTEM_CPP
