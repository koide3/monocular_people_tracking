#ifndef MONOCULAR_PEOPLE_TRACKING_PERSON_TRACKER_HPP
#define MONOCULAR_PEOPLE_TRACKING_PERSON_TRACKER_HPP

#include <memory>
#include <fstream>
#include <Eigen/Dense>

#include <kkl/math/gaussian.hpp>
#include <monocular_people_tracking/observation.hpp>

namespace kkl {
namespace alg {
  template<typename T, typename System>
  class UnscentedKalmanFilterX;
}
}

namespace monocular_people_tracking {

class TrackSystem;

class PersonTracker {
public:
  using Ptr = std::shared_ptr<PersonTracker>;
  using UnscentedKalmanFilter = kkl::alg::UnscentedKalmanFilterX<float, TrackSystem>;

  PersonTracker(ros::NodeHandle& nh, const std::shared_ptr<TrackSystem>& track_system, const ros::Time& stamp, long id, const Eigen::Vector4f& neck_ankle);
  ~PersonTracker();

  void predict(const ros::Time& stamp);
  void correct(const ros::Time& stamp, const Observation::Ptr& observation);

  Observation::Ptr get_last_associated() const { return last_associated; }

  double prob(const Eigen::Vector4f& x) const {
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> dist = expected_measurement_distribution();
    return kkl::math::gaussianProbMul<float, 4>(dist.first, dist.second, x);
  }

  double squared_mahalanobis_distance(const Eigen::Vector4f& x) const {
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> dist = expected_measurement_distribution();
    return kkl::math::squaredMahalanobisDistance<float, 4>(dist.first, dist.second, x);
  }

  double prob(const Eigen::Vector2f& x) const {
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> dist = expected_measurement_distribution();
    Eigen::Vector2f mean = dist.first.head<2>();
    Eigen::Matrix2f cov = dist.second.block<2, 2>(0, 0);
    return kkl::math::gaussianProbMul<float, 2>(mean, cov, x);
  }

  double squared_mahalanobis_distance(const Eigen::Vector2f& x) const {
    std::pair<Eigen::VectorXf, Eigen::MatrixXf> dist = expected_measurement_distribution();
    Eigen::Vector2f mean = dist.first.head<2>();
    Eigen::Matrix2f cov = dist.second.block<2, 2>(0, 0);
    return kkl::math::squaredMahalanobisDistance<float, 2>(mean, cov, x);
  }

  std::pair<Eigen::VectorXf, Eigen::MatrixXf> expected_measurement_distribution() const;

  long id() const {
    return id_;
  }

  double trace() const;
  Eigen::Vector3f pos() const;
  Eigen::Vector2f vel() const;
  Eigen::MatrixXf cov() const;

  long correction_count() const {
    return correction_count_;
  }

  bool is_valid() const {
    return correction_count() > validation_correction_count;

  }

private:
  Eigen::Vector3f estimate_init_state(const std::shared_ptr<TrackSystem>& track_system, const Eigen::Vector4f& neck_ankle) const;

private:
  long id_;
  long correction_count_;
  long validation_correction_count;
  ros::Time prev_stamp;

  Observation::Ptr last_associated;
  std::unique_ptr<UnscentedKalmanFilter> ukf;
  mutable boost::optional<std::pair<Eigen::VectorXf, Eigen::MatrixXf>> expected_measurement_dist;
};

}

#endif // PERSON_TRACKER_HPP
