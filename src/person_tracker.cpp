#include <monocular_people_tracking/person_tracker.hpp>

#include <cppoptlib/problem.h>
#include <cppoptlib/solver/neldermeadsolver.h>

#include <kkl/alg/unscented_kalman_filter.hpp>
#include <monocular_people_tracking/track_system.hpp>


namespace monocular_people_tracking {

PersonTracker::PersonTracker(ros::NodeHandle& nh, const std::shared_ptr<TrackSystem>& track_system, const ros::Time& stamp, long id, const Eigen::Vector4f& neck_ankle)
  : id_(id)
{
  validation_correction_count = nh.param<int>("validation_correction_cound", 5);

  Eigen::VectorXf mean = Eigen::VectorXf::Zero(5);
  // mean.head<3>() = track_system->transform_footprint2odom(Eigen::Vector3f(2.0f, 0.0f, 1.7f));
  mean.head<3>() = estimate_init_state(track_system, neck_ankle);
  mean[2] = 1.6f;

  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(5, 5) * nh.param<double>("init_cov_scale", 1.0);
  cov(2, 2) = 1e-1f;

  ukf.reset(new UnscentedKalmanFilter(track_system, mean, cov));

  prev_stamp = stamp;
  correction_count_ = 0;
  last_associated = nullptr;
}

PersonTracker::~PersonTracker() {

}

void PersonTracker::predict(const ros::Time& stamp) {
  expected_measurement_dist = boost::none;
  last_associated = nullptr;

  ukf->system->set_dt((stamp - prev_stamp).toSec());
  prev_stamp = stamp;

  ukf->predict(Eigen::Vector2f::Zero());
}

void PersonTracker::correct(const ros::Time& stamp, const Observation::Ptr& observation) {
  expected_measurement_dist = boost::none;
  last_associated = observation;
  correction_count_ ++;

  if(observation->ankle) {
    ukf->correct(observation->neck_ankle_vector());
  } else {
    ukf->correct(observation->neck_vector());
  }
}


std::pair<Eigen::VectorXf, Eigen::MatrixXf> PersonTracker::expected_measurement_distribution() const {
  if(!expected_measurement_dist) {
    expected_measurement_dist = ukf->expected_measurement_distribution<Eigen::Vector4f>();
  }

  return *expected_measurement_dist;
}

double PersonTracker::trace() const {
  return ukf->cov.trace();
}

Eigen::Vector3f PersonTracker::pos() const {
  return ukf->mean.head<3>();
}

Eigen::Vector2f PersonTracker::vel() const {
  return ukf->mean.tail<2>();
}

Eigen::MatrixXf PersonTracker::cov() const {
  return ukf->cov;
}


class Projection : public cppoptlib::Problem<float> {
public:
  Projection(const Eigen::Matrix3f& camera_matrix, const Eigen::Isometry3f& footprint2camera, const Eigen::Vector4f& observation)
    : proj(camera_matrix * footprint2camera.matrix().block<3, 4>(0, 0)),
      observation(observation)
  {}

  float value(const Eigen::VectorXf& x) {
    return error(x, observation);
  }

  void gradient(const Eigen::VectorXf& x, Eigen::VectorXf& grad) {
    grad = grad_error(x, observation);
  }

private:
  Eigen::Vector2f project(const Eigen::Vector3f& x_) const {
    Eigen::Vector4f x(x_[0], x_[1], x_[2], 1.0f);
    Eigen::Vector3f uvs = proj * x;
    return uvs.head<2>() / uvs[2];
  }

  Eigen::Matrix<float, 2, 3> grad_project(const Eigen::Vector3f& x_, bool fixed_z = false) const {
    Eigen::Matrix<float, 3, 4> P = proj;
    if(fixed_z) {
      P.col(2).setZero();
    }

    Eigen::Vector4f x(x_[0], x_[1], x_[2], 1.0f);
    Eigen::Vector3f l = P * x;

    auto lhs = P.block<2, 3>(0, 0) * l[2];
    auto rhs = l.head<2>() * P.block<1, 3>(2, 0);
    auto grad = (lhs - rhs) / (l[2] * l[2]);

    return grad;
  }

  float error(const Eigen::Vector3f& x, const Eigen::Vector4f& y) const {
    Eigen::Vector3f x1 = x;
    Eigen::Vector2f proj1 = project(x1);
    Eigen::Vector3f x2(x[0], x[1], 0.0f);
    Eigen::Vector2f proj2 = project(x2);

    Eigen::Vector4f estimated(proj1[0], proj1[1], proj2[0], proj2[1]);

    return (estimated - y).squaredNorm();
  }

  Eigen::Vector3f grad_error(const Eigen::Vector3f& x, const Eigen::Vector4f& y) const {
    Eigen::Vector4f projs;
    Eigen::Matrix<float, 4, 3> grads;

    Eigen::Vector3f x1 = x;
    projs.head<2>() = project(x1);
    grads.block<2, 3>(0, 0) = grad_project(x1);

    Eigen::Vector3f x2(x[0], x[1], 0.0f);
    projs.tail<2>() = project(x2);
    grads.block<2, 3>(2, 0) = grad_project(x2, true);

    Eigen::Vector4f errors = 2 * (projs - y);

    return grads.transpose() * errors;
  }

private:
  Eigen::Matrix<float, 3, 4> proj;
  Eigen::Vector4f observation;
};

Eigen::Vector3f PersonTracker::estimate_init_state(const std::shared_ptr<TrackSystem>& track_system, const Eigen::Vector4f& neck_ankle) const {
  Projection f(track_system->camera_matrix, track_system->footprint2camera, neck_ankle);

  Eigen::VectorXf x0 = Eigen::Vector3f(2.0f, 0.0f, 1.6f);
  cppoptlib::NelderMeadSolver<Projection> solver;
  solver.minimize(f, x0);

  return track_system->transform_footprint2odom(x0);
}

}
