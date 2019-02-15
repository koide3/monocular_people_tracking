#ifndef MONOCULAR_PEOPLE_TRACKING_ASSOCIATION_DISTANCE_HPP
#define MONOCULAR_PEOPLE_TRACKING_ASSOCIATION_DISTANCE_HPP

#include <memory>
#include <vector>
#include <boost/optional.hpp>

#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <monocular_people_tracking/person_tracker.hpp>


namespace monocular_people_tracking {

class AssociationDistance {
public:
  AssociationDistance(ros::NodeHandle& private_nh)
      : maha_sq_thresh(private_nh.param<double>("association_maha_sq_thresh", 9.0)),
        neck_ankle_max_dist(private_nh.param<int>("association_neck_ankle_max_dist", 200)),
        neck_max_dist(private_nh.param<int>("association_neck_max_dist", 100))
  {
  }

  boost::optional<double> operator() (const PersonTracker::Ptr& tracker, const Observation::Ptr& observation) const {
      auto expected_measurement = tracker->expected_measurement_distribution();
      Eigen::Vector2f expected_neck = expected_measurement.first.head<2>();
      Eigen::Vector2f expected_ankle = expected_measurement.first.tail<2>();
      if(expected_neck.x() < 100.0 || expected_ankle.x() < 100.0) {
          return boost::none;
      }

      if(observation->ankle) {
        double distance = (expected_measurement.first - observation->neck_ankle_vector()).norm();
        if(!observation->min_distance){
          observation->min_distance = distance;
        } else {
          observation->min_distance = std::min(distance, *observation->min_distance);
        }

        if(distance > neck_ankle_max_dist) {
          return boost::none;
        }

        double sq_maha = tracker->squared_mahalanobis_distance(observation->neck_ankle_vector());
        if(sq_maha > maha_sq_thresh) {
          return boost::none;
        }
        return -tracker->prob(observation->neck_ankle_vector());
      }

      double distance = (expected_measurement.first.head<2>() - observation->neck_vector()).norm();
      if(distance > neck_max_dist) {
          return boost::none;
      }

      double sq_maha = tracker->squared_mahalanobis_distance(observation->neck_vector());
      if(sq_maha > maha_sq_thresh) {
        return boost::none;
      }

      return -tracker->prob(observation->neck_vector()) + 1.0;
  }

private:
  double maha_sq_thresh;
  int neck_ankle_max_dist;
  int neck_max_dist;
};

}

#endif
