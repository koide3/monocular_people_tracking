#ifndef MONOCULAR_PEOPLE_TRACKING_OBSERVATION_HPP
#define MONOCULAR_PEOPLE_TRACKING_OBSERVATION_HPP

#include <memory>
#include <Eigen/Dense>
#include <boost/optional.hpp>

#include <ros/node_handle.h>
#include <sensor_msgs/CameraInfo.h>

namespace monocular_people_tracking {

struct Joint {
public:
    Joint()
    : confidence(0.0),
      x(0.0),
      y(0.0)
    {}

    Joint(float confidence, float x, float y)
      : confidence(confidence),
        x(x),
        y(y)
    {}

    float confidence;
    float x;
    float y;
};

struct Observation {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Observation>;

  Observation(ros::NodeHandle& private_nh, const Joint& neck_, const Joint& lankle, const Joint& rankle, const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
    const double confidence_thresh = private_nh.param<double>("detection_confidence_thresh", 0.3);

    if(neck_.confidence > confidence_thresh) {
      neck = Eigen::Vector2f(neck_.x, neck_.y);
    }

    if(lankle.confidence > confidence_thresh && rankle.confidence > confidence_thresh) {
      ankle = Eigen::Vector2f(lankle.x + rankle.x, lankle.y + rankle.y) / 2.0f;
    } else if (lankle.confidence > confidence_thresh) {
      ankle = Eigen::Vector2f(lankle.x, lankle.y);
    } else if (rankle.confidence > confidence_thresh) {
      ankle = Eigen::Vector2f(rankle.x, rankle.y);
    }

    close2border = false;

    int border_thresh_w = private_nh.param<int>("detection_border_thresh_w", 100);
    int border_thresh_h = private_nh.param<int>("detection_border_thresh_h", 25);;
    if(neck) {
      if(neck->x() < border_thresh_w || neck->x() > camera_info_msg->width - border_thresh_w ||
         neck->y() < border_thresh_h || neck->y() > camera_info_msg->height - border_thresh_h )
      {
        close2border = true;
      }
    }

    if(ankle) {
      if(ankle->x() < border_thresh_w || ankle->x() > camera_info_msg->width - border_thresh_w ||
         ankle->y() < border_thresh_h || ankle->y() > camera_info_msg->height - border_thresh_h )
      {
        close2border = true;
      }
    }
  }

  bool is_valid() const {
    return  static_cast<bool>(neck);
  }

  Eigen::Vector4f neck_ankle_vector() const {
    Eigen::Vector4f x;
    x.head<2>() = *neck;
    x.tail<2>() = *ankle;
    return  x;
  }

  Eigen::Vector2f neck_vector() const {
    return  *neck;
  }

  bool close2border;
  boost::optional<Eigen::Vector2f> neck;
  boost::optional<Eigen::Vector2f> ankle;

  boost::optional<double> min_distance;
};

}

#endif // OBSERVATION_HPP
