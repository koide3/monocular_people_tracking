#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <visualization_msgs/MarkerArray.h>
#include <tfpose_ros/Persons.h>
#include <monocular_people_tracking/TrackArray.h>

#include <kkl/cvk/cvutils.hpp>
#include <kkl/math/gaussian.hpp>

#include <monocular_people_tracking/observation.hpp>
#include <monocular_people_tracking/people_tracker.hpp>

namespace monocular_people_tracking {

class MonocularPeopleTrackingNode {
public:
  MonocularPeopleTrackingNode()
    : nh(),
      private_nh("~"),
      poses_sub(nh.subscribe("/pose_estimator/pose", 10, &MonocularPeopleTrackingNode::poses_callback, this)),
      camera_info_sub(nh.subscribe("/top_front_camera/camera_info", 60, &MonocularPeopleTrackingNode::camera_info_callback, this)),
      tracks_pub(private_nh.advertise<monocular_people_tracking::TrackArray>("tracks", 10)),
      markers_pub(private_nh.advertise<visualization_msgs::MarkerArray>("markers", 10)),
      image_trans(private_nh),
      image_pub(image_trans.advertise("tracking_image", 5))
  {
    ROS_INFO("start monocular_people_tracking_node");
    color_palette = cvk::create_color_palette(16);
    tf_listener.reset(new tf::TransformListener());
  }

private:
  void camera_info_callback(const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
      ROS_INFO("camera_info");
      this->camera_info_msg = camera_info_msg;
  }

  void poses_callback(const tfpose_ros::PersonsConstPtr& poses_msg) {
      ROS_INFO("pose_callback");
      if(camera_info_msg == nullptr) {
          ROS_INFO("waiting for the camera info msg...");
          return;
      }

      std::vector<Observation::Ptr> observations;
      observations.reserve(poses_msg->persons.size());
      for(const auto& person : poses_msg->persons) {
        Joint neck, lankle, rankle;
        for(const auto& joint: person.body_part) {
            switch (joint.part_id) {
            case 1:
                neck = Joint(joint.confidence, joint.x * poses_msg->image_w, joint.y * poses_msg->image_h);
                break;
            case 13:
                lankle = Joint(joint.confidence, joint.x * poses_msg->image_w, joint.y * poses_msg->image_h);
                break;
            case 10:
                rankle = Joint(joint.confidence, joint.x * poses_msg->image_w, joint.y * poses_msg->image_h);
                break;
            }
        }

        auto observation = std::make_shared<Observation>(neck, lankle, rankle, camera_info_msg);
        if(observation->is_valid()) {
          observations.push_back(observation);
        }
      }

      if(!people_tracker) {
        people_tracker.reset(new PeopleTracker(tf_listener, poses_msg->header.frame_id, camera_info_msg));
      }

      const auto& stamp = poses_msg->header.stamp;
      people_tracker->predict(stamp);
      people_tracker->correct(stamp, observations);

      if(image_pub.getNumSubscribers()) {
        cv::Mat frame = cv::Mat(poses_msg->image_h, poses_msg->image_w, CV_8UC3, cv::Scalar::all(255));
        cv_bridge::CvImage cv_image(poses_msg->header, "bgr8");
        cv_image.image = visualize(frame, observations);
        image_pub.publish(cv_image.toImageMsg());
      }

      if(markers_pub.getNumSubscribers()) {
        markers_pub.publish(create_markers(poses_msg->header.stamp));
      }
  }

  visualization_msgs::MarkerArrayConstPtr create_markers(const ros::Time& stamp) const {
    visualization_msgs::MarkerArrayPtr markers(new visualization_msgs::MarkerArray());
    if(!people_tracker) {
      return markers;
    }

    // boxes
    markers->markers.resize(2);
    markers->markers[0].header.stamp = stamp;
    markers->markers[0].header.frame_id = "odom";
    markers->markers[0].action = visualization_msgs::Marker::ADD;
    markers->markers[0].lifetime = ros::Duration(1.0);

    markers->markers[0].ns = "lines";
    markers->markers[0].type = visualization_msgs::Marker::LINE_LIST;
    markers->markers[0].colors.reserve(people_tracker->get_people().size() * 2);
    markers->markers[0].points.reserve(people_tracker->get_people().size() * 2);

    markers->markers[0].scale.x = 0.05;
    markers->markers[0].pose.position.z = -0.3;
    markers->markers[0].pose.orientation.w = 1.0;

    // spheres
    markers->markers[1] = markers->markers[0];
    markers->markers[1].ns = "spheres";
    markers->markers[1].type = visualization_msgs::Marker::SPHERE_LIST;
    markers->markers[1].colors.reserve(people_tracker->get_people().size() * 2);
    markers->markers[1].points.reserve(people_tracker->get_people().size() * 2);
    markers->markers[1].scale.x = markers->markers[1].scale.y = markers->markers[1].scale.z = 0.5f;

    for(const auto& person : people_tracker->get_people()) {
      if(!person->is_valid()) {
        continue;
      }

      const auto& color = color_palette[person->id() % color_palette.size()];
      std_msgs::ColorRGBA rgba;
      rgba.r = color[2] / 255.0;
      rgba.g = color[1] / 255.0;
      rgba.b = color[0] / 255.0;
      rgba.a = 0.5f;

      auto& lines = markers->markers[0];
      auto& spheres = markers->markers[1];

      lines.colors.push_back(rgba);
      lines.colors.push_back(rgba);
      spheres.colors.push_back(rgba);

      Eigen::Vector3f pos = person->pos();
      geometry_msgs::Point line_pt1, line_pt2;
      line_pt1.x = pos.x();   line_pt1.y = pos.y();   line_pt1.z = pos.z();
      line_pt2.x = pos.x();   line_pt2.y = pos.y();   line_pt2.z = 0.0;
      lines.points.push_back(line_pt1);
      lines.points.push_back(line_pt2);

      spheres.points.push_back(line_pt1);
    }

    return markers;
  }

  cv::Mat visualize(const cv::Mat& frame, const std::vector<Observation::Ptr>& observations) const {
    cv::Mat canvas = frame.clone();

    cv::Mat layer(canvas.size(), CV_8UC3, cv::Scalar(0, 32, 0));
    cv::rectangle(layer, cv::Rect(100, 25, layer.cols - 200, layer.rows - 50), cv::Scalar(0, 0, 0), -1);
    canvas += layer;

    for(const auto& observation: observations) {
      if(observation->neck) {
        cv::circle(canvas, cv::Point(observation->neck->x(), observation->neck->y()), 5, cv::Scalar(0, 0, 255), -1);
      }
      if(observation->ankle) {
        cv::circle(canvas, cv::Point(observation->ankle->x(), observation->ankle->y()), 5, cv::Scalar(0, 0, 255), -1);
      }
    }

    for(const auto& person : people_tracker->get_people()) {
      if(!person->is_valid()) {
        continue;
      }

      auto dist = person->expected_measurement_distribution();
      cv::putText(canvas, (boost::format("id:%d") % person->id()).str(), cv::Point(dist.first[0] - 15, dist.first[1] - 25), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar::all(255));
      cv::putText(canvas, (boost::format("tf:%.2f") % person->trace()).str(), cv::Point(dist.first[0] - 15, dist.first[1] - 15), CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar::all(255));

      Eigen::Matrix2f neck_cov = dist.second.block<2, 2>(0, 0);
      Eigen::Matrix2f ankle_cov = dist.second.block<2, 2>(2, 2);
      Eigen::Vector3f neck_params = kkl::math::errorEllipse<float>(neck_cov, 3.0);
      Eigen::Vector3f ankle_params = kkl::math::errorEllipse<float>(ankle_cov, 3.0);

      cv::RotatedRect neck(cv::Point2f(dist.first[0], dist.first[1]), cv::Size2f(neck_params[0], neck_params[1]), neck_params[2]);
      cv::RotatedRect ankle(cv::Point2f(dist.first[2], dist.first[3]), cv::Size2f(ankle_params[0], ankle_params[1]), ankle_params[2]);
      cv::ellipse(canvas, neck, cv::Scalar(0, 255, 0));
      cv::ellipse(canvas, ankle, cv::Scalar(0, 255, 0));
      cv::line(canvas, cv::Point(dist.first[0], dist.first[1]), cv::Point(dist.first[2], dist.first[3]), cv::Scalar(0, 255, 0));
    }

    return canvas;
  }

private:
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  std::shared_ptr<tf::TransformListener> tf_listener;

  ros::Subscriber poses_sub;
  ros::Subscriber camera_info_sub;

  ros::Publisher tracks_pub;
  ros::Publisher markers_pub;

  image_transport::ImageTransport image_trans;
  image_transport::Publisher image_pub;

  boost::circular_buffer<cv::Scalar> color_palette;

  sensor_msgs::CameraInfoConstPtr camera_info_msg;
  std::unique_ptr<PeopleTracker> people_tracker;
};

}

int main(int argc, char** argv) {
  ros::init(argc, argv, "monocular_people_tracking");
  std::unique_ptr<monocular_people_tracking::MonocularPeopleTrackingNode> node(new monocular_people_tracking::MonocularPeopleTrackingNode());
  ros::spin();

  return 0;
}
