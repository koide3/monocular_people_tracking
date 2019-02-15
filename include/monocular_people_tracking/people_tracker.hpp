#ifndef MONOCULAR_PEOPLE_TRACKING_PEOPLE_TRACKER_HPP
#define MONOCULAR_PEOPLE_TRACKING_PEOPLE_TRACKER_HPP

#include <memory>
#include <vector>
#include <boost/optional.hpp>

#include <tf/transform_listener.h>
#include <sensor_msgs/CameraInfo.h>
#include <monocular_people_tracking/person_tracker.hpp>
#include <monocular_people_tracking/association_distance.hpp>

namespace kkl {
  namespace alg {
    template<typename Tracker, typename Observation>
    class DataAssociation;
  }
}

namespace monocular_people_tracking {

class TrackSystem;

class PeopleTracker {
public:
  PeopleTracker(ros::NodeHandle& private_nh, const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg);
  ~PeopleTracker();

  void predict(const ros::Time& stamp);

  void correct(const ros::Time& stamp, const std::vector<Observation::Ptr>& observations);

  const std::vector<PersonTracker::Ptr>& get_people() const { return people; }

private:
  std::shared_ptr<TrackSystem> track_system;
  std::unique_ptr<kkl::alg::DataAssociation<PersonTracker::Ptr, Observation::Ptr>> data_association;

  double remove_trace_thresh;
  double dist_to_exists_thresh;

  long id_gen;
  std::vector<PersonTracker::Ptr> people;
  std::vector<PersonTracker::Ptr> removed_people;
};

}

#endif // PEOPLE_TRACKER_HPP
