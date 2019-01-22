#include <monocular_people_tracking/people_tracker.hpp>

#include <kkl/alg/data_association.hpp>
#include <kkl/alg/nearest_neighbor_association.hpp>

#include <monocular_people_tracking/track_system.hpp>

namespace kkl {
  namespace alg {

template<>
boost::optional<double> distance(const monocular_people_tracking::PersonTracker::Ptr& tracker, const monocular_people_tracking::Observation::Ptr& observation) {
  if(observation->ankle) {
    double distance = (tracker->expected_measurement_distribution().first - observation->neck_ankle_vector()).norm();
    if(!observation->min_distance){
      observation->min_distance = distance;
    } else {
      observation->min_distance = std::min(distance, *observation->min_distance);
    }

    if(distance > 200) {
      return boost::none;
    }

    double sq_maha = tracker->squared_mahalanobis_distance(observation->neck_ankle_vector());
    if(sq_maha > pow(3.0, 2)) {
      return boost::none;
    }
    return -tracker->prob(observation->neck_ankle_vector());
  }

  double sq_maha = tracker->squared_mahalanobis_distance(observation->neck_vector());
  if(sq_maha > pow(3.0, 2)) {
    return boost::none;
  }

  return -tracker->prob(observation->neck_vector()) + 1.0;
}
  }
}


namespace monocular_people_tracking {

PeopleTracker::PeopleTracker(const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
  id_gen = 0;
  data_association.reset(new kkl::alg::NearestNeighborAssociation<PersonTracker::Ptr, Observation::Ptr>());

  track_system.reset(new TrackSystem(tf_listener, camera_frame_id, camera_info_msg));
}

PeopleTracker::~PeopleTracker() {

}

void PeopleTracker::predict(const ros::Time& stamp) {
  track_system->update_matrices(stamp);
  for(const auto& person : people) {
    person->predict(stamp);
  }
}

void PeopleTracker::correct(const ros::Time& stamp, const std::vector<Observation::Ptr>& observations) {
  std::vector<bool> associated(observations.size(), false);
  auto associations = data_association->associate(people, observations);
  for(const auto& assoc : associations) {
    associated[assoc.observation] = true;
    people[assoc.tracker]->correct(stamp, observations[assoc.observation]);
  }

  for(int i=0; i<observations.size(); i++) {
    if(!associated[i] && observations[i]->ankle) {
      if(observations[i]->min_distance && *observations[i]->min_distance < 80) {
        continue;
      }

      if(observations[i]->close2border) {
        continue;
      }

      PersonTracker::Ptr tracker(new PersonTracker(track_system, stamp, id_gen++, observations[i]->neck_ankle_vector()));
      tracker->correct(stamp, observations[i]);
      people.push_back(tracker);
    }
  }

  auto remove_loc = std::partition(people.begin(), people.end(), [&](const PersonTracker::Ptr& tracker) {
    return tracker->trace() < 5.0;
  });
  removed_people.clear();
  std::copy(remove_loc, people.end(), std::back_inserter(removed_people));
  people.erase(remove_loc, people.end());
}

}
