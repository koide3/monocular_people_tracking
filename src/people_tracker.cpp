#include <monocular_people_tracking/people_tracker.hpp>

#include <kkl/alg/data_association.hpp>
#include <kkl/alg/nearest_neighbor_association.hpp>

#include <monocular_people_tracking/track_system.hpp>


namespace monocular_people_tracking {

PeopleTracker::PeopleTracker(ros::NodeHandle& private_nh, const std::shared_ptr<tf::TransformListener>& tf_listener, const std::string& camera_frame_id, const sensor_msgs::CameraInfoConstPtr& camera_info_msg) {
    id_gen = 0;
    remove_trace_thresh = private_nh.param<double>("tracking_remove_trace_thresh", 5.0);
    dist_to_exists_thresh = private_nh.param<double>("tracking_newtrack_dist2exists_thersh", 100.0);

    data_association.reset(new kkl::alg::NearestNeighborAssociation<PersonTracker::Ptr, Observation::Ptr, AssociationDistance>(AssociationDistance(private_nh)));
    track_system.reset(new TrackSystem(private_nh, tf_listener, camera_frame_id, camera_info_msg));
}

PeopleTracker::~PeopleTracker() {

}

void PeopleTracker::predict(ros::NodeHandle& nh, const ros::Time& stamp) {
    track_system->update_matrices(stamp);
    for(const auto& person : people) {
        person->predict(stamp);
    }
}

void PeopleTracker::correct(ros::NodeHandle& nh, const ros::Time& stamp, const std::vector<Observation::Ptr>& observations) {
    if(!observations.empty()) {

        std::vector<bool> associated(observations.size(), false);
        auto associations = data_association->associate(people, observations);
        for(const auto& assoc : associations) {
            associated[assoc.observation] = true;
            people[assoc.tracker]->correct(stamp, observations[assoc.observation]);
        }

        for(int i=0; i<observations.size(); i++) {
            if(!associated[i] && observations[i]->ankle) {
                if(observations[i]->min_distance && *observations[i]->min_distance < dist_to_exists_thresh) {
                    continue;
                }

                if(observations[i]->close2border) {
                    continue;
                }

                PersonTracker::Ptr tracker(new PersonTracker(nh, track_system, stamp, id_gen++, observations[i]->neck_ankle_vector()));
                tracker->correct(stamp, observations[i]);
                people.push_back(tracker);
            }
        }
    }

    auto remove_loc = std::partition(people.begin(), people.end(), [&](const PersonTracker::Ptr& tracker) {
        return tracker->trace() < remove_trace_thresh;
    });
    removed_people.clear();
    std::copy(remove_loc, people.end(), std::back_inserter(removed_people));
    people.erase(remove_loc, people.end());
}

}
