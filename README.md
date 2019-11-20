# monocular_people_tracking

This package provides a monocular vision-based people tracking method. It utilizes *tf-pose-estimation* to detect people joints, and by projecting the detected ankle onto the ground plane, it estimates the person position in the robot space. To make the estimation robust, it estimates the height of the person in addition to his/her position with Unscented Kalman Filter.

## Dependencies

- tf-pose-estimation


## Related packages

- [ccf_person_identification](https://github.com/koide3/ccf_person_identification)
- [monocular_people_tracking](https://github.com/koide3/monocular_people_tracking)
- [monocular_person_following](https://github.com/koide3/monocular_person_following)


## Papers
- Kenji Koide, Jun Miura, and Emanuele Menegatti, Monocular Person Tracking and Identification for Person Following Robots, Robotics and Autonomous Systems.

- Kenji Koide and Jun Miura, Convolutional Channel Features-based Person Identification for Person Following Robots, 15th International Conference IAS-15, Baden-Baden, Germany, 2018 [[link]](https://www.researchgate.net/publication/325854919_Convolutional_Channel_Features-Based_Person_Identification_for_Person_Following_Robots_Proceedings_of_the_15th_International_Conference_IAS-15).
