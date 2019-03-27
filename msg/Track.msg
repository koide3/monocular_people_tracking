int32 id
float64 age

geometry_msgs/Point pos
float32 height
geometry_msgs/Vector3 vel

float32[25] cov

float32[4] expected_measurement_mean
float32[16] expected_measurement_cov

tfpose_ros/Person[] associated
geometry_msgs/Point32[] associated_neck_ankle
