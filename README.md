The ES EKF state estimator was implemented and tuned for 3 use cases:
1) All of the data is coming properly: LIDAR, IMU and GNSS copming with a different rate
2)  Lidar is not calibrated properly and does not provide accurate measurements
3)  Sensor dropout

The estimated position is compared with the real position (groundtruthed) which was recorded using CARLA.

The variances of the sensors inputs were tuned as following:

var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.1
var_lidar = 0.1

