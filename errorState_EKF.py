# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data ###################################################################################

################################################################################################
# This is where you will load the data from the pickle files. For parts 1 and 2, you will use
# p1_data.pkl. For Part 3, you will use pt3_data.pkl.
################################################################################################
with open('data/pt3_data.pkl', 'rb') as file:
    data = pickle.load(file)

################################################################################################
# Each element of the data dictionary is stored as an item from the data dictionary, which we
# will store in local variables, described by the following:
#   gt: Data object containing ground truth. with the following fields:
#     a: Acceleration of the vehicle, in the inertial frame
#     v: Velocity of the vehicle, in the inertial frame
#     p: Position of the vehicle, in the inertial frame
#     alpha: Rotational acceleration of the vehicle, in the inertial frame
#     w: Rotational velocity of the vehicle, in the inertial frame
#     r: Rotational position of the vehicle, in Euler (XYZ) angles in the inertial frame
#     _t: Timestamp in ms.
#   imu_f: StampedData object with the imu specific force data (given in vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   imu_w: StampedData object with the imu rotational velocity (given in the vehicle frame).
#     data: The actual data
#     t: Timestamps in ms.
#   gnss: StampedData object with the GNSS data.
#     data: The actual data
#     t: Timestamps in ms.
#   lidar: StampedData object with the LIDAR data (positions only).
#     data: The actual data
#     t: Timestamps in ms.
################################################################################################
gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

################################################################################################
# Let's plot the ground truth trajectory to see what it looks like. When you're testing your
# code later, feel free to comment this out.
################################################################################################
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()

################################################################################################
# Remember that our LIDAR data is actually just a set of positions estimated from a separate
# scan-matching system, so we can insert it into our solver as another position measurement,
# just as we do for GNSS. However, the LIDAR frame is not the same as the frame shared by the
# IMU and the GNSS. To remedy this, we transform the LIDAR data to the IMU frame using our 
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.
#
# THIS IS THE CODE YOU WILL MODIFY FOR PART 2 OF THE ASSIGNMENT.
################################################################################################
# Correct calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.1).
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Incorrect calibration rotation matrix, corresponding to Euler RPY angles (0.05, 0.05, 0.05).
#C_li = np.array([
#     [ 0.9975 , -0.04742,  0.05235],
#     [ 0.04992,  0.99763, -0.04742],
#     [-0.04998,  0.04992,  0.9975 ]
#])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

#### 2. Constants ##############################################################################

################################################################################################
# Now that our data is set up, we can start getting things ready for our solver. One of the
# most important aspects of a filter is setting the estimated sensor variances correctly.
# We set the values here.
################################################################################################
var_imu_f = 0.10
var_imu_w = 0.25
var_gnss  = 0.1
var_lidar = 0.1
#var_lidar = 1e6  # Effectively ignores LIDAR measurements

################################################################################################
# We can also set up some constants that won't change for any iteration of our solver.
################################################################################################
g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

#### 3. Initial Values #########################################################################

################################################################################################
# Let's set up some initial values for our ES-EKF solver.
################################################################################################
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.eye(9) * 0.1  # covariance of estimate
gnss_i  = 0
lidar_i = 0

#### 4. Measurement Update #####################################################################

################################################################################################
# Since we'll need a measurement update for both the GNSS and the LIDAR data, let's make
# a function for it.
################################################################################################
def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    # 3.1 Compute Kalman Gain
    # H matrix is [I_3x3 | 0_3x3 | 0_3x3] which extracts position from error state
    # Measurement covariance: R = sensor_var
    R = np.array([[sensor_var, 0, 0],
                  [0, sensor_var, 0],
                  [0, 0, sensor_var]])
    
    # Kalman gain: K = P_check * H^T * (H * P_check * H^T + R)^{-1}
    S_inv = np.linalg.inv(h_jac @ p_cov_check @ h_jac.T + R)  # Innovation covariance inverse
    K = p_cov_check @ h_jac.T @ S_inv
    
    # 3.2 Compute error state
    # Innovation: y_k - p_check (measurement minus predicted position)
    innovation = y_k - p_check
    error_state = K @ innovation  # 9x1 error state correction
    
    # 3.3 Correct predicted state
    # Extract position, velocity, and orientation errors
    dp = error_state[:3]
    dv = error_state[3:6]
    dtheta = error_state[6:9]
    
    p_hat = p_check + dp
    v_hat = v_check + dv
    
    # Quaternion correction: q_hat = q_check * q_delta
    # where q_delta is the quaternion representation of small rotation dtheta
    q_delta = Quaternion(axis_angle=dtheta)
    q_hat_np = q_delta.quat_mult_left(q_check, out='np')
    q_hat = q_hat_np
    
    # Normalize quaternion to ensure unit norm
    q_hat_norm = np.linalg.norm(q_hat)
    q_hat = q_hat / q_hat_norm
    
    # 3.4 Compute corrected covariance
    # P_hat = (I - K * H) * P_check (Joseph form for numerical stability)
    p_cov_hat = (np.eye(9) - K @ h_jac) @ p_cov_check

    return p_hat, v_hat, q_hat, p_cov_hat

#### 5. Main Filter Loop #######################################################################

################################################################################################
# Now that everything is set up, we can start taking in the sensor data and creating estimates
# for our state in a loop.
################################################################################################
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]
    
    # Get IMU measurements
    f_imu = imu_f.data[k]  # specific force in vehicle frame
    w_imu = imu_w.data[k]  # angular velocity in vehicle frame
    
    # 1. Update state with IMU inputs (motion model / time update)
    
    # Convert quaternion to rotation matrix (from vehicle to inertial)
    q_prev = Quaternion(*q_est[k-1])
    C_I_v = q_prev.to_mat()  # rotation matrix from vehicle frame to inertial frame
    
    # Acceleration in inertial frame: a_inertial = R * a_vehicle + g
    f_inertial = C_I_v @ f_imu + g
    
    # Propagate state using kinematic model
    # Position: p_new = p_old + v_old * dt + 0.5 * a * dt^2
    p_new = p_est[k-1] + v_est[k-1] * delta_t + 0.5 * f_inertial * (delta_t ** 2)
    
    # Velocity: v_new = v_old + a * dt
    v_new = v_est[k-1] + f_inertial * delta_t
    
    # Orientation: q_new = q_old * exp(w * dt / 2)
    # Using axis-angle representation: axis-angle = w_imu * dt
    q_delta = Quaternion(axis_angle=w_imu * delta_t)
    q_new_np = q_prev.quat_mult_left(q_delta.to_numpy(), out='np')
    # Normalize quaternion
    q_new_np = q_new_np / np.linalg.norm(q_new_np)
    
    # 1.1 Linearize the motion model and compute Jacobians
    
    # F matrix: Jacobian of error state dynamics wrt error state
    # State order: [p, v, theta]
    # Error state propagation:
    # dp_new = dp_old + dv_old * dt
    # dv_new = dv_old - R(theta_old)^T * skew(a_imu) * dtheta_old * dt
    # dtheta_new = dtheta_old
    
    F = np.eye(9)
    # Top-left: position integration
    F[:3, 3:6] = np.eye(3) * delta_t
    # Middle: velocity wrt orientation error
    F[3:6, 6:9] = -C_I_v @ skew_symmetric(f_imu) * delta_t
    
    # L matrix is fixed (already defined in constants section)
    # L = [0_3x3 | 0_3x3]
    #     [I_3x3 | 0_3x3]
    #     [0_3x3 | I_3x3]
    
    # 2. Propagate uncertainty
    # Q is the process noise covariance for IMU measurements (6x6)
    # var_imu_f is the variance of force measurements
    # var_imu_w is the variance of angular velocity measurements
    Q = np.diag([var_imu_f, var_imu_f, var_imu_f, var_imu_w, var_imu_w, var_imu_w])
    
    # Covariance propagation: P_new = F * P_old * F^T + L * Q * L^T
    p_cov_new = F @ p_cov[k-1] @ F.T + l_jac @ Q @ l_jac.T
    
    # Save predicted state and covariance
    p_est[k] = p_new
    v_est[k] = v_new
    q_est[k] = q_new_np
    p_cov[k] = p_cov_new
    
    # 3. Check availability of GNSS and LIDAR measurements and apply updates
    
    # Check for GNSS measurement at this time
    while gnss_i < len(gnss.t) and gnss.t[gnss_i] <= imu_f.t[k]:
        if abs(gnss.t[gnss_i] - imu_f.t[k]) < 1e-2:
            # Apply GNSS measurement update
            p_est[k], v_est[k], q_est[k], p_cov[k] = \
                measurement_update(var_gnss, p_cov[k], gnss.data[gnss_i], 
                                   p_est[k], v_est[k], q_est[k])
        gnss_i += 1
    
    # Check for LIDAR measurement at this time
    while lidar_i < len(lidar.t) and lidar.t[lidar_i] <= imu_f.t[k]:
        if lidar.t[lidar_i] == imu_f.t[k]:
            # Apply LIDAR measurement update
            p_est[k], v_est[k], q_est[k], p_cov[k] = \
                measurement_update(var_lidar, p_cov[k], lidar.data[lidar_i],
                                   p_est[k], v_est[k], q_est[k])
        lidar_i += 1

#### 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
# Clip error values for plotting so extreme outliers do not distort the visualization
pos_clip = 10.0  # meters
angle_clip = 0.5  # radians
for i in range(3):
    position_error = np.clip(gt.p[:, i] - p_est[:num_gt, i], -pos_clip, pos_clip)
    pos_bound = np.clip(3 * p_cov_std[:num_gt, i], -pos_clip, pos_clip)
    ax[0, i].plot(range(num_gt), position_error)
    ax[0, i].plot(range(num_gt),  pos_bound, 'r--')
    ax[0, i].plot(range(num_gt), -pos_bound, 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    angle_error = np.clip(angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]), -angle_clip, angle_clip)
    ang_bound = np.clip(3 * p_cov_euler_std[:num_gt, i], -angle_clip, angle_clip)
    ax[1, i].plot(range(num_gt), angle_error)
    ax[1, i].plot(range(num_gt),  ang_bound, 'r--')
    ax[1, i].plot(range(num_gt), -ang_bound, 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()


plt.figure(figsize=(12,4))
plt.title("Sensor Availability Timeline")

# IMU timestamps (always present)
plt.plot(imu_f.t, np.zeros_like(imu_f.t), 'k.', markersize=2, label='IMU')

# GNSS timestamps
plt.plot(gnss.t, np.ones_like(gnss.t), 'g.', markersize=4, label='GNSS')

# LIDAR timestamps
plt.plot(lidar.t, 2*np.ones_like(lidar.t), 'r.', markersize=4, label='LIDAR')

plt.yticks([0,1,2], ['IMU','GNSS','LIDAR'])
plt.xlabel("Time [ms]")
plt.grid(True)
plt.legend()
plt.show()


#### 7. Submission #############################################################################

################################################################################################
# Now we can prepare your results for submission to the Coursera platform. Uncomment the
# corresponding lines to prepare a file that will save your position estimates in a format
# that corresponds to what we're expecting on Coursera.
################################################################################################

# Pt. 1 submission
#p1_indices = [9000, 9400, 9800, 10200, 10600]
#p1_str = ''
#for val in p1_indices:
#    for i in range(3):
#        p1_str += '%.3f ' % (p_est[val, i])
#with open('pt1_submission.txt', 'w') as file:
#    file.write(p1_str)

# Pt. 2 submission
##p2_indices = [9000, 9400, 9800, 10200, 10600]
##p2_str = ''
##for val in p2_indices:
##    for i in range(3):
##        p2_str += '%.3f ' % (p_est[val, i])
##with open('pt2_submission.txt', 'w') as file:
##    file.write(p2_str)

# Pt. 3 submission
p3_indices = [6800, 7600, 8400, 9200, 10000]
p3_str = ''
for val in p3_indices:
    for i in range(3):
        p3_str += '%.3f ' % (p_est[val, i])
with open('pt3_submission.txt', 'w') as file:
    file.write(p3_str)
