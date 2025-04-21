# Cloud Info
Header header 

int32[] startRingIndex
int32[] endRingIndex

int32[]  pointColInd # point column index in range image
float32[] pointRange # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 raw_points      # original cloud
sensor_msgs/PointCloud2 prev_raw_points      # original previous cloud
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed

sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 matching_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature

# Range Image messages
sensor_msgs/Image img_msg # original range image from raw data

# Extracted feature messages in range image
std_msgs/Int32MultiArray feature_msg

std_msgs/Int32MultiArray matched_feature_msg


# 3rd party messages
sensor_msgs/PointCloud2 key_frame_cloud
sensor_msgs/PointCloud2 key_frame_color
sensor_msgs/PointCloud2 key_frame_poses
sensor_msgs/PointCloud2 key_frame_map
