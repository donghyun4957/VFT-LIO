#include <ros/ros.h>
#include "utility.h"
#include "lio_sam/cloud_info.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

struct MulranPointXYZIRT { // from the file player's topic https://github.com/irapkaist/file_player_mulran, see https://github.com/irapkaist/file_player_mulran/blob/17da0cb6ef66b4971ec943ab8d234aa25da33e7e/src/ROSThread.cpp#L7
     PCL_ADD_POINT4D;
     float intensity;
     uint32_t t;
     int ring;

     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 }EIGEN_ALIGN16;
 POINT_CLOUD_REGISTER_POINT_STRUCT (MulranPointXYZIRT,
     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
     (uint32_t, t, t) (int, ring, ring)
 )


// Use the Velodyne point format as a common representation
using PointXYZIRT = MulranPointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<MulranPointXYZIRT>::Ptr tmpMulranCloudIn;
    pcl::PointCloud<PointType>::Ptr   deskewedCloud;
    pcl::PointCloud<PointType>::Ptr   prevDeskewedCloud;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;
    

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;

    ros::NodeHandle nh;
    ros::Subscriber subOuster;
    ros::Subscriber subDeskew;
    ros::Publisher pubRangeImg;
    ros::Publisher pubImgmsg;
    cv::Mat Img_Ouster;

    // ros::Time lastProcessedTime = ros::Time::now();
    // double desiredPeriod = 0.15;
    // **
    //std::vector<int> px_offset {0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18};
    //std::vector<int> v {0, 6, 12, 18};
    //std::vector<int> px_offset;
    //int H = 64;
    //int W = 1024;
    //cv::Mat Img = cv::Mat::zeros(W, H, CV_8UC1);

public:
    ImageProjection():
    deskewFlag(0)
    {

        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        //subImu        = nh.subscribe<sensor_msgs::Imu>("/imu/data_raw", 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());
        //subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);
        pubImgmsg = nh.advertise<sensor_msgs::Image>("rangeimage_topic",1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        tmpMulranCloudIn.reset(new pcl::PointCloud<MulranPointXYZIRT>());
        deskewedCloud.reset(new pcl::PointCloud<PointType>());
        prevDeskewedCloud.reset(new pcl::PointCloud<PointType>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        deskewedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {

        if (!cachePointCloud(laserCloudMsg))
            return;

        if (!deskewInfo())
            return;

        projectPointCloud();

        publishClouds();
        
        saveHistory();

        resetParameters();

    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
        {
            return false;
        }

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
    
        cloudQueue.pop_front();
        //if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        
        
        if (sensor == SensorType::MULRAN)
        {
            pcl::fromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            cout << "here4";
            ros::shutdown();
        }
        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        // **
        timeScanEnd = timeScanCur + laserCloudIn->points.back().t;
 
        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }
        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }
        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            // for (auto &field : currentCloudMsg.fields)
            // {
            //     if (field.name == "time" || field.name == "t")
            //     {
            //         deskewFlag = 1;
            //         break;
            //     }
            // }
            // if (deskewFlag == -1)
            //     ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }
        return true;
    }

    //void getImg(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    void getImg(pcl::PointCloud<PointType>::Ptr   Cloud)
    {
        //pcl::PointCloud<pcl::PointXYZI>::Ptr Ouster(new pcl::PointCloud<pcl::PointXYZI>);
        //pcl::fromROSMsg(*laserCloudMsg, *Ouster);
        std::vector<int> v {0, 6, 12, 18};
        std::vector<int> px_offset;

        // Format : PCAP
        for (int j = 0; j < 16; j++) {
            px_offset.insert(px_offset.end(), v.begin(), v.end());
        }

        pcl::PointCloud<pcl::PointXYZI>::VectorType intensity = Cloud->points;
        for (pcl::PointXYZI &point : intensity) {
            point.intensity = point.intensity * 0.001*255;
        }

        cv::Mat Img = cv::Mat::zeros(Horizon_SCAN, N_SCAN, CV_8UC1);
        for (int u = 0; u < N_SCAN; u++) {
            //cout << "here3";
            for (int v = 0; v < Horizon_SCAN; v++) {
                int vv = (v + px_offset[u] - 1) % Horizon_SCAN;
                int index = vv * N_SCAN + u;
                if (index <= 0) {
                    index = 1;
                }
                else if (index >= intensity.size()){
                    index = intensity.size()-1;
                }
                Img.at<uchar>(v, u) = intensity[index].intensity;
            }
        }
        // **
        Img_Ouster = Img.t();
        cv::imwrite("/home/jason/WorkSpace/test_ws/src/rangeimg.jpg", Img_Ouster);
        pubImg(Img_Ouster);
        Img_Ouster = cv::Mat::zeros(N_SCAN,Horizon_SCAN,CV_8SC1);
    }

    void pubImg(cv::Mat& Img_Ouster)
    {
        cloudInfo.img_msg = *(cv_bridge::CvImage(cloudHeader, "mono8", Img_Ouster).toImageMsg());
        sensor_msgs::ImagePtr imgmsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", Img_Ouster).toImageMsg();
        imgmsg->header.stamp = cloudHeader.stamp;
        imgmsg->header.frame_id = "camera";
        pubImgmsg.publish(imgmsg);
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);
        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            // float range = pointDistance(thisPoint);
            // if (range < lidarMinRange || range > lidarMaxRange)
            // {
            //     deskewedCloud->push_back(thisPoint); // 40000 pcl is decreased when uncommen it 
            //     //cout << "range : " << range << endl;
            //     continue;
            // }

            // int rowIdn = laserCloudIn->points[i].ring;
            // if (rowIdn < 0 || rowIdn >= N_SCAN)
            // {
            //     deskewedCloud->push_back(thisPoint); // 500 pcl is decreased when uncomment it 
            //     continue;
            // }

            // if (rowIdn % downsampleRate != 0)
            // {
            //     deskewedCloud->push_back(thisPoint); // no pcl is decreased whenuncomment it
            //     continue;
            // }

            // int columnIdn = -1;
            // if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER || sensor == SensorType::MULRAN)
            // {
            //     float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            //     static float ang_res_x = 360.0/float(Horizon_SCAN);
            //     columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
            //     if (columnIdn >= Horizon_SCAN)
            //         columnIdn -= Horizon_SCAN;
            // }
            // else if (sensor == SensorType::LIVOX)
            // {
            //     columnIdn = columnIdnCountVec[rowIdn];
            //     columnIdnCountVec[rowIdn] += 1;
            // }
            
            // if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
            // {
            //     deskewedCloud->push_back(thisPoint); // no pcl is decreased whenuncomment it
            //     continue;
            // }

            // if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
            // {
            //     deskewedCloud->push_back(thisPoint); // no pcl is decreased whenuncomment it
            //     continue;
            // }

            // **
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].t);
            //cout << "  relTime in another loop : " << laserCloudIn->points[i].t << "  dt : " << (timeScanEnd - timeScanCur) << endl;
            deskewedCloud->push_back(thisPoint);
            
            //rangeMat.at<float>(rowIdn, columnIdn) = range;

            //int index = columnIdn + rowIdn * Horizon_SCAN;
            //fullCloud->points[index] = thisPoint;
        }

        getImg(deskewedCloud);       

        sensor_msgs::PointCloud2 deskewPcl;
        sensor_msgs::PointCloud2 prevDeskewPcl;
        pcl::toROSMsg(*deskewedCloud, deskewPcl);
        deskewPcl.header.stamp = cloudHeader.stamp;
        deskewPcl.header.frame_id = lidarFrame;
        cloudInfo.raw_points = deskewPcl;

        pcl::toROSMsg(*prevDeskewedCloud, prevDeskewPcl);
        prevDeskewPcl.header.stamp = cloudHeader.stamp;
        prevDeskewPcl.header.frame_id = lidarFrame;
        cloudInfo.prev_raw_points = prevDeskewPcl;
    }

    void saveHistory()
    {
        prevDeskewedCloud->clear();
        prevDeskewedCloud->resize(deskewedCloud->size());

        for (size_t i = 0; i < deskewedCloud->size(); ++i)
        {
        prevDeskewedCloud->points[i] = deskewedCloud->points[i];
        }

    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, deskewedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
