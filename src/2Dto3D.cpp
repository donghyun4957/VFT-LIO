#include <cmath>
#include <iostream>
#include <ros/ros.h>

#include "utility.h"
#include "lio_sam/cloud_info.h"

#include <list>
#include <vector>
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
#include <std_msgs/Int32MultiArray.h>

using namespace std;

//void print_vector(std::vector<int> v){
//    for (std::vector<int>::const_iterator it = v.begin(); it != v.end(); ++it)
//    {
//        cout << *it << endl;   
//    }   
//}

class ImgtoPcl : public ParamServer
{
private:
    // node handler
    ros::NodeHandle nh;
    ros::Subscriber subOS1;
    ros::Subscriber subFeature;
    ros::Publisher pubFeature;
    ros::Publisher pub3Dfeature;
    //ros::Publisher pub3Dfeature_prev;
    lio_sam::cloud_info cloudInfo;
    //pcl::PointCloud<pcl::PointXYZI> ptCloud;
    // pcl saver
    
    // **
    //pcl::PointCloud<pcl::PointXYZI> ptCloud;
    //pcl::PointCloud<pcl::PointXYZI> featPcl;
    // **
    //std::vector<int> pixel_offset {0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18, 0, 6, 12, 18};
    std_msgs::Header cloudHeader;
    int done = 0;
public:
    ImgtoPcl()
    {
        cout<<"img to pcl constructor start" << endl;
        // **
        pubFeature = nh.advertise<lio_sam::cloud_info>("lio_sam/3D_feature", 100);
        pub3Dfeature = nh.advertise<sensor_msgs::PointCloud2>("/feature_pointclouds", 100);
        //pub3Dfeature_prev = nh.advertise<sensor_msgs::PointCloud2>("/feature_pointclouds_prev", 100);
        //subOS1 = nh.subscribe<lio_sam::cloud_info>("lio_sam/deskew/cloud_info",100, &ImgtoPcl::cloud3D_handler, this,ros::TransportHints().tcpNoDelay());
        subFeature = nh.subscribe<lio_sam::cloud_info>("lio_sam/2D_feature", 100, &ImgtoPcl::feature2D_handler, this,ros::TransportHints().tcpNoDelay());
    }
    // **
    //void cloud3D_handler(const lio_sam::cloud_infoConstPtr& ConvertingMsg)
    //{
        // **
        //pcl::sPointCloud<pcl::PointXYZI>::Ptr ptCloud(new pcl::PointCloud<pcl::PointXYZI>);
    //    pcl::fromROSMsg(ConvertingMsg->cloud_deskewed, ptCloud);
    //    done = 1;
    //    cout << 1;
    //}
    // **
    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }
    // **

    //void feature2D_handler (const std_msgs::Int32MultiArray::ConstPtr& msg)
    void feature2D_handler (const lio_sam::cloud_info::ConstPtr& ConvertingMsg)
    {
        ROS_INFO("received 2D features");
        //pcl::PointCloud<pcl::PointXYZI> ptCloud;
        pcl::PointCloud<pcl::PointXYZI>::Ptr ptCloud(new pcl::PointCloud<pcl::PointXYZI>);
        //cout << "here" << endl;
        pcl::PointCloud<pcl::PointXYZI>::Ptr prev_ptCloud(new pcl::PointCloud<pcl::PointXYZI>);
        //cout << "here2" << endl;
        pcl::PointCloud<pcl::PointXYZI> featPcl;

        //pcl::PointCloud<pcl::PointXYZI> featPcl_test;
        //cout << "here3" << endl;
        pcl::PointCloud<pcl::PointXYZI> register_featPcl;
        //cout << 'a' << endl;
        cloudInfo = *ConvertingMsg;
        cloudHeader = ConvertingMsg->header;
        //pcl::fromROSMsg(cloudInfo.cloud_deskewed, ptCloud);        
        pcl::fromROSMsg(cloudInfo.raw_points, *ptCloud);
        pcl::fromROSMsg(cloudInfo.prev_raw_points, *prev_ptCloud); 
        //cout << "size of ptCloud : " << ptCloud->points.size() << endl;      
        //cout << "size of prev_ptCloud : " << prev_ptCloud->points.size() << endl;
        //cout << "Size of cloudInfo.cloud_deskewed: " << cloudInfo.cloud_deskewed.data.size() << endl;
        int H = 64;
        int W = 1024;
        int iter = 0;
        //cout << 'b' << endl;
        
        // Format : PCAP
        
        // **
        std::vector<int> v {0, 6, 12, 18};
        std::vector<int> pixel_offset;
        for (int j = 0; j < 16; j++) {
            pixel_offset.insert(pixel_offset.end(), v.begin(), v.end());
        }
        //if(done == 1){
        //cout << 'c' << endl;

        int vv1;    int vv2;
        int index1; int index2; int i = 0;

                    // **
        std::vector<int> point_container{0,0,0,0}; //p1_x, p1_y, p2_x, p2_y
        //cout << "cloudInfo.feature_msg.data: " << cloudInfo.feature_msg << endl;

        //cout << "d" << endl;

        for (std::vector<int>::const_iterator it = cloudInfo.matched_feature_msg.data.begin(); it != cloudInfo.matched_feature_msg.data.end(); ++it){          
            point_container[iter%4] = *it;
            //cout << "e" << endl;
            if (iter%4 == 3){
                // print_vector(point_container);

                vv1 = (point_container[0] + pixel_offset[point_container[1]] - 1) % W;
                index1 = vv1 * H + point_container[1];
                //cout << "index1 just now: " << index1 << endl;
                vv2 = (point_container[2] + pixel_offset[point_container[3]] - 1) % W;
                index2 = vv2 * H + point_container[3];
                //cout << "index2 just now: " << index2 << endl;
                //PointType conditionPoint;
                // conditionPoint.x = ptCloud->points[index1].x;
                // conditionPoint.y = ptCloud->points[index1].y;
                // conditionPoint.z = ptCloud->points[index1].z;
                // conditionPoint.intensity = ptCloud->points[index1].intensity;
                // if (pointDistance(conditionPoint) != FLT_MAX)
                // {
                featPcl.push_back(prev_ptCloud->points[index1]);
                //cout << "index1: " << index1 << endl;
                //cout << "points in ptCloud : " << ptCloud.points[index1] << endl; 
                featPcl.push_back(ptCloud->points[index2]);
                //cout << "index2: " << index2 << endl;
                //featPcl_test.push_back(prev_ptCloud->points[index1]);

                register_featPcl.push_back(ptCloud->points[index2]);   
            }
        
            iter = iter+1;
            //cout << "f" << endl;
        }

        // for (const int &value : cloudInfo.feature_msg.data) {
        //     point_container[iter % 2] = value;

        //     if (iter % 2 == 1) {

        //     int vv1 = (point_container[0] + pixel_offset[point_container[1]] - 1) % W;
        //     int index1 = vv1 * H + point_container[1];
        //     cout << "vv1 : " << vv1 << "index1 : " << index1 << endl;
        //     cout << "Size of cloudInfo.cloud_deskewed: " << cloudInfo.cloud_deskewed.data.size() << endl;
        //     cout << "Size of ptCloud: " << ptCloud.points.size() << endl;

        //     featPcl.push_back(ptCloud.points[index1]);
        //     }

        //     iter++;
        // }
        //}   
    
        //cout << 'd';

        //cout << "size of featPcl : " << featPcl.size();
        sensor_msgs::PointCloud2 feature_cloud;
        //sensor_msgs::PointCloud2 feature_cloud_prev;
        sensor_msgs::PointCloud2 matched_feature_cloud;
        pcl::toROSMsg(register_featPcl, feature_cloud);
        //pcl::toROSMsg(featPcl_test, feature_cloud_prev);
        pcl::toROSMsg(featPcl, matched_feature_cloud);
        //cout << 'e';
        // **
        //freeCloudInfoMemory();

        //cout << 'f';

        feature_cloud.header.stamp = cloudHeader.stamp;
        feature_cloud.header.frame_id = lidarFrame;
        //feature_cloud_prev.header.stamp = cloudHeader.stamp;
        //feature_cloud_prev.header.frame_id = lidarFrame;        
        matched_feature_cloud.header.stamp = cloudHeader.stamp;
        matched_feature_cloud.header.frame_id = lidarFrame;        
        cloudInfo.cloud_corner = feature_cloud;
        cloudInfo.matching_corner = matched_feature_cloud;
        pubFeature.publish(cloudInfo);
        pub3Dfeature.publish(feature_cloud);
        //pub3Dfeature_prev.publish(feature_cloud_prev);

        //cout << 'g' << endl;
        ROS_INFO("finished 2D features");
    }
};
int main(int argc, char **argv)
{
    ros::init(argc, argv, "2Dto3D");
    ImgtoPcl IP;
    ROS_INFO("\033[1;32m----> feature 2D points to 3D points converter Started.\033[0m");
    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    return 0;
}