#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <yaml-cpp/yaml.h>
#include<random>
#define SHOW 0
class SSC
{
private:
    std::vector<int> order_vec = {0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 15, 16, 14, 17, 9, 18, 19}; // 语义优先级
    typedef std::tuple<u_char, u_char, u_char> Color;
    std::map<uint32_t, Color> _color_map, _argmax_to_rgb; // key:合并后的语义标签, value:RGB颜色
    YAML::Node learning_map;
    std::vector<int> label_map; // 语义标签映射
    double max_dis=50; // 激光有效范围
    double min_dis=5;
    int rings=24; // 极坐标网格径向划分数量
    int sectors=360; // 极坐标网格方位角划分数量
    int sectors_range=360;
    bool rotate=false; // 数据增强旋转
    bool occlusion=false; // 遮挡处理
    bool remap=true; // 是否重新映射标签
    std::shared_ptr<std::default_random_engine> random_generator;
    std::shared_ptr<std::uniform_int_distribution<int> > random_distribution;
    struct timeval time_t;
    bool show=false;
    std::shared_ptr<pcl::visualization::CloudViewer> viewer;
    int fastAtan2(float y,float x);
public:
    SSC(std::string conf_file);
    ~SSC();
    double getScore(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZL>::Ptr cloud2, Eigen::Matrix4f& trans);
    double getScore(std::string cloud_file1, std::string cloud_file2, std::string label_file1, std::string label_file2, Eigen::Matrix4f& transform);
    double getScore(std::string cloud_file1, std::string cloud_file2, Eigen::Matrix4f& transform);
    double getScore(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZL>::Ptr cloud2, double &angle,float& diff_x,float& diff_y);
    double getScore(std::string cloud_file1,std::string cloud_file2,std::string label_file1,std::string label_file2,double &angle,float& diff_x,float& diff_y);
    double getScore(std::string cloud_file1,std::string cloud_file2,double &angle,float& diff_x,float& diff_y);

    // 从 KITTI 格式 的一帧数据里，读出带语义标签的点云（PointXYZL），并可按配置做 标签重映射 和 数据增强（旋转、遮挡）
    pcl::PointCloud<pcl::PointXYZL>::Ptr getLCloud(std::string file_cloud, std::string file_label);

    // 加载带标签的点云
    pcl::PointCloud<pcl::PointXYZL>::Ptr getLCloud(std::string file_cloud);

    // 将带语义标签的点云转换为彩色点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getColorCloud(pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud_in);

    // 从带语义的点云生成 2D 极坐标语义图（SSC 描述子）
    cv::Mat calculateSSC( pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud);

    // 把一帧带语义的 3D 点云压成一维极坐标描述子，每个扇区只保留该方向上距离最近的、指定语义类的点（距离 + xy + 语义 ID）
    cv::Mat project(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud);
    
    // 把单通道SSC转成RGB
    cv::Mat getColorImage(cv::Mat &desc);

    // 对两帧1维ssc(来自project)计算ssc2到ssc1的粗旋转角和累计平移
    void globalICP(cv::Mat& isc_dis1,cv::Mat& isc_dis2,double &angle,float& diff_x,float& diff_y);

    // 对两帧1维ssc(来自project)计算ssc2到ssc1的粗旋转，然后使用PCL ICP细化旋转和平移
    Eigen::Matrix4f globalICP(cv::Mat &ssc_dis1, cv::Mat &ssc_dis2);

    // 
    double calculateSim(cv::Mat &desc1, cv::Mat &desc2);
};
