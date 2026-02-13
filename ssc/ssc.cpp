#include "ssc.h"
#include "pcl/impl/point_types.hpp"
#include "pcl/io/ply_io.h"
#include <sys/stat.h>
#include <cstdlib>
SSC::SSC(std::string conf_file)
{ 
    auto data_cfg = YAML::LoadFile(conf_file);
    show = data_cfg["show"].as<bool>();
    remap = data_cfg["remap"].as<bool>(); // 是否重新映射标签
    show_save_dir_ = data_cfg["show_save_dir"] ? data_cfg["show_save_dir"].as<std::string>() : "../out/show";
    if (show)
    {
        // viewer.reset(new pcl::visualization::CloudViewer("viewer"));
    }
    rotate = data_cfg["rotate"].as<bool>();
    occlusion = data_cfg["occlusion"].as<bool>();

    // 随机初始化
    gettimeofday(&time_t, nullptr);
    random_generator.reset(new std::default_random_engine(time_t.tv_usec));
    random_distribution.reset(new std::uniform_int_distribution<int>(-18000, 18000));

    // 将 learning_map 中的原始语义标签映射到新的语义标签
    learning_map = data_cfg["learning_map"];
    label_map.resize(260);
    for (auto it = learning_map.begin(); it != learning_map.end(); ++it)
    {
        label_map[it->first.as<int>()] = it->second.as<int>();
    }


    YAML::const_iterator it;
    // 读取每个语义标签对应的 RGB 颜色，可以按照原始语义标签查找颜色
    auto color_map = data_cfg["color_map"]; // 每个语义标签对应的rgb颜色映射
    for (it = color_map.begin(); it != color_map.end(); ++it)
    {
        // Get label and key
        int key = it->first.as<int>(); // <- key
        Color color = std::make_tuple(
            static_cast<u_char>(color_map[key][0].as<unsigned int>()),
            static_cast<u_char>(color_map[key][1].as<unsigned int>()),
            static_cast<u_char>(color_map[key][2].as<unsigned int>()));
        _color_map[key] = color;
    }

    // 反向颜色映射 在生成 2D 描述符图像时，可以根据这个 map 直接渲染出彩色的语义图
    auto learning_class = data_cfg["learning_map_inv"];
    for (it = learning_class.begin(); it != learning_class.end(); ++it)
    {
        int key = it->first.as<int>(); // <- key
        _argmax_to_rgb[key] = _color_map[learning_class[key].as<unsigned int>()];
    }
}

SSC::~SSC()
{
}

pcl::PointCloud<pcl::PointXYZL>::Ptr SSC::getLCloud(std::string file_cloud, std::string file_label)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr re_cloud(new pcl::PointCloud<pcl::PointXYZL>()); // 新建空的带语义标签点云

    // 打开标签文件
    std::ifstream in_label(file_label, std::ios::binary); 
    if (!in_label.is_open())
    {
        std::cerr << "No file:" << file_label << std::endl;
        exit(-1);
    }

    in_label.seekg(0, std::ios::end);
    uint32_t num_points = in_label.tellg() / sizeof(uint32_t); // 标签文件大小/uint32_t大小 = 点云数量
    in_label.seekg(0, std::ios::beg);
    std::vector<uint32_t> values_label(num_points);
    in_label.read((char *)&values_label[0], num_points * sizeof(uint32_t)); // 把整份标签读取到values_label中

    // 打开点云文件
    std::ifstream in_cloud(file_cloud, std::ios::binary);
    std::vector<float> values_cloud(4 * num_points);
    // 按点云每点 4 个 float 把整帧读进 values_cloud
    in_cloud.read((char *)&values_cloud[0], 4 * num_points * sizeof(float));
    
    // re_cloud 的点数设为 num_points，后面按索引往里填坐标和标签
    re_cloud->points.resize(num_points);
    re_cloud->width = num_points;
    re_cloud->height = 1;

    float random_angle = 0, max_angle = 0;
    float cs = 1, ss = 0;
    if (rotate || occlusion)
    {
        // 随机生成一个扇形区域的旋转角度 [random_angle,random_angle+M_PI/6]
        random_angle = (*random_distribution)(*random_generator) * M_PI / 18000.0;
        max_angle = random_angle + M_PI / 6.;

        cs = cos(random_angle);
        ss = sin(random_angle);
    }

    for (uint32_t i = 0; i < num_points; ++i)
    {
        if (occlusion)
        {
            // 若点云点落在扇形中视为遮挡，跳过
            float theta = atan2(values_cloud[4 * i + 1], values_cloud[4 * i]);
            if (theta > random_angle && theta < max_angle)
            {
                continue;
            }
        }

        // 重映射语义标签 remap 时用 label_map[低16位]，否则用原始 values_label[i]
        uint32_t sem_label;
        if (remap)
        {
            sem_label = label_map[(int)(values_label[i] & 0x0000ffff)];
        }
        else
        {
            sem_label = values_label[i];
        }

        // 若 sem_label == 0（未标注），把该点写成 (0,0,0)、label=0，然后 continue
        if (sem_label == 0)
        {
            re_cloud->points[i].x = 0;
            re_cloud->points[i].y = 0;
            re_cloud->points[i].z = 0;
            re_cloud->points[i].label = 0;
            continue;
        }

        // 若开启 rotate，用之前的 cos/sin 对当前点做绕 Z 轴旋转写 xy；否则直接抄原始 x、y
        if (rotate)
        {
            re_cloud->points[i].x = values_cloud[4 * i] * cs - values_cloud[4 * i + 1] * ss;
            re_cloud->points[i].y = values_cloud[4 * i] * ss + values_cloud[4 * i + 1] * cs;
        }
        else
        {
            re_cloud->points[i].x = values_cloud[4 * i];
            re_cloud->points[i].y = values_cloud[4 * i + 1];
        }

        // 抄原始 z 和 label
        re_cloud->points[i].z = values_cloud[4 * i + 2];
        re_cloud->points[i].label = sem_label;
    }
    return re_cloud;
}

pcl::PointCloud<pcl::PointXYZL>::Ptr SSC::getLCloud2(std::string file_cloud, std::string file_label, bool fixed_z, int down_sample_num, float scale)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr re_cloud(new pcl::PointCloud<pcl::PointXYZL>()); // 新建空的带语义标签点云

    // 打开标签文件
    std::ifstream in_label(file_label, std::ios::binary); 
    if (!in_label.is_open())
    {
        std::cerr << "No file:" << file_label << std::endl;
        exit(-1);
    }

    in_label.seekg(0, std::ios::end);
    uint32_t num_points = in_label.tellg() / sizeof(uint32_t); // 标签文件大小/uint32_t大小 = 点云数量
    in_label.seekg(0, std::ios::beg);
    std::vector<uint32_t> values_label(num_points);
    in_label.read((char *)&values_label[0], num_points * sizeof(uint32_t)); // 把整份标签读取到values_label中

    // 打开点云文件
    std::ifstream in_cloud(file_cloud, std::ios::binary);
    std::vector<float> values_cloud(4 * num_points);
    // 按点云每点 4 个 float 把整帧读进 values_cloud
    in_cloud.read((char *)&values_cloud[0], 4 * num_points * sizeof(float));
    
    uint32_t step = down_sample_num; // 既然你定义的 down_sample 是 /4，那么步长就是 4
    uint32_t down_sample = num_points / step;

    // re_cloud->points.resize(down_sample);
    // re_cloud->width = down_sample;
    // re_cloud->height = 1;

    uint32_t count = 0; // 存入新点云的独立计数器
    pcl::PointXYZL l_point;
    for (uint32_t i = 0; i < num_points && count < down_sample; i += step)
    {
        uint32_t sem_label;
        uint32_t raw_id = values_label[i] & 0x0000ffff;

        // 再次提醒：必须检查 label_map 边界
        if (remap && raw_id < label_map.size()) {
            sem_label = label_map[raw_id];
        } else {
            sem_label = remap ? 0 : values_label[i];
        }

        if(fixed_z){
            if(values_cloud[4 * i + 2] < -1 || values_cloud[4 * i + 2] > -0.5){
                continue;
            }
        }

        // 将数据填入 count 指向的位置，而不是 i
        if (sem_label == 0) {
            l_point.x = 0;
            l_point.y = 0;
            l_point.z = 0;
            l_point.label = 0;
        } else {
            l_point.x = values_cloud[4 * i] * scale;
            l_point.y = values_cloud[4 * i + 1] * scale;
            if(fixed_z){
                l_point.z = 0;
            }
            else{
                l_point.z = values_cloud[4 * i + 2] * scale;
            }
            l_point.label = sem_label;
        }
        re_cloud->points.push_back(l_point);
        count++; // 每次处理完一个采样点，移动目标指针
    }

    re_cloud->width = re_cloud->points.size();
    re_cloud->height = 1;
    std::cout<<"re_cloud size: "<<re_cloud->points.size()<<std::endl;
    return re_cloud;
}

pcl::PointCloud<pcl::PointXYZL>::Ptr SSC::getLCloud(std::string file_cloud)
{
    // 加载带标签的点云
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
    if (pcl::io::loadPCDFile<pcl::PointXYZL>(file_cloud, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s\n", file_cloud.c_str());
        return nullptr;
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr SSC::getColorCloud(pcl::PointCloud<pcl::PointXYZL>::Ptr &cloud_in)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    outcloud->points.resize(cloud_in->points.size());
    for (size_t i = 0; i < outcloud->points.size(); i++)
    {
        // 按每个点的语义设置颜色
        outcloud->points[i].x = cloud_in->points[i].x;
        outcloud->points[i].y = cloud_in->points[i].y;
        outcloud->points[i].z = cloud_in->points[i].z;
        outcloud->points[i].r = std::get<0>(_argmax_to_rgb[cloud_in->points[i].label]);
        outcloud->points[i].g = std::get<1>(_argmax_to_rgb[cloud_in->points[i].label]);
        outcloud->points[i].b = std::get<2>(_argmax_to_rgb[cloud_in->points[i].label]);
    }
    outcloud->height = 1;
    outcloud->width = outcloud->points.size();
    return outcloud;
}

cv::Mat SSC::project(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud)
{
    auto sector_step = 2. * M_PI / sectors_range; // 把圆均分成 sectors_range 个扇区

    // 新建一个 sectors_range x 1 的矩阵，用于存储每个扇区的距离、x、y、语义标签
    cv::Mat ssc_dis = cv::Mat::zeros(cv::Size(sectors_range, 1), CV_32FC4);

    for (uint i = 0; i < filtered_pointcloud->points.size(); i++)
    {
        auto label = filtered_pointcloud->points[i].label;
        // 只保留 label 为 13, 14, 16, 18, 19 的点对应 building, fence栅栏, pole电线杆, truck, traffic-sign交通标志
        if (label == 13 || label == 14 || label == 16 || label == 18 || label == 19)
        {
            float distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
            if (distance < 1e-2)
            {
                continue;
            }
            // int sector_id = cv::fastAtan2(filtered_pointcloud->points[i].y, filtered_pointcloud->points[i].x);
            // atan2(y,x) 返回的是 [-π,π] 之间的弧度值，表示点 (x,y) 相对于原点 (0,0) 的方位角
            float angle = M_PI + std::atan2(filtered_pointcloud->points[i].y, filtered_pointcloud->points[i].x);
            int sector_id = std::floor(angle / sector_step); // 将弧度值转换为扇区索引
            if (sector_id >= sectors_range || sector_id < 0)
                continue;
            // 仅当该扇区为空或当前点更近时才覆盖，使每扇区保留“最近点”，结果稳定且与遍历顺序无关
            float old_dis = ssc_dis.at<cv::Vec4f>(0, sector_id)[0];
            if (old_dis < 1e-6f || distance < old_dis)
            {
                ssc_dis.at<cv::Vec4f>(0, sector_id)[0] = distance;
                ssc_dis.at<cv::Vec4f>(0, sector_id)[1] = filtered_pointcloud->points[i].x;
                ssc_dis.at<cv::Vec4f>(0, sector_id)[2] = filtered_pointcloud->points[i].y;
                ssc_dis.at<cv::Vec4f>(0, sector_id)[3] = label;
            }
        }
    }
    return ssc_dis;
}

cv::Mat SSC::calculateSSC(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud)
{
    auto ring_step = (max_dis - min_dis) / rings; // 径向分割
    auto sector_step = 360. / sectors; // 方位角分割
    cv::Mat ssc = cv::Mat::zeros(cv::Size(sectors, rings), CV_8U); // ssc 为 rings × sectors 的 8 位图，初值为 0
    for (int i = 0; i < (int)filtered_pointcloud->points.size(); i++)
    {
        auto label = filtered_pointcloud->points[i].label;
        if (order_vec[label] > 0)
        {
            double distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);
            if (distance >= max_dis || distance < min_dis)
                continue;

            // cv::fastAtan2[0,2π]
            int sector_id = cv::fastAtan2(filtered_pointcloud->points[i].y, filtered_pointcloud->points[i].x)/ sector_step;
            int ring_id = (distance - min_dis) / ring_step;
            if (ring_id >= rings || ring_id < 0)
                continue;
            if (sector_id >= sectors || sector_id < 0)
                continue;

            // 若当前点的 order_vec 大于该格子已有标签的 order_vec，则用当前 label 覆盖该格子
            if (order_vec[label] > order_vec[ssc.at<unsigned char>(ring_id, sector_id)])
            {
                ssc.at<unsigned char>(ring_id, sector_id) = label;
            }
        }
    }
    return ssc;
}

cv::Mat SSC::getColorImage(cv::Mat &desc)
{
    cv::Mat out = cv::Mat::zeros(desc.size(), CV_8UC3);
    for (int i = 0; i < desc.rows; ++i)
    {
        for (int j = 0; j < desc.cols; ++j)
        {
            out.at<cv::Vec3b>(i, j)[0] = std::get<2>(_argmax_to_rgb[(int)desc.at<uchar>(i, j)]);
            out.at<cv::Vec3b>(i, j)[1] = std::get<1>(_argmax_to_rgb[(int)desc.at<uchar>(i, j)]);
            out.at<cv::Vec3b>(i, j)[2] = std::get<0>(_argmax_to_rgb[(int)desc.at<uchar>(i, j)]);
        }
    }
    return out;
}

void SSC::globalICP(cv::Mat &ssc_dis1, cv::Mat &ssc_dis2, double &angle, float &diff_x, float &diff_y)
{
    // angle：绕 Z 轴从 frame2 到 frame1 的大概旋转角
    // diff_x, diff_y：累计平移（米），表示把 frame2 平移多少能与 frame1 在 xy 上对齐

    // 1. 粗旋转 ssc_dis1，ssc_dis2 是1维描述子
    double similarity = 100000;
    int sectors = ssc_dis1.cols;
    for (int i = 0; i < sectors; ++i)
    {
        float dis_count = 0;
        for (int j = 0; j < sectors; ++j)
        {
            int new_col = j + i >= sectors ? j + i - sectors : j + i;
            // ssc_dis [distance, x, y, label],只对比距离
            cv::Vec4f vec1 = ssc_dis1.at<cv::Vec4f>(0, j);
            cv::Vec4f vec2 = ssc_dis2.at<cv::Vec4f>(0, new_col);
            // if(vec1[3]==vec2[3]){
            dis_count += fabs(vec1[0] - vec2[0]); // 只对比距离，
            // }
        }

        // 取最小距离对应的扇区偏转作为最佳偏转角度
        if (dis_count < similarity)
        {
            similarity = dis_count;
            angle = i;
        }
    }

    // 按照扇区移位计算粗略的z轴旋转角
    int angle_o = angle;
    angle = M_PI * (360. - angle * 360. / sectors) / 180.;

    // 用该角对ssc_dis2的 x y 做旋转
    auto cs = cos(angle);
    auto sn = sin(angle);
    auto temp_dis1 = ssc_dis1.clone();
    auto temp_dis2 = ssc_dis2.clone();
    for (int i = 0; i < sectors; ++i)
    {
        temp_dis2.at<cv::Vec4f>(0, i)[1] = ssc_dis2.at<cv::Vec4f>(0, i)[1] * cs - ssc_dis2.at<cv::Vec4f>(0, i)[2] * sn;
        temp_dis2.at<cv::Vec4f>(0, i)[2] = ssc_dis2.at<cv::Vec4f>(0, i)[1] * sn + ssc_dis2.at<cv::Vec4f>(0, i)[2] * cs;
    }

    // 迭代计算平移
    for (int i = 0; i < 100; ++i)
    {
        float dx = 0, dy = 0;
        int diff_count = 1;

        // 对 temp_dis1 的每个有效扇区
        for (int j = 0; j < sectors; ++j)
        {
            cv::Vec4f vec1 = temp_dis1.at<cv::Vec4f>(0, j);
            if (vec1[0] <= 0)
            {
                continue;
            }
            int min_id = -1;
            float min_dis = 1000000.;
            // 在 temp_dis2 里在 j + angle_o ± 10 扇区范围内找 xy 欧氏距离最近的一格（vec_temp[1],[2]），忽略 vec_temp[0] ≤ 0 的扇区
            for (int k = j + angle_o - 10; k < j + angle_o + 10; ++k)
            {
                cv::Vec4f vec_temp;
                int temp_id = k;
                if (k < 0)
                {
                    temp_id = k + sectors;
                }
                else if (k >= sectors)
                {
                    temp_id = k - sectors;
                }
                vec_temp = temp_dis2.at<cv::Vec4f>(0, temp_id);
                if (vec_temp[0] <= 0)
                {
                    continue;
                }
                float temp_dis = (vec1[1] - vec_temp[1]) * (vec1[1] - vec_temp[1]) + (vec1[2] - vec_temp[2]) * (vec1[2] - vec_temp[2]);
                if (temp_dis < min_dis)
                {
                    min_dis = temp_dis;
                    min_id = temp_id;
                }
            }
            if (min_id < 0)
            {
                continue;
            }

            // 只信任已经比较接近的点对，只把“已经比较对齐”的点对当作可靠对应
            // 若某对点当前 xy 差很大，说明可能是错误对应（最近点找错了，或该扇区本身不稳定）
            cv::Vec4f vec2 = temp_dis2.at<cv::Vec4f>(0, min_id);
            if (fabs(vec1[1] - vec2[1]) < 3 && fabs(vec1[2] - vec2[2]) < 3)
            {
                dx += vec1[1] - vec2[1];
                dy += vec1[2] - vec2[2];
                diff_count++;
            }
        }

        // 计算平均偏移量
        dx = 1. * dx / diff_count;
        dy = 1. * dy / diff_count;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

        for (int j = 0; j < sectors; ++j)
        {
            if (temp_dis2.at<cv::Vec4f>(0, j)[0] != 0)
            {
                // 把 temp_dis2 里所有有效点的 [1]、[2] 加上 (dx, dy)
                temp_dis2.at<cv::Vec4f>(0, j)[1] += dx;
                temp_dis2.at<cv::Vec4f>(0, j)[2] += dy;
                if (show)
                {
                    pcl::PointXYZRGB p;
                    p.x = temp_dis2.at<cv::Vec4f>(0, j)[1];
                    p.y = temp_dis2.at<cv::Vec4f>(0, j)[2];
                    p.z = 0;
                    p.r = std::get<0>(_argmax_to_rgb[(int)temp_dis2.at<cv::Vec4f>(0, j)[3]]);
                    p.g = std::get<1>(_argmax_to_rgb[(int)temp_dis2.at<cv::Vec4f>(0, j)[3]]);
                    p.b = std::get<2>(_argmax_to_rgb[(int)temp_dis2.at<cv::Vec4f>(0, j)[3]]);
                    temp_cloud->points.emplace_back(p);
                }
            }

            if (show && temp_dis1.at<cv::Vec4f>(0, j)[0] != 0)
            {
                pcl::PointXYZRGB p;
                p.x = temp_dis1.at<cv::Vec4f>(0, j)[1];
                p.y = temp_dis1.at<cv::Vec4f>(0, j)[2];
                p.z = 0;
                p.r = std::get<0>(_argmax_to_rgb[(int)temp_dis1.at<cv::Vec4f>(0, j)[3]]);
                p.g = std::get<1>(_argmax_to_rgb[(int)temp_dis1.at<cv::Vec4f>(0, j)[3]]);
                p.b = std::get<2>(_argmax_to_rgb[(int)temp_dis1.at<cv::Vec4f>(0, j)[3]]);
                temp_cloud->points.emplace_back(p);
            }
        }
        if (show)
        {
            temp_cloud->height = 1;
            temp_cloud->width = temp_cloud->points.size();
            // viewer->showCloud(temp_cloud);
            usleep(1000000);
        }

        diff_x += dx;
        diff_y += dy;
        if (show)
        {
            std::cout << i << " diff " << diff_x << " " << diff_y << " " << dx << " " << dy << std::endl;
        }
        
        // 若偏移量足够小，则停止迭代
        if (fabs(dx) < 1e-5 && fabs(dy) < 1e-5)
        {
            break;
        }
    }
}

Eigen::Matrix4f SSC::globalICP(cv::Mat &ssc_dis1, cv::Mat &ssc_dis2){
    // 1. 用扇区距离估计粗旋转角
    double similarity = 100000;
    float angle=0;
    int sectors = ssc_dis1.cols;
    for (int i = 0; i < sectors; ++i)
    {
        float dis_count = 0;
        for (int j = 0; j < sectors; ++j)
        {
            int new_col = j + i >= sectors ? j + i - sectors : j + i;
            cv::Vec4f vec1 = ssc_dis1.at<cv::Vec4f>(0, j);
            cv::Vec4f vec2 = ssc_dis2.at<cv::Vec4f>(0, new_col);
            // if(vec1[3]==vec2[3]){
            dis_count += fabs(vec1[0] - vec2[0]);
            // }
        }
        if (dis_count < similarity)
        {
            similarity = dis_count;
            angle = i;
        }
    }
    angle = M_PI * (360. - angle * 360. / sectors) / 180.;
    auto cs = cos(angle);
    auto sn = sin(angle);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>),cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < sectors; ++i)
    {
        // 把ssc_dis1的每个有效扇区点 x y 压入cloud1
        if(ssc_dis1.at<cv::Vec4f>(0, i)[3]>0){
            cloud1->push_back(pcl::PointXYZ(ssc_dis1.at<cv::Vec4f>(0, i)[1],ssc_dis1.at<cv::Vec4f>(0, i)[2],0.));
        }

        // 把ssc_dis2的每个有效扇区点 x y 压入cloud2
        if(ssc_dis2.at<cv::Vec4f>(0, i)[3]>0){
            float tpx = ssc_dis2.at<cv::Vec4f>(0, i)[1] * cs - ssc_dis2.at<cv::Vec4f>(0, i)[2] * sn;
            float tpy = ssc_dis2.at<cv::Vec4f>(0, i)[1] * sn + ssc_dis2.at<cv::Vec4f>(0, i)[2] * cs;
            cloud2->push_back(pcl::PointXYZ(tpx,tpy,0.));
        }
    }

    // 2. 用 ICP 细化旋转和平移
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud2);
    icp.setInputTarget(cloud1);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    auto trans=icp.getFinalTransformation();
    Eigen::Affine3f trans1 = Eigen::Affine3f::Identity();
    trans1.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
    return trans*trans1.matrix(); // 把之前的粗旋转和ICP细化旋转合并
}

double SSC::calculateSim(cv::Mat &desc1, cv::Mat &desc2)
{
    double similarity = 0;
    int sectors = desc1.cols;
    int rings = desc1.rows;
    int valid_num = 0;
    for (int p = 0; p < sectors; p++)
    {
        for (int q = 0; q < rings; q++)
        {
            if (desc1.at<unsigned char>(q, p) == 0 && desc2.at<unsigned char>(q, p) == 0)
            {
                continue;
            }

            valid_num++; // 该格至少有一幅有语义，算“有效格”

            if (desc1.at<unsigned char>(q, p) == desc2.at<unsigned char>(q, p))
            {
                // 语义相同
                similarity++;
            }
        }
    }
    // std::cout<<similarity<<std::endl;
    return similarity / valid_num; // 有效格的语义相同比例
}

double SSC::getScore(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZL>::Ptr cloud2, double &angle, float &diff_x, float &diff_y)
{
    angle = 0;
    diff_x = 0;
    diff_y = 0;
    // 计算语义点云1维描述子
    cv::Mat ssc_dis1 = project(cloud1);
    cv::Mat ssc_dis2 = project(cloud2);
    // 估计ssc2到ssc1的粗旋转与累计平移
    globalICP(ssc_dis1, ssc_dis2, angle, diff_x, diff_y);
    // 若平移量太大，则认为匹配失败
    if (fabs(diff_x)>5 || fabs(diff_y) > 5)
    {
        diff_x = 0;
        diff_y = 0;
    }
    // 把粗旋转和平移合并
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() << diff_x, diff_y, 0;
    transform.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));

    // 把ssc2平移到ssc1的位置
    pcl::PointCloud<pcl::PointXYZL>::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    transformPointCloud(*cloud2, *trans_cloud, transform);

    // 计算语义点云2维描述子
    auto desc1 = calculateSSC(cloud1);
    auto desc2 = calculateSSC(trans_cloud);
    // 计算相似度得分
    auto score = calculateSim(desc1, desc2);
    if (show)
    {
        transform.translation() << diff_x, diff_y, 0;
        transformPointCloud(*cloud2, *trans_cloud, transform);
        auto color_cloud1 = getColorCloud(cloud1);
        auto color_cloud2 = getColorCloud(trans_cloud);
        *color_cloud2 += *color_cloud1;
        // viewer->showCloud(color_cloud2);
        auto color_image1 = getColorImage(desc1);
        auto color_image2 = getColorImage(desc2);
        // cv::imshow("color image1", color_image1);
        // cv::imshow("color image2", color_image2);
        // 保存点云与 SSC
        if (!show_save_dir_.empty())
        {
            std::string mkdir_cmd = "mkdir -p " + show_save_dir_;
            if (std::system(mkdir_cmd.c_str()) == 0)
            {
                std::string ply_path = show_save_dir_ + "/cloud_merged.ply";
                if (!color_cloud2->points.empty())
                    pcl::io::savePLYFileBinary(ply_path, *color_cloud2);
                cv::imwrite(show_save_dir_ + "/ssc1.png", color_image1);
                cv::imwrite(show_save_dir_ + "/ssc2.png", color_image2);
            }
        }
        // cv::waitKey(0);
    }

    return score;
}

double SSC::getScore(pcl::PointCloud<pcl::PointXYZL>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZL>::Ptr cloud2, Eigen::Matrix4f& transform)
{
    cv::Mat ssc_dis1 = project(cloud1);
    cv::Mat ssc_dis2 = project(cloud2);
    transform=globalICP(ssc_dis1, ssc_dis2);
    pcl::PointCloud<pcl::PointXYZL>::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZL>);
    transformPointCloud(*cloud2, *trans_cloud, transform);
    auto desc1 = calculateSSC(cloud1);
    auto desc2 = calculateSSC(trans_cloud);
    auto score = calculateSim(desc1, desc2);
    if (show)
    {
        transform(2,3)=0.;
        transformPointCloud(*cloud2, *trans_cloud, transform);

        auto color_cloud1 = getColorCloud(cloud1);
        auto origin_color_cloud2 = getColorCloud(cloud2);

        std::string ply_path_origin_cloud1 = show_save_dir_ + "/origin_color_cloud1.ply";
        std::string ply_path_origin_cloud2 = show_save_dir_ + "/origin_color_cloud2.ply";
        if (!color_cloud1->points.empty() && !origin_color_cloud2->points.empty()){
            pcl::io::savePLYFileBinary(ply_path_origin_cloud1, *color_cloud1);
            pcl::io::savePLYFileBinary(ply_path_origin_cloud2, *origin_color_cloud2);
        }

        auto color_cloud2 = getColorCloud(trans_cloud);
        *color_cloud2 += *color_cloud1;
        // viewer->showCloud(color_cloud2);
        auto color_image1 = getColorImage(desc1);
        auto color_image2 = getColorImage(desc2);
        // cv::imshow("color image1", color_image1);
        // cv::imshow("color image2", color_image2);
        // 保存点云与 SSC
        if (!show_save_dir_.empty())
        {
            std::string mkdir_cmd = "mkdir -p " + show_save_dir_;
            if (std::system(mkdir_cmd.c_str()) == 0)
            {
                std::string ply_path = show_save_dir_ + "/cloud_merged.ply";
                if (!color_cloud2->points.empty())
                    pcl::io::savePLYFileBinary(ply_path, *color_cloud2);
                cv::imwrite(show_save_dir_ + "/ssc1.png", color_image1);
                cv::imwrite(show_save_dir_ + "/ssc2.png", color_image2);
            }
        }
        // cv::waitKey(0);
    }
    return score;
}

double SSC::getScore(std::string cloud_file1, std::string cloud_file2, std::string label_file1, std::string label_file2, double &angle, float &diff_x, float &diff_y)
{
    // 加载带语义标签的点云
    auto cloudl1 = getLCloud(cloud_file1, label_file1);
    auto cloudl2 = getLCloud(cloud_file2, label_file2);
    // 计算相似度得分
    auto score = getScore(cloudl1, cloudl2, angle, diff_x, diff_y);
    return score;
}

double SSC::getScore(std::string cloud_file1, std::string cloud_file2, std::string label_file1, std::string label_file2, Eigen::Matrix4f& transform, bool fixed_z, int down_sample_num, float scale)
{
    auto cloudl1 = getLCloud2(cloud_file1, label_file1, fixed_z, down_sample_num, scale);
    auto cloudl2 = getLCloud2(cloud_file2, label_file2, fixed_z, down_sample_num, scale);
    auto score = getScore(cloudl1, cloudl2, transform);
    return score;
}


double SSC::getScore(std::string cloud_file1, std::string cloud_file2, double &angle, float &diff_x, float &diff_y)
{
    auto cloudl1 = getLCloud(cloud_file1);
    auto cloudl2 = getLCloud(cloud_file2);
    auto score = getScore(cloudl1, cloudl2, angle, diff_x, diff_y);
    return score;
}

double SSC::getScore(std::string cloud_file1, std::string cloud_file2, Eigen::Matrix4f& transform)
{
    auto cloudl1 = getLCloud(cloud_file1);
    auto cloudl2 = getLCloud(cloud_file2);
    auto score = getScore(cloudl1, cloudl2, transform);
    return score;
}
