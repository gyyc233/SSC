#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "ssc.h"
#include <iomanip>
#include <sstream>


std::string format_index(int index, int width = 10) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(width) << index;
    return ss.str();
}

void zfill(std::string& in_str,int len){
    // 若帧号不足len位，则补齐0
        while (in_str.size() < len)
        {
            in_str = "0" + in_str;
        }
}
int main(int argc,char** argv){
    std::string conf_file="../config/config_kitti_origin_single.yaml";
    if(argc>1){
        conf_file=argv[1];
    }
    auto data_cfg = YAML::LoadFile(conf_file);
    auto cloud_path=data_cfg["eval_loopclosure"]["cloud_path"].as<std::string>();
    auto label_path=data_cfg["eval_loopclosure"]["label_path"].as<std::string>();
    auto out_file=data_cfg["eval_loopclosure"]["out_file"].as<std::string>();
    auto file_name_length=data_cfg["file_name_length"].as<int>();

    std::ofstream result_file;
        result_file.open(out_file, std::ios::out);
        if (result_file.is_open()) {
            result_file << "query_id,match_id,candidate_id,score,rotation,x,y,dis" << std::endl;
        }

    std::vector<std::string> cloud_files, sem_files;
    SSC ssc(conf_file);

    for(int i=0;i<2700;i++){
        std::string filename_id = format_index(i);
        std::string cloud_file, sem_file;
        cloud_file = cloud_path + filename_id;
        cloud_file = cloud_file + ".bin";

        std::string sem_id = format_index(i,6);
        sem_file = label_path + sem_id;
        sem_file = sem_file + ".label";

        cloud_files.push_back(cloud_file);
        sem_files.push_back(sem_file);
    }

    for(int i=400;i<2700;i+=5){
        if(i<1200 || (i>1600 && i< 2300) || i>2650)
        {
            continue;
        }

        int current_frame_id = i;
        int best_match_id = -1;
        int candidate_id = -1;
        double max_score = 0;
        double rotation = 0;
        double x = 0;
        double y = 0;
        double dist = 0;

        std::cout << "已处理至第 " << i << " 帧..." << std::endl;
        int j_min = 0;
        int j_max = 0 ;
        if(i>1200 && i<1600){
            j_min = 350; // 550
            j_max = 1000; // 899
        }
        else if(i>2200 && i<2600){
            j_min = 0;
            j_max = 350; // 150
        }
        else if(i>2450 && i<2700){
            j_min = 600; // 800
            j_max = 1100; // 900
        }

        float scale = 0.1f;
        for(int j =j_min; j<j_max;j+=1){
            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
            double score=ssc.getScore(cloud_files[i],cloud_files[j],sem_files[i],sem_files[j],transform, true, 60,scale);
            return 0;

            if(score > max_score){
                max_score = score;
                best_match_id = j;
                candidate_id = j;

                rotation = atan2(transform(1, 0), transform(0, 0)); // Yaw
                x = transform(0, 3);
                y = transform(1, 3);
                dist = sqrt(pow(x,2)+pow(y,2));
            }
        }

        if (result_file.is_open() && max_score !=0) {        
            if(max_score < 0.7){
                best_match_id = -1;
            }

            result_file << current_frame_id << "," 
                            << best_match_id << "," 
                            << candidate_id << "," 
                            << max_score << "," 
                            << rotation << "," 
                            << x << "," 
                            << y << ","
                            << dist << std::endl;
        }
    }

    result_file.close();
    return 0;
}