
// OpenCV
#include <opencv2/opencv.hpp>

// STD
#include <cstdio>
#include <ctime>
#include <chrono>
#include <memory>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "icar_detector/detector.hpp"

void showRect(cv::Mat img, cv::RotatedRect rotatedRect, cv::Scalar color)
{
    // 获取旋转矩形的四个顶点
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // 在图像上绘制旋转矩形
    for (int i = 0; i < 4; i++){
        cv::line(img, vertices[i], vertices[(i+1)%4], color, 2);
    }
}

int main()
{
    auto config = Config("./icar_detector/config/detecor_config.json");
    
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    // 加载图像
    cv::Mat scr = cv::imread("/mnt/d/Chromedownload/smart-car/edgeboard/res/samples/train/204.jpg");
    // cv::Mat scr = cv::imread("/mnt/d/Chromedownload/mee/mee/20.jpg");

    auto pkg_path = ament_index_cpp::get_package_share_directory("icar_detector");
    pkg_path = pkg_path.substr(0, pkg_path.find("icar_detector")) + "icar_detector" + "/icar_detector";
    std::cout << "pkg_path: " << pkg_path << std::endl;
    auto model_path = pkg_path + "/model/mlp.onnx";
    auto label_path = pkg_path + "/model/label.txt";
    float threshold = 0.6f;
    std::vector<std::string> Class_names_ = {"injured", "negative", "robber", "slipping", "thief"};
    std::vector<std::string> ignore_classes = {"negative"};
    
    Detector detector(config.params);
    detector.classifier =
    std::make_unique<SignClassifier>(model_path, label_path, threshold, ignore_classes);
    detector.detect(scr);
    cv::Mat resized_img = detector.resized_img;
    cv::Mat hsv = detector.hsv_img;

    //********************************************** 锥桶识别 *****************************************************

    cv::Mat Conical_mask = detector.Conical_mask;

    //********************************************** 爆炸物识别 *****************************************************
    
    cv::Mat Boom_mask = detector.Boom_mask;

    //********************************************** 路牌识别 *****************************************************

    cv::Mat Sign_mask = detector.Sign_mask;
    //********************************************** 结果输出 *****************************************************
    cv::Mat Conical_result;
    cv::cvtColor(Conical_mask, Conical_result, cv::COLOR_GRAY2BGR);

    cv::Mat Boom_result;
    cv::cvtColor(Boom_mask, Boom_result, cv::COLOR_GRAY2BGR);
    // cv::drawContours(Boom_result, contours, -1, cv::Scalar(0, 0, 255), 2);

    cv::Mat Sign_result;
    cv::cvtColor(Sign_mask, Sign_result, cv::COLOR_GRAY2BGR);

    for (const auto & conical : detector.conicals) {
        showRect(resized_img, conical, cv::Scalar(0,255,255));
    }
    for (const auto & boom : detector.booms) {
        showRect(resized_img, boom, cv::Scalar(0,0,255));
    }
    for (auto & sign : detector.signs){
        // 获取索引为 label_id 的类别名称
        std::string class_name = Class_names_[sign.label_id];
        // 获取类别名称的第一个字母
        char first_letter = class_name[0];
        // 在图像上绘制第一个字母
        cv::putText(resized_img, std::string(1, first_letter), sign.center, cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 0), 4);

        showRect(resized_img, sign, cv::Scalar(0,255,0));
    }
    
    cv::imshow("resized_img", resized_img);
    // cv::imshow("Conical_result", Conical_result);
    // cv::imshow("Boom_result", Boom_result);
    // cv::imshow("Sign_result", Sign_result);

    cv::waitKey(0);
    return 0;
}

