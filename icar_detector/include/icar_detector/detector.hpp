
#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>

#include "icar_detector/Marks.hpp"
#include "icar_detector/sign_classifier.hpp"


enum Label {
    LABEL_BOMB=0,           // AI标签：爆炸物√
    LABEL_BRIDGE=1,         // AI标签：坡道
    LABEL_SAFETY=2,         // AI标签：普通车辆：没有
    LABEL_CONE=3,           // AI标签：锥桶√
    LABEL_CROSSWALK=4,      // AI标签：斑马线----------------
    LABEL_DANGER=5,         // AI标签：危险车辆：没有
    LABEL_EVIL=6,           // AI标签：恐怖分子：牌子√
    LABEL_BLOCK=7,          // AI标签：障碍物:黑块------------
    LABEL_PATIENT=8,        // AI标签：伤员：牌子√
    LABEL_PROP=9,           // AI标签：道具车：没有
    LABEL_SPY=10,           // AI标签：嫌疑车辆：没有
    LABEL_THIEF=11,         // AI标签：盗贼：牌子√
    LABEL_TUMBLE=12         // AI标签：跌倒：牌子√
};

/**
 * @brief 目标检测结果
 *
 */
struct PredictResult
{
    int type;          // ID
    std::string label; // 标签
    float score;       // 置信度
    int x;             // 坐标
    int y;             // 坐标
    int width;         // 尺寸
    int height;        // 尺寸
};

class Detector
{
public:
    Detector(const Params & input_params);

    std::vector<PredictResult> detect(const cv::Mat & input);

    cv::Mat preprocessImage(const cv::Mat & input);
    std::vector<Conical> findConicals(const cv::Mat & hsv_img);
    std::vector<Sign> findSigns(const cv::Mat & hsv_img);
    std::vector<Obstacle> findObstacles(const cv::Mat & hsv_img);

    std::vector<PredictResult> packer(std::vector<Conical> conicals, std::vector<Sign> signs);

    // For debug usage
    void drawResults(cv::Mat & img);
    void showRect(cv::Mat img, cv::RotatedRect rotatedRect, cv::Scalar color);

    Params params;

    std::unique_ptr<SignClassifier> classifier;

    // Debug msgs
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // Marks
    std::vector<Conical> conicals;
    std::vector<Obstacle> obstacles;
    std::vector<Sign> signs;
    std::vector<Cross> cross;
    // Mats
    cv::Mat resized_img;
    cv::Mat hsv_img;
    cv::Mat Conical_mask;
    cv::Mat Boom_mask;
    cv::Mat Sign_mask;
    cv::Mat Obstacle_mask;

private:
    cv::RotatedRect conicalRect(std::vector<cv::Point> contour);
    bool isConical(const Conical & conical, const Params & params);
    bool isObstacle(const Obstacle & obstacle, const Params & params);
    bool isSign(const Sign & sign, const Params & params);
    double w_judge(cv::Mat & Obstacle_mask);
    std::vector<Cross> Cross_times_count(cv::Mat & Obstacle_mask);
};

#endif // !DETECTOR_HPP_
