
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

    void initDetect_thread(void);
    std::vector<PredictResult> detect(const cv::Mat & input); //阻塞方式的调用 

    cv::Mat inputImage(const cv::Mat & input);
    std::vector<Conical> findConicals(const cv::Mat & hsv_img);
    std::vector<Sign> findSigns(const cv::Mat & hsv_img);
    std::vector<Obstacle> findObstacles(const cv::Mat & hsv_img);
    Tool_man thinObstacles(std::vector<Obstacle> obstacles);

    std::vector<PredictResult> packer(std::vector<Conical> & conicals, std::vector<Sign> & signs, std::vector<Cross> & cross, std::vector<Barrier> & barriers);
    // For debug usage
    void output_dataset(cv::Mat & roi);
    cv::Mat drawResults(cv::Mat & img);
    void showRect(cv::Mat img, cv::RotatedRect rotatedRect, cv::Scalar color);

    Params params;

    std::unique_ptr<SignClassifier> sign_classifier;
    std::unique_ptr<SignClassifier> obstacle_classifier;

    // Marks
    std::vector<Conical> conicals;
    std::vector<Sign> signs;
    std::vector<Obstacle> obstacles;
    std::vector<Cross> cross;
    std::vector<Barrier> barriers;

    std::vector<PredictResult> result;
    // Mats
    cv::Mat resized_img;
    cv::Mat hsv_img;
    cv::Mat Conical_mask;
    cv::Mat Sign_mask;
    cv::Mat Obstacle_mask;

private:
    void start_findConicals(void);
    void start_findSigns(void);
    void start_findObstacles(void);
    void start_packer(void);
    cv::RotatedRect conicalRect(std::vector<cv::Point> contour);
    bool isConical(const Conical & conical, const Params & params);
    bool isObstacle(const Obstacle & obstacle, const Params & params);
    bool isSign(const Sign & sign, const Params & params);
    bool isBarrier(const Barrier & barrier, const Params & params);
    bool isCross(const Cross & cros, const Params & params);
    float w_judge(cv::RotatedRect & r_rect, cv::Mat & result);
};

#endif // !DETECTOR_HPP_
