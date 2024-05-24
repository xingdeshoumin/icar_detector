
// OpenCV
#include <opencv2/opencv.hpp>

// STD
#include <cstdio>
#include <ctime>
#include <chrono>

#include "icar_detector/detector.hpp"

Detector::Detector(const Params & input_params) : params(input_params){}

std::vector<PredictResult> Detector::detect(const cv::Mat & input)
{
    hsv_img = preprocessImage(input);
    conicals = findConicals(hsv_img);
    signs = findSigns(hsv_img);
    obstacles = findObstacles(hsv_img);

    if (!signs.empty()){
        sign_classifier->sign_classify(signs);
    }
    if (!obstacles.empty()){
        obstacle_classifier->obstacles_classify(obstacles);
    }

    // 这堆东西到通信包的转换
    return packer(conicals, signs);
}

std::vector<PredictResult> Detector::packer(std::vector<Conical> conicals, std::vector<Sign> signs)
{
    cv::Point2f p[4];
    std::vector<PredictResult> predict_results;
    // conicals
    for (const auto & conical : conicals) {
        conical.points(p);
        predict_results.push_back({3, "LABEL_CONE", 1.2, (int)p[0].x, (int)p[0].y, (int)conical.width, (int)conical.length});
    }
    // signs
    for (const auto & sign : signs) {
        sign.points(p);
        if (sign.mlp.label_id == 0)
            predict_results.push_back({0, "LABEL_BOMB", sign.mlp.confidence, (int)p[0].x, (int)p[0].y, (int)sign.width, (int)sign.length});
        if (sign.mlp.label_id == 1)
            predict_results.push_back({8, "LABEL_PATIENT", sign.mlp.confidence, (int)p[0].x, (int)p[0].y, (int)sign.width, (int)sign.length});
        if (sign.mlp.label_id == 3)
            predict_results.push_back({6, "LABEL_EVIL", sign.mlp.confidence, (int)p[0].x, (int)p[0].y, (int)sign.width, (int)sign.length});
        if (sign.mlp.label_id == 4)
            predict_results.push_back({12, "LABEL_TUMBLE", sign.mlp.confidence, (int)p[0].x, (int)p[0].y, (int)sign.width, (int)sign.length});
        if (sign.mlp.label_id == 5)
            predict_results.push_back({11, "LABEL_THIEF", sign.mlp.confidence, (int)p[0].x, (int)p[0].y, (int)sign.width, (int)sign.length});
    }

    return predict_results;

}

cv::Mat Detector::preprocessImage(const cv::Mat & rgb_img)
{
    cv::resize(rgb_img, resized_img, cv::Size(640, 480));
    // 转换色彩空间HSV
    cv::cvtColor(resized_img, hsv_img, cv::COLOR_BGR2HSV);

    return hsv_img;
}

std::vector<Conical> Detector::findConicals(const cv::Mat & hsv_img)
{
    // HSV 筛选mask
    cv::Scalar Conical_LowerLimit(params.Conical_LowerLimit_LH, params.Conical_LowerLimit_LS, params.Conical_LowerLimit_LV);
    cv::Scalar Conical_UpperLimit(params.Conical_LowerLimit_UH, params.Conical_LowerLimit_US, params.Conical_LowerLimit_UV);
    cv::inRange(hsv_img, Conical_LowerLimit, Conical_UpperLimit, Conical_mask);
    // mask膨胀
    cv::Mat Conical_dilate_mask;
    cv::Mat Conical_mask_kernel = cv::Mat::ones(params.Conical_kernel, params.Conical_kernel, CV_8U);
    cv::dilate(Conical_mask, Conical_dilate_mask, Conical_mask_kernel);
    // 寻找mask轮廓
    cv::findContours(Conical_dilate_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 轮廓检测是否为锥桶
    std::vector<Conical> conicals;
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Conical_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        // auto r_rect = cv::minAreaRect(contour);
        auto conical = Conical(r_rect);
        
        if (isConical(conical, params)){
            conicals.emplace_back(conical);
        }
    }

    return conicals;
}

std::vector<Sign> Detector::findSigns(const cv::Mat & hsv_img)
{
    // HSV 筛选mask
    cv::Scalar Sign_LowerLimit1(params.Sign_LowerLimit_LH1, params.Sign_LowerLimit_LS1, params.Sign_LowerLimit_LV1);
    cv::Scalar Sign_UpperLimit1(params.Sign_LowerLimit_UH1, params.Sign_LowerLimit_US1, params.Sign_LowerLimit_UV1);
    cv::Mat Sign_mask1;
    cv::inRange(hsv_img, Sign_LowerLimit1, Sign_UpperLimit1, Sign_mask1);
    cv::Scalar Sign_LowerLimit2(params.Sign_LowerLimit_LH2, params.Sign_LowerLimit_LS2, params.Sign_LowerLimit_LV2);
    cv::Scalar Sign_UpperLimit2(params.Sign_LowerLimit_UH2, params.Sign_LowerLimit_US2, params.Sign_LowerLimit_UV2);
    cv::Mat Sign_mask2;
    cv::inRange(hsv_img, Sign_LowerLimit2, Sign_UpperLimit2, Sign_mask2);
    cv::bitwise_or(Sign_mask1, Sign_mask2, Sign_mask);
    // mask膨胀
    cv::Mat Sign_dilate_mask;
    cv::Mat Sign_mask_kernel = cv::Mat::ones(params.Sign_kernel, params.Sign_kernel, CV_8U);
    cv::dilate(Sign_mask, Sign_dilate_mask, Sign_mask_kernel);

    // 寻找mask轮廓
    cv::findContours(Sign_dilate_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 轮廓检测是否为路牌
    std::vector<Sign> signs;
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Sign_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        auto sign = Sign(r_rect);
        
        if (isSign(sign, params)){
            // // 使用矩形框裁剪二值化图像
            cv::Mat roi = Sign_mask(r_rect.boundingRect());

            cv::resize(roi, roi, cv::Size(28, 28));

            if (params.OUPUT_DATASET_Signs){
                output_dataset(roi);
            }
            sign.mlp.sign_img = roi.clone();
            signs.emplace_back(sign);
        }
    }
    return signs;
}

std::vector<Obstacle> Detector::findObstacles(const cv::Mat & hsv_img)
{
    // HSV 筛选mask
    cv::Scalar Obstacle_LowerLimit(params.Obstacle_LowerLimit_LH, params.Obstacle_LowerLimit_LS, params.Obstacle_LowerLimit_LV);
    cv::Scalar Obstacle_UpperLimit(params.Obstacle_LowerLimit_UH, params.Obstacle_LowerLimit_US, params.Obstacle_LowerLimit_UV);
    cv::inRange(hsv_img, Obstacle_LowerLimit, Obstacle_UpperLimit, Obstacle_mask);
    // mask膨胀
    cv::Mat Obstacle_dilate_mask;
    cv::Mat Obstacle_mask_kernel = cv::Mat::ones(params.Obstacle_kernel, params.Obstacle_kernel, CV_8U);
    cv::dilate(Obstacle_mask, Obstacle_dilate_mask, Obstacle_mask_kernel);
    // 寻找mask轮廓
    cv::findContours(Obstacle_dilate_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 轮廓检测是否为锥桶
    std::vector<Obstacle> obstacles;
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Obstacle_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        // auto r_rect = cv::minAreaRect(contour);
        auto obstacle = Obstacle(r_rect);
        
        if (isObstacle(obstacle, params)){
            cv::Mat roi = Obstacle_mask(r_rect.boundingRect());
            cv::resize(roi, roi, cv::Size(28, 28));
            // // 白色占比判断
            // if (w_judge(roi) > params.Obstacle_w_judge)
            if (params.OUPUT_DATASET_Obstacles){
                output_dataset(roi);
            }
            obstacle.mlp.sign_img = roi.clone();
            obstacles.emplace_back(obstacle);
        }
    }
    
    // 判断斑马线
    // cross = Cross_times_count(Obstacle_mask);

    return obstacles;
}

void Detector::output_dataset(cv::Mat & roi)
{
    // 使用系统时间来命名你的图片
    auto now = std::chrono::system_clock::now();
    auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
    auto value = now_ns.time_since_epoch();
    long duration = value.count();
    std::string filename = "./icar_detector/image/" + std::to_string(duration) + ".jpg";

    std::cout << filename << " saved" << std::endl;
    cv::imwrite(filename, roi);
}

cv::RotatedRect Detector::conicalRect(std::vector<cv::Point> contour)
{
    int Xmin = contour[0].x;
    int Xmax = contour[0].x;
    int Ymin = contour[0].y;
    int Ymax = contour[0].y;

    // 遍历数组，查找最大值和最小值
    for (int i = 0; i < (int)contour.size(); ++i) {
        Xmin = std::min(Xmin, contour[i].x);
        Xmax = std::max(Xmax, contour[i].x);
        Ymin = std::min(Ymin, contour[i].y);
        Ymax = std::max(Ymax, contour[i].y);
    }

    return cv::RotatedRect(cv::Point(Xmin,Ymin), cv::Point(Xmax,Ymin), cv::Point(Xmax,Ymax));
}

bool Detector::isConical(const Conical & conical, const Params & params)
{
    float ratio = conical.width / conical.length;
    bool ratio_ok = params.Conical_min_ratio < ratio && ratio < params.Conical_max_ratio;
    
    if (ratio_ok){
        if (params.SHOW_LOGS) std::cout << "isConical: ratio: " << ratio << std::endl;
    }

    return ratio_ok;
}

bool Detector::isSign(const Sign & sign, const Params & params)
{
    float ratio = sign.width / sign.length;
    bool ratio_ok = params.Sign_min_ratio < ratio && ratio < params.Sign_max_ratio;

    if (ratio_ok){
        if (params.SHOW_LOGS) std::cout << "isSign: ratio: " << ratio << std::endl;
    }

    return ratio_ok;
}

bool Detector::isObstacle(const Obstacle & obstacle, const Params & params){
    float ratio = obstacle.width / obstacle.length;
    bool ratio_ok = params.Obstacle_min_ratio < ratio && ratio < params.Obstacle_max_ratio;

    if (ratio_ok){
        if (params.SHOW_LOGS) std::cout << "isObstacle: ratio: " << ratio << std::endl;
    }

    return ratio_ok;
}

std::vector<Cross> Detector::Cross_times_count(cv::Mat & Obstacle_mask)
{
    std::vector<Cross> cross;
    std::vector<cv::Point> points;
    int count = 0;
    for (int i = 0; i < Obstacle_mask.rows; ++i) {
          
        int transition = 0;
        for (int j = 1; j < Obstacle_mask.cols; ++j) {
            if (Obstacle_mask.at<uchar>(i, j-1) == 0 && Obstacle_mask.at<uchar>(i, j) == 255) {
                points.push_back(cv::Point(j, i));
                transition++;
            }
        }
        if (transition > params.Cross_times) {
            
            count++;
        }
    }
    if (count > 20)
    {
        // 使用minAreaRect找到包围这些点的最小面积矩形  
        std::cout << "Cross_count: " << count << std::endl;
        cv::RotatedRect rotatedRect = cv::minAreaRect(points);  
            auto cros = Cross(rotatedRect);
            cross.emplace_back(cros);
    }
    return cross;
}

double Detector::w_judge(cv::Mat & Obstacle_mask)
{
    int whitePixelCount = 0;
    int totalPixelCount = Obstacle_mask.rows * Obstacle_mask.cols;

    // 遍历图像，统计白色像素数量  
    for (int y = 0; y < Obstacle_mask.rows; ++y) {  
        for (int x = 0; x < Obstacle_mask.cols; ++x) {  
            if (Obstacle_mask.at<uchar>(y, x) > 240) { // 假设亮度大于240的像素为白色  
                ++whitePixelCount;  
            }  
        }  
    }  
    // 计算白色像素占比  
    float whiteRatio = static_cast<float>(whitePixelCount) / totalPixelCount;  
    if (params.SHOW_LOGS) std::cout << "Obstacle_whiteRatio: " << whiteRatio << std::endl;
    return whiteRatio;  
}

void Detector::drawResults(cv::Mat & img)
{
    for (const auto & conical : conicals) {
        showRect(img, conical, cv::Scalar(0,255,255));
    }
    for (const auto & obstacle : obstacles) {
        cv::putText(
        img, obstacle.mlp.classfication_result, obstacle.bottom, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(255,255,0), 2);

        showRect(img, obstacle, cv::Scalar(255,255,0));
    }
    for (const auto & cros : cross) {
        showRect(img, cros, cv::Scalar(255,0,255));
    }
    for (auto & sign : signs){
        cv::putText(
        img, sign.mlp.classfication_result, sign.bottom, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0,255,0), 2);

        showRect(img, sign, cv::Scalar(0,255,0));
    }
}

void Detector::showRect(cv::Mat img, cv::RotatedRect rotatedRect, cv::Scalar color)
{
    // 获取旋转矩形的四个顶点
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // 在图像上绘制旋转矩形
    for (int i = 0; i < 4; i++){
        cv::line(img, vertices[i], vertices[(i+1)%4], color, 2);
    }
}
