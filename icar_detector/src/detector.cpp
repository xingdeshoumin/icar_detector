
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
    findConicals(hsv_img);
    findBooms(hsv_img);
    findSigns(hsv_img);

    if (!signs.empty()){
        classifier->classify(signs);
    }

    // 这堆东西到通信包的转换
    

    return packer(conicals, booms, signs);
}

std::vector<PredictResult> Detector::packer(std::vector<Conical> conicals, std::vector<Boom> booms, std::vector<Sign> signs)
{
    cv::Point2f p[4];
    std::vector<PredictResult> predict_results;
    // conicals
    for (const auto & conical : conicals) {
        conical.points(p);
        predict_results.push_back({3, "xxx", 1.2, p[0].x, p[0].y, conical.width, conical.length});
    }
    // booms
    for (const auto & boom : booms) {
        boom.points(p);
        predict_results.push_back({0, "xxx", 1.2, p[0].x, p[0].y, boom.width, boom.length});
    }
    // signs
    for (const auto & sign : signs) {
        sign.points(p);
        if (sign.label_id == 0)
            predict_results.push_back({8, "xxx", sign.confidence, p[0].x, p[0].y, sign.width, sign.length});
        if (sign.label_id == 2)
            predict_results.push_back({6, "xxx", sign.confidence, p[0].x, p[0].y, sign.width, sign.length});
        if (sign.label_id == 3)
            predict_results.push_back({12, "xxx", sign.confidence, p[0].x, p[0].y, sign.width, sign.length});
        if (sign.label_id == 4)
            predict_results.push_back({11, "xxx", sign.confidence, p[0].x, p[0].y, sign.width, sign.length});
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
    // 寻找mask轮廓
    cv::findContours(Conical_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 轮廓检测是否为锥桶

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

std::vector<Boom> Detector::findBooms(const cv::Mat & hsv_img)
{
    // HSV 筛选mask
    cv::Scalar Boom_LowerLimit1(params.Boom_LowerLimit_LH1, params.Boom_LowerLimit_LS1, params.Boom_LowerLimit_LV1);
    cv::Scalar Boom_UpperLimit1(params.Boom_LowerLimit_UH1, params.Boom_LowerLimit_US1, params.Boom_LowerLimit_UV1);
    cv::Mat Boom_mask1;
    cv::inRange(hsv_img, Boom_LowerLimit1, Boom_UpperLimit1, Boom_mask1);
    cv::Scalar Boom_LowerLimit2(params.Boom_LowerLimit_LH2, params.Boom_LowerLimit_LS2, params.Boom_LowerLimit_LV2);
    cv::Scalar Boom_UpperLimit2(params.Boom_LowerLimit_UH2, params.Boom_LowerLimit_US2, params.Boom_LowerLimit_UV2);
    cv::Mat Boom_mask2;
    cv::inRange(hsv_img, Boom_LowerLimit2, Boom_UpperLimit2, Boom_mask2);
    cv::bitwise_or(Boom_mask1, Boom_mask2, Boom_mask);
    // mask膨胀
    cv::Mat Boom_mask_kernel = cv::Mat::ones(params.Boom_kernel, params.Boom_kernel, CV_8U);
    cv::dilate(Boom_mask, Boom_mask, Boom_mask_kernel);

    // 寻找mask轮廓
    cv::findContours(Boom_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 轮廓检测是否为爆炸物
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Boom_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        auto boom = Boom(r_rect);
        
        if (isBoom(boom, params)){
            booms.emplace_back(boom);
        }
    }
    return booms;
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
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Sign_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        auto sign = Sign(r_rect);
        
        if (isSign(sign, params)){
            // // 使用矩形框裁剪二值化图像
            cv::Mat roi = Sign_mask(r_rect.boundingRect());

            cv::resize(roi, roi, cv::Size(28, 28));

            if (params.OUPUT_DATASET){
                // 使用系统时间来命名你的图片
                auto now = std::chrono::system_clock::now();
                auto now_ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(now);
                auto value = now_ns.time_since_epoch();
                long duration = value.count();
                std::string filename = "./icar_detector/image/" + std::to_string(duration) + ".jpg";

                std::cout << filename << " saved" << std::endl;
                cv::imwrite(filename, roi);
            }
            sign.sign_img = roi.clone();
            signs.emplace_back(sign);
        }
    }
    return signs;
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

    std::cout << "isConical: ratio: " << ratio << std::endl;

    return ratio_ok;
}

bool Detector::isBoom(const Boom & boom, const Params & params)
{
    float ratio = boom.width / boom.length;
    bool ratio_ok = params.Boom_min_ratio < ratio && ratio < params.Boom_max_ratio;

    if (ratio_ok){
        std::cout << "isBoom: ratio: " << ratio << std::endl;
    }

    return ratio_ok;
}

bool Detector::isSign(const Sign & sign, const Params & params)
{
    float ratio = sign.width / sign.length;
    bool ratio_ok = params.Sign_min_ratio < ratio && ratio < params.Sign_max_ratio;

    if (ratio_ok){
        std::cout << "isSign: ratio: " << ratio << std::endl;
    }

    return ratio_ok;
}
