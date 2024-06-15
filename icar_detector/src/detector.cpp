
// OpenCV
#include <opencv2/opencv.hpp>

// STD
#include <cstdio>
#include <ctime>
#include <chrono>
#include <thread>

#include "icar_detector/detector.hpp"

Detector::Detector(const Params & input_params) : params(input_params){}

void Detector::initDetect_thread(void)
{
    std::thread t1(&Detector::start_findConicals, this);
    t1.detach();
    std::thread t2(&Detector::start_findSigns, this);
    t2.detach();
    std::thread t3(&Detector::start_findObstacles, this);
    t3.detach();
    std::thread t4(&Detector::start_packer, this);
    t4.detach();
}

void Detector::start_findConicals(void)
{
    while(true) {
        if (!hsv_img.empty()){
            conicals = findConicals(hsv_img);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(params.thread_sleep));
    }
}

void Detector::start_findSigns(void)
{
    while(true) {
        if (!hsv_img.empty()){
            std::vector<Sign> signs_s;
            signs_s = findSigns(hsv_img);
            if (!signs_s.empty()){
                sign_classifier->sign_classify(signs_s);
            }
            signs = signs_s;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(params.thread_sleep));
    }
}

void Detector::start_findObstacles(void)
{
    while(true) {
        if (!hsv_img.empty()){
            std::vector<Obstacle> obstacles_s;
            std::vector<Cross> cross_s;
            std::vector<Barrier> barriers_s;
            obstacles_s = findObstacles(hsv_img);
            if (!obstacles_s.empty()){
                obstacle_classifier->obstacles_classify(obstacles_s);
                Tool_man values = thinObstacles(obstacles_s);
                cross_s = values.c;
                barriers_s = values.b;
            }
            cross = cross_s;
            barriers = barriers_s;
            obstacles = obstacles_s;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(params.thread_sleep));
    }
}

void Detector::start_packer(void)
{
    while(true) {
        result = packer(conicals, signs, cross, barriers);
        std::this_thread::sleep_for(std::chrono::milliseconds(params.thread_sleep));
    }
}

std::vector<PredictResult> Detector::detect(const cv::Mat & input)
{
    hsv_img = inputImage(input);
    conicals = findConicals(hsv_img);
    signs = findSigns(hsv_img);
    obstacles = findObstacles(hsv_img);

    if (!signs.empty()){
        sign_classifier->sign_classify(signs);
    }
    if (!obstacles.empty()){
        obstacle_classifier->obstacles_classify(obstacles);
        Tool_man values = thinObstacles(obstacles);
        cross = values.c;
        barriers = values.b;
    }

    // 这堆东西到通信包的转换
    return packer(conicals, signs, cross, barriers);
}

std::vector<PredictResult> Detector::packer(std::vector<Conical> & conicals, std::vector<Sign> & signs, std::vector<Cross> & cross, std::vector<Barrier> & barriers)
{
    cv::Point2f p[4];
    std::vector<PredictResult> predict_results;
    // conicals
    std::vector<Conical> conicals1(conicals);
    for (const auto & conical : conicals1) {
        conical.points(p);
        predict_results.push_back({3, "LABEL_CONE", 1.2, (int)p[0].x, (int)p[0].y, (int)conical.width, (int)conical.length});
    }
    // signs
    std::vector<Sign> signs1(signs);
    for (const auto & sign : signs1) {
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
        if (sign.mlp.label_id == 6)
            predict_results.push_back({1, "LABEL_BRIDGE", sign.mlp.confidence, (int)p[0].x, (int)p[0].y, (int)sign.width, (int)sign.length});
    }
    std::vector<Cross> cross1(cross);
    for (const auto & cros : cross1) {
        cros.points(p);
        predict_results.push_back({4, "LABEL_CROSSWALK", 1.2, (int)p[0].x, (int)p[0].y, (int)cros.width, (int)cros.length});
    }
    std::vector<Barrier> barriers1(barriers);
    for (const auto & barrier : barriers1) {
        barrier.points(p);
        predict_results.push_back({7, "LABEL_BLOCK", 1.2, (int)p[0].x, (int)p[0].y, (int)barrier.width, (int)barrier.length});
    }

    return predict_results;

}

cv::Mat Detector::inputImage(const cv::Mat & rgb_img)
{
    cv::resize(rgb_img, resized_img, cv::Size(320, 240));
    // 转换色彩空间HSV
    cv::cvtColor(resized_img, hsv_img, cv::COLOR_BGR2HSV);

    return hsv_img;
}

std::vector<Conical> Detector::findConicals(const cv::Mat & img)
{
    // HSV 筛选mask
    cv::Scalar Conical_LowerLimit(params.Conical_LowerLimit_LH, params.Conical_LowerLimit_LS, params.Conical_LowerLimit_LV);
    cv::Scalar Conical_UpperLimit(params.Conical_LowerLimit_UH, params.Conical_LowerLimit_US, params.Conical_LowerLimit_UV);
    cv::inRange(img, Conical_LowerLimit, Conical_UpperLimit, Conical_mask);
    // 使用开运算（先腐蚀后膨胀）来细化mask 滤除噪声
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(params.Conical_filter, params.Conical_filter));
    cv::morphologyEx(Conical_mask, Conical_mask, cv::MORPH_OPEN, element);
    // 寻找mask轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(Conical_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 轮廓特征检测是否为锥桶
    std::vector<Conical> conicals;
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Conical_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        auto conical = Conical(r_rect);
        // 长宽比检测&&白色占比检测
        if (isConical(conical, params) && w_judge(r_rect, Conical_mask) > params.Conical_w_judge){
            conicals.emplace_back(conical);
        }
    }

    return conicals;
}

std::vector<Sign> Detector::findSigns(const cv::Mat & img)
{
    // HSV 筛选mask
    cv::Scalar Sign_LowerLimit1(params.Sign_LowerLimit_LH1, params.Sign_LowerLimit_LS1, params.Sign_LowerLimit_LV1);
    cv::Scalar Sign_UpperLimit1(params.Sign_LowerLimit_UH1, params.Sign_LowerLimit_US1, params.Sign_LowerLimit_UV1);
    cv::Mat Sign_mask1;
    cv::inRange(img, Sign_LowerLimit1, Sign_UpperLimit1, Sign_mask1);
    cv::Scalar Sign_LowerLimit2(params.Sign_LowerLimit_LH2, params.Sign_LowerLimit_LS2, params.Sign_LowerLimit_LV2);
    cv::Scalar Sign_UpperLimit2(params.Sign_LowerLimit_UH2, params.Sign_LowerLimit_US2, params.Sign_LowerLimit_UV2);
    cv::Mat Sign_mask2;
    cv::inRange(img, Sign_LowerLimit2, Sign_UpperLimit2, Sign_mask2);
    cv::bitwise_or(Sign_mask1, Sign_mask2, Sign_mask);
    // mask膨胀(让人脑袋和身体为一个整体)
    cv::Mat Sign_dilate_mask;
    cv::Mat Sign_mask_kernel = cv::Mat::ones(params.Sign_kernel, params.Sign_kernel, CV_8U);
    cv::dilate(Sign_mask, Sign_dilate_mask, Sign_mask_kernel);
    // 寻找mask轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(Sign_dilate_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 轮廓检测是否为路牌
    std::vector<Sign> signs;
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Sign_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        auto sign = Sign(r_rect);
        
        if (isSign(sign, params) && w_judge(r_rect, Sign_mask) > params.Sign_w_judge){
            // // 使用矩形框裁剪二值化图像
            cv::Mat roi = Sign_mask(r_rect.boundingRect());
            cv::resize(roi, roi, cv::Size(params.roi_size, params.roi_size));

            if (params.OUPUT_DATASET_Signs){
                output_dataset(roi);
            }
            sign.mlp.sign_img = roi.clone();
            signs.emplace_back(sign);
        }
    }
    return signs;
}

std::vector<Obstacle> Detector::findObstacles(const cv::Mat & img)
{
    // 拉高亮度
    std::vector<cv::Mat> hsvChannels;
    cv::split(img, hsvChannels);
    hsvChannels[2] += params.Obstacle_light;
    cv::merge(hsvChannels, img);
    // HSV 筛选mask
    cv::Scalar Obstacle_LowerLimit(params.Obstacle_LowerLimit_LH, params.Obstacle_LowerLimit_LS, params.Obstacle_LowerLimit_LV);
    cv::Scalar Obstacle_UpperLimit(params.Obstacle_LowerLimit_UH, params.Obstacle_LowerLimit_US, params.Obstacle_LowerLimit_UV);
    cv::inRange(img, Obstacle_LowerLimit, Obstacle_UpperLimit, Obstacle_mask);
    // 使用开运算（先腐蚀后膨胀）来细化mask
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(params.Obstacle_filter, params.Obstacle_filter));
    cv::morphologyEx(Obstacle_mask, Obstacle_mask, cv::MORPH_OPEN, element);
    // mask膨胀
    cv::Mat Obstacle_dilate_mask;
    cv::Mat Obstacle_mask_kernel = cv::Mat::ones(params.Obstacle_kernel, params.Obstacle_kernel, CV_8U);
    cv::dilate(Obstacle_mask, Obstacle_dilate_mask, Obstacle_mask_kernel);
    // 寻找mask轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(Obstacle_dilate_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 轮廓检测是否为锥桶
    std::vector<Obstacle> obstacles;
    for (const auto & contour : contours) {
        if (contour.size() < static_cast<long unsigned int>(params.Obstacle_SIZE)) continue;

        auto r_rect = conicalRect(contour);
        auto obstacle = Obstacle(r_rect);
        // ROI
        cv::Mat roi = Obstacle_mask(r_rect.boundingRect());
        cv::resize(roi, roi, cv::Size(params.roi_size, params.roi_size));
        
        if (params.OUPUT_DATASET_Obstacles){
            output_dataset(roi);
        }
        obstacle.mlp.sign_img = roi.clone();
        obstacles.emplace_back(obstacle);
    }
    
    // 判断斑马线 // 用什么传统，卷积神经网络才是神
    // cross = Cross_times_count(Obstacle_mask);

    return obstacles;
}

Tool_man Detector::thinObstacles(std::vector<Obstacle> obstacles)
{
    std::vector<Cross> cross;
    std::vector<Barrier> barriers;

    for (const auto & obstacle : obstacles) {
    if (obstacle.mlp.label_id == 1) continue;
    // 创建一个空白图像
    cv::Mat mask = cv::Mat::zeros(Obstacle_mask.size(), Obstacle_mask.type());
    // 创建一个用于绘制旋转矩形的点集
    cv::Point2f vertices[4];
    obstacle.points(vertices);
    // 将点集转换为vector
    std::vector<cv::Point> contour(vertices, vertices + 4);
    // 绘制旋转矩形
    cv::fillConvexPoly(mask, contour, cv::Scalar(255, 255, 255));
    // 按位与操作
    cv::Mat result;
    cv::bitwise_and(Obstacle_mask, mask, result);

    // 寻找mask轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(result, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto & contour : contours) {
        auto r_rect = cv::minAreaRect(contour);
        if (obstacle.mlp.label_id == 0){
            if (contour.size() < static_cast<long unsigned int>(params.Cross_SIZE)) continue;
            if (w_judge(r_rect, result) < params.Cross_w_judge) continue;
            auto cros = Cross(r_rect);
            if (isCross(cros, params)){cross.emplace_back(cros);}
        }
        if (obstacle.mlp.label_id == 2){
            if (contour.size() < static_cast<long unsigned int>(params.Barrier_SIZE)) continue;
            if (w_judge(r_rect, result) < params.Barrier_w_judge) continue;
            auto barrier = Barrier(r_rect);
            if (isBarrier(barrier, params)){barriers.emplace_back(barrier);}
        }
    }
    }
    // 误识别
    if (cross.size() < 3) cross.clear();

    Tool_man values = {cross, barriers};

    return values;
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

bool Detector::isBarrier(const Barrier & barrier, const Params & params){

    float ratio = barrier.width / barrier.length;
    bool ratio_ok = params.Barrier_min_ratio < ratio && ratio < params.Barrier_max_ratio;

    if (ratio_ok){
        if (params.SHOW_LOGS) std::cout << "isBarrier: ratio: " << ratio << std::endl;
    }

    return ratio_ok;
}

bool Detector::isCross(const Cross & cros, const Params & params){

    float ratio = cros.width / cros.length;
    bool ratio_ok = params.Cross_min_ratio < ratio && ratio < params.Cross_max_ratio;

    if (ratio_ok){
        if (params.SHOW_LOGS) std::cout << "isCross: ratio: " << ratio << std::endl;
    }

    return ratio_ok;
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

float Detector::w_judge(cv::RotatedRect & r_rect, cv::Mat & result)
{
    // 创建一个和二值化图像同样大小的掩码，并初始化为全黑色
    cv::Mat mask = cv::Mat::zeros(result.size(), CV_8UC1);
    // 获取最小面积矩形的四个顶点
    cv::Point2f rectPoints[4];
    r_rect.points(rectPoints);
    // 将矩形的四个顶点转换为整数坐标
    cv::Point pts[4];
    for (int i = 0; i < 4; ++i) {
        pts[i] = rectPoints[i];
    }
    // 使用 fillConvexPoly 函数来填充掩码
    cv::fillConvexPoly(mask, pts, 4, cv::Scalar(255));
    // 使用掩码来获取矩形内的像素
    cv::Mat rectImg;
    result.copyTo(rectImg, mask);

    // 计算矩形内的白色像素数量
    int whitePixelCount = cv::countNonZero(rectImg);

    // 计算矩形的总像素数量
    int totalPixelCount = r_rect.size.area();

    // 计算白色像素的占比
    float whitePixelRatio = static_cast<float>(whitePixelCount) / totalPixelCount;
    if (params.SHOW_LOGS) std::cout << "whitePixelRatio: " << whitePixelRatio << std::endl;
    return whitePixelRatio;
}

cv::Mat Detector::drawResults(cv::Mat & img)
{
    cv::Mat img_copy = img.clone();

    std::vector<Conical> conicals1(conicals);
    for (const auto & conical : conicals1) {
        showRect(img_copy, conical, cv::Scalar(0,255,255));
    }
    std::vector<Sign> signs1(signs);
    for (auto & sign : signs1){
        cv::putText(
        img_copy, sign.mlp.classfication_result, sign.bottom, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0,255,0), 2);

        showRect(img_copy, sign, cv::Scalar(0,255,0));
    }
    std::vector<Obstacle> obstacles1(obstacles);
    for (const auto & obstacle : obstacles1) {
        cv::putText(
        img_copy, obstacle.mlp.classfication_result, obstacle.bottom, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(255,255,0), 2);

        showRect(img_copy, obstacle, cv::Scalar(255,255,0));
    }
    std::vector<Cross> cross1(cross);
    for (const auto & cros : cross1) {
        showRect(img_copy, cros, cv::Scalar(255,0,255));
    }
    std::vector<Barrier> barriers1(barriers);
    for (const auto & barrier : barriers1) {
        showRect(img_copy, barrier, cv::Scalar(255,0,255));
    }

    return img_copy;
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
