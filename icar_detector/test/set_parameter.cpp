#include <cstdio>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include "icar_detector/Marks.hpp"

cv::RotatedRect conicalRect(std::vector<cv::Point> contour)
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

void showRect(cv::Mat img, cv::RotatedRect rotatedRect, cv::Scalar color)
{
    // 获取旋转矩形的四个顶点
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);

    // 在图像上绘制旋转矩形
    for (int i = 0; i < 4; i++){
        cv::line(img, vertices[i], vertices[(i+1)%4], color, 3);
    }
}

cv::Mat select_parameter(const cv::Mat & rgb_img)
{
    int l_h=0;
    int l_s=0;
    int l_v=0;
    int u_h=255;
    int u_s=255;
    int u_v=255;
    int kernel=0;
    int size=0;
    int corrosion=0;
    cv::namedWindow("Tracking", cv::WINDOW_NORMAL);
    cv::resizeWindow("Tracking", 800, 600);

    cv::createTrackbar("LH", "Tracking", &l_h, 255, nullptr);
    cv::createTrackbar("LS", "Tracking", &l_s, 255, nullptr);
    cv::createTrackbar("LV", "Tracking", &l_v, 255, nullptr);
    cv::createTrackbar("UH", "Tracking", &u_h, 255, nullptr);
    cv::createTrackbar("US", "Tracking", &u_s, 255, nullptr);
    cv::createTrackbar("UV", "Tracking", &u_v, 255, nullptr);
    cv::createTrackbar("SIZE", "Tracking", &size, 255, nullptr);
    cv::createTrackbar("kernel", "Tracking", &kernel, 100, nullptr);
    cv::createTrackbar("corrosion", "Tracking", &corrosion, 100, nullptr);

    while(1)
    {
        cv::Mat hsv;
        cv::cvtColor(rgb_img, hsv, cv::COLOR_BGR2HSV);

        // Define the lower and upper limits
        cv::Scalar lowerLimit(l_h, l_s, l_v);
        cv::Scalar upperLimit(u_h, u_s, u_v);

        cv::Mat Obstacle_mask;
        cv::inRange(hsv, lowerLimit, upperLimit, Obstacle_mask);
        // mask膨胀
        cv::Mat mask_kernel = cv::Mat::ones(kernel, kernel, CV_8U);
        cv::dilate(Obstacle_mask, Obstacle_mask, mask_kernel);

        // 创建一个结构元素
        if (corrosion > 0){
            cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(corrosion, corrosion));

            // 使用开运算（先腐蚀后膨胀）来细化mask
            cv::morphologyEx(Obstacle_mask, Obstacle_mask, cv::MORPH_OPEN, element);
        }

        // 寻找mask轮廓
        using std::vector;
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(Obstacle_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        vector<Obstacle> booms;

        for (const auto & contour : contours) {
            if (contour.size() < static_cast<long unsigned int>(size)) continue;

            auto r_rect = conicalRect(contour);
            auto boom = Obstacle(r_rect);
            booms.emplace_back(boom);
        }

        cv::imshow("frame", rgb_img);
        cv::Mat Obstacle_result;
        cv::cvtColor(Obstacle_mask, Obstacle_result, cv::COLOR_GRAY2BGR);
        cv::drawContours(Obstacle_result, contours, -1, cv::Scalar(0, 0, 255), 2);
        for (const auto & boom : booms) {
            showRect(Obstacle_result, boom, cv::Scalar(255,0,0));
        }
        cv::imshow("Obstacle_result", Obstacle_result);
        cv::waitKey(1);
    }
}

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  cv::Mat scr = cv::imread("/mnt/d/Miniconda3/envs/ML_01/Data/output4/1485.jpg");
  cv::Mat resized_img;
  cv::Size size(640, 480);
  cv::resize(scr, resized_img, size);
  select_parameter(resized_img);
  return 0;
}

