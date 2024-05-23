
#ifndef MARKS_HPP_
#define MARKS_HPP_

// OpenCV
#include <opencv2/core.hpp>

// STD
#include <iostream>
#include <fstream>

// Nlohmann
#include <nlohmann/json.hpp>

struct Params
{
    bool OUPUT_DATASET = true;          // 是否输出数据集
    bool SHOW_LOGS = true;              // 是否输出log
    int Conical_LowerLimit_LH = 27;     // 锥桶HSV参数
    int Conical_LowerLimit_LS = 113;
    int Conical_LowerLimit_LV = 115;
    int Conical_LowerLimit_UH = 37;
    int Conical_LowerLimit_US = 255;
    int Conical_LowerLimit_UV = 200;
    int Conical_kernel = 7;
    int Conical_SIZE = 10;              // 锥桶轮廓筛选
    float Conical_max_ratio = 1.0f;     // 锥桶长宽比最大值
    float Conical_min_ratio = 0.5f;     // 锥桶长宽比最小值
    int Sign_LowerLimit_LH1 = 0;        // 标志HSV参数
    int Sign_LowerLimit_LS1 = 120;
    int Sign_LowerLimit_LV1 = 0;
    int Sign_LowerLimit_UH1 = 22;
    int Sign_LowerLimit_US1 = 255;
    int Sign_LowerLimit_UV1 = 121;
    int Sign_LowerLimit_LH2 = 160;
    int Sign_LowerLimit_LS2 = 120;
    int Sign_LowerLimit_LV2 = 0;
    int Sign_LowerLimit_UH2 = 183;
    int Sign_LowerLimit_US2 = 255;
    int Sign_LowerLimit_UV2 = 121;
    int Sign_kernel = 7;
    int Sign_SIZE = 20;
    float Sign_max_ratio = 1.2f;        // 标志长宽比最大值
    float Sign_min_ratio = 0.6f;        // 标志长宽比最小值
    int Obstacle_LowerLimit_LH = 0;
    int Obstacle_LowerLimit_LS = 0;
    int Obstacle_LowerLimit_LV = 0;
    int Obstacle_LowerLimit_UH = 0;
    int Obstacle_LowerLimit_US = 0;
    int Obstacle_LowerLimit_UV = 0;
    int Obstacle_kernel = 0.0;
    int Obstacle_SIZE = 40;
    float Obstacle_max_ratio;
    float Obstacle_min_ratio;
    float Obstacle_w_judge;
    int Cross_times;
    int roi_size = 28;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
    Params,
    OUPUT_DATASET,
    SHOW_LOGS,
    Conical_LowerLimit_LH,
    Conical_LowerLimit_LS,
    Conical_LowerLimit_LV,
    Conical_LowerLimit_UH,
    Conical_LowerLimit_US,
    Conical_LowerLimit_UV,
    Conical_kernel,
    Conical_SIZE,
    Conical_max_ratio,
    Conical_min_ratio,
    Sign_LowerLimit_LH1,
    Sign_LowerLimit_LS1,
    Sign_LowerLimit_LV1,
    Sign_LowerLimit_UH1,
    Sign_LowerLimit_US1,
    Sign_LowerLimit_UV1,
    Sign_LowerLimit_LH2,
    Sign_LowerLimit_LS2,
    Sign_LowerLimit_LV2,
    Sign_LowerLimit_UH2,
    Sign_LowerLimit_US2,
    Sign_LowerLimit_UV2,
    Sign_kernel,
    Sign_SIZE,
    Sign_max_ratio,
    Sign_min_ratio,
    Obstacle_LowerLimit_LH,
    Obstacle_LowerLimit_LS,
    Obstacle_LowerLimit_LV,
    Obstacle_LowerLimit_UH,
    Obstacle_LowerLimit_US,
    Obstacle_LowerLimit_UV,
    Obstacle_kernel,
    Obstacle_SIZE,
    Obstacle_max_ratio,
    Obstacle_min_ratio,
    Obstacle_w_judge,
    Cross_times,
    roi_size
    ) // 构造函数
};

struct Config : public Params
{
    Config() = default;
    explicit Config(const std::string & jsonPath)
    {
        std::ifstream config_if(jsonPath);
        // 在这里处理打开的 JSON 文件，例如读取配置信息
        if (!config_if.good())
        {
            std::cerr << "Error: Params file path:[" << jsonPath
                        << "] not find .\n";
            exit(-1);
        }
        nlohmann::json config_json;
        config_if >> config_json;
        try
        {
            params = config_json.get<Params>();
        }
        catch (const nlohmann::detail::exception &e)
        {
            std::cerr << "Json Params Parse failed :" << e.what() << '\n';
            exit(-1);
        }
    }

    Params params;
};

struct Obstacle : public cv::RotatedRect
{
    Obstacle() = default;
    explicit Obstacle(cv::RotatedRect box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color;
    cv::Point2f top, bottom;
    double length;
    double width;
    float tilt_angle;
};

struct Conical : public cv::RotatedRect
{
    Conical() = default;
    explicit Conical(cv::RotatedRect box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color;
    cv::Point2f top, bottom;
    double length;
    double width;
    float tilt_angle;
};

struct  Cross : public cv::RotatedRect
{
    Cross() = default;
    explicit Cross(cv::RotatedRect box) : cv::RotatedRect(box)
    {
        cv::Point2f p[4];
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color;
    cv::Point2f top, bottom;
    double length;
    double width;
    float tilt_angle;
};

struct Sign : public cv::RotatedRect
{
    Sign() = default;
    explicit Sign(cv::RotatedRect box) : cv::RotatedRect(box)
    {
        box.points(p);
        std::sort(p, p + 4, [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
        tilt_angle = tilt_angle / CV_PI * 180;
    }

    int color;
    cv::Point2f p[4];
    cv::Point2f top, bottom;
    double length;
    double width;
    float tilt_angle;
    // for MLP!!!!!!!!!!!
    cv::Mat sign_img;
    int label_id;
    float confidence;
    std::string number;
    std::string classfication_result;
};

#endif // !MARKS_HPP_

