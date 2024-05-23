
// OpenCV
#include <opencv2/opencv.hpp>

// STD
#include <cstdio>
#include <ctime>
#include <chrono>
#include <memory>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "icar_detector/detector.hpp"

int main()
{
    // 加载config
    auto config = Config("./icar_detector/config/detecor_config.json");

    // 加载图像
    // cv::Mat scr = cv::imread("/mnt/d/Chromedownload/smart-car/edgeboard/res/samples/train/254.jpg");
    // cv::Mat scr = cv::imread("/mnt/d/Chromedownload/mee/mee/1.jpg");
    // /mnt/d/Chromedownload/smart-car/smart-car/edgeboard/res/samples/train
    // 指定文件夹路径
    cv::String folderPath = "/mnt/d/Chromedownload/smart-car/smart-car/edgeboard/res/samples/train/*.jpg"; 

    // 获取文件夹中所有图片的路径
    std::vector<cv::String> imagePaths;
    cv::glob(folderPath, imagePaths, false);

    // 创建窗口
    cv::namedWindow("Image Viewer", cv::WINDOW_NORMAL);

    // 初始化Detector
    auto pkg_path = ament_index_cpp::get_package_share_directory("icar_detector");
    pkg_path = pkg_path.substr(0, pkg_path.find("icar_detector")) + "icar_detector" + "/icar_detector";
    if (config.params.SHOW_LOGS) std::cout << "pkg_path: " << pkg_path << std::endl;
    auto model_path = pkg_path + "/model/mlp.onnx";
    auto label_path = pkg_path + "/model/label.txt";
    float threshold = 0.4f;
    std::vector<std::string> ignore_classes = {"negative"};
    
    Detector detector(config.params);
    detector.classifier =
    std::make_unique<SignClassifier>(model_path, label_path, threshold, ignore_classes, config.params.SHOW_LOGS);

    // 读取第一张图片
    int currentImageIndex = 0;
    double fps = 0;
    bool out = false;

    //********************************************** Loop *****************************************************
    while(true){

    auto startTime = std::chrono::high_resolution_clock::now();

    cv::Mat currentImage = cv::imread(imagePaths[currentImageIndex]);
    std::cout << "img_path: " << imagePaths[currentImageIndex] << std::endl;
    
    detector.detect(currentImage);
    cv::Mat hsv = detector.hsv_img;

    //********************************************** 锥桶识别 *****************************************************

    cv::Mat Conical_mask = detector.Conical_mask;

    //********************************************** 路牌识别 *****************************************************

    cv::Mat Sign_mask = detector.Sign_mask;

    //********************************************** 结果输出 *****************************************************
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = endTime - startTime;
    double seconds = diff.count();
    // Time elapsed
    fps = 1.0 / seconds;

    // std::cout << "fps: " << fps << std::endl;
    detector.drawResults(detector.resized_img);

    // 在图片左上角绘制帧率
    cv::putText(detector.resized_img, std::to_string(fps), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    cv::imshow("Image Viewer", detector.resized_img);
    // cv::imshow("Conical_result", Conical_result);
    // cv::imshow("Boom_result", Boom_result);
    // cv::imshow("Sign_result", Sign_result);

    // 等待用户按键
    char key = cv::waitKey(1);
    // std::cout << currentImageIndex << std::endl;
    if (key == 'a' || key == 'A') {
        // 反向浏览
        currentImageIndex = (currentImageIndex - 1 + imagePaths.size()) % imagePaths.size();
    } else if (key == 'd' || key == 'D') {
        // 正向浏览
        currentImageIndex = (currentImageIndex + 1) % imagePaths.size();
    } else if (key == 'f' || key == 'F') {
        // 输出数据集
        out = true;
    } else if (key == 27) {
        // 按下 ESC 键退出
        break;
    }
    if (out == true){
        detector.params.OUPUT_DATASET = true;
        out = false;
    }
    else{
        detector.params.OUPUT_DATASET = false;
    }
    }
    return 0;
}

