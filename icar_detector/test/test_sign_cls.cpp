
// OpenCV
#include <opencv2/opencv.hpp>

// STD
#include <cstdio>
#include <ctime>
#include <chrono>
#include <memory>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "icar_detector/detector.hpp"

bool numeric_string_compare(std::string a, std::string b)
{
    return std::stoi(a.substr(a.find_last_of("/") + 1)) < std::stoi(b.substr(b.find_last_of("/") + 1));
}

int main()
{
    // 加载config
    auto config = Config("./icar_detector/config/detecor_config.json");

    // 加载图像
    // /mnt/d/Chromedownload/smart-car/edgeboard/res/samples/train/
    // /mnt/d/Chromedownload/mee/mee/
    // /mnt/d/Chromedownload/smart-car/smart-car/edgeboard/res/samples/train/
    // /mnt/d/Miniconda3/envs/ML_01/Data/output1/
    // /mnt/d/Miniconda3/envs/ML_01/Data/output4/
    // /mnt/d/Miniconda3/envs/ML_01/Data/output3/
    // /mnt/d/Chromedownload/annotations/images/
    // 指定文件夹路径
    std::string folderPath = "/mnt/d/Miniconda3/envs/ML_01/Data/output4/*.jpg"; 

    // 获取文件夹中所有图片的路径
    std::vector<std::string> imagePaths;
    cv::glob(folderPath, imagePaths, false);

    // 对imagePaths按照数字顺序排序
    std::sort(imagePaths.begin(), imagePaths.end(), numeric_string_compare);

    // 创建窗口
    cv::namedWindow("Image Viewer", cv::WINDOW_NORMAL);

    // 初始化Detector
    auto pkg_path = ament_index_cpp::get_package_share_directory("icar_detector");  
    pkg_path = pkg_path.substr(0, pkg_path.find("icar_detector")) + "icar_detector" + "/icar_detector";
    if (config.params.SHOW_LOGS) std::cout << "pkg_path: " << pkg_path << std::endl;
    auto sign_model_path = pkg_path + "/model/sign_mlp.onnx";
    auto sign_label_path = pkg_path + "/model/sign_label.txt";
    auto obstacle_model_path = pkg_path + "/model/obstacle_mlp.onnx";
    auto obstacle_label_path = pkg_path + "/model/obstacle_label.txt";
    float threshold = 0.7f;
    std::vector<std::string> ignore_classes = {"negative"};
    
    Detector detector(config.params);
    detector.sign_classifier =
    std::make_unique<SignClassifier>(sign_model_path, sign_label_path, threshold, ignore_classes, config.params.SHOW_LOGS);
    detector.obstacle_classifier =
    std::make_unique<SignClassifier>(obstacle_model_path, obstacle_label_path, (threshold), ignore_classes, config.params.SHOW_LOGS);
    if (config.params.THREADS) detector.initDetect_thread();
    
    // 读取第一张图片
    int currentImageIndex = 0;
    double fps = 0;

    //********************************************** Loop *****************************************************
    while(true){

    
    auto startTime = std::chrono::high_resolution_clock::now();
    cv::Mat currentImage = cv::imread(imagePaths[currentImageIndex]);
    std::cout << "img_path: " << imagePaths[currentImageIndex] << std::endl;
    
    if (!config.params.THREADS) detector.detect(currentImage);
    if (config.params.THREADS) detector.inputImage(currentImage);
    //********************************************** 锥桶识别 *****************************************************
    //********************************************** 路牌识别 *****************************************************
    //********************************************** 结果输出 *****************************************************
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = endTime - startTime;
    double seconds = diff.count();
    // Time elapsed
    fps = (1.0 / seconds)*0.1 + fps*0.9;

    // std::cout << "fps: " << fps << std::endl;
    cv::Mat show_img = detector.drawResults(detector.resized_img);
    cv::resize(show_img, show_img, cv::Size(1280, 960));
    // 在图片左上角绘制帧率
    cv::putText(show_img, std::to_string(fps), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

    cv::imshow("Image Viewer", show_img);
    // if (!detector.Conical_mask.empty()) cv::imshow("Conical_mask", detector.Conical_mask.clone());
    // if (!detector.Sign_mask.empty()) cv::imshow("Sign_mask", detector.Sign_mask.clone());
    if (!detector.Obstacle_mask.empty()) cv::imshow("Obstacle_mask", detector.Obstacle_mask.clone());

    // 等待用户按键
    char key = cv::waitKey(1);
    // std::cout << currentImageIndex << std::endl;
    detector.params.OUPUT_DATASET_Signs = false;
    detector.params.OUPUT_DATASET_Obstacles = false;
    if (key == 'a' || key == 'A') {
        // 反向浏览
        currentImageIndex = (currentImageIndex - 1 + imagePaths.size()) % imagePaths.size();
    } else if (key == 'd' || key == 'D') {
        // 正向浏览
        currentImageIndex = (currentImageIndex + 1) % imagePaths.size();
    } else if (key == 'f' || key == 'F') {
        // 输出数据集
        detector.params.OUPUT_DATASET_Signs = true;
    } else if (key == 'g' || key == 'G') {
        // 输出数据集
        detector.params.OUPUT_DATASET_Obstacles = true;
    } else if (key == 27) {
        // 按下 ESC 键退出
        break;
    }


    }
    return 0;
}

