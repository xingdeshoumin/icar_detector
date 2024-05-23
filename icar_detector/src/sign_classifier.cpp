
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// STL
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstdio>

#include "icar_detector/Marks.hpp"
#include "icar_detector/sign_classifier.hpp"

SignClassifier::SignClassifier(
    const std::string & model_path, const std::string & label_path, const double thre,
    const std::vector<std::string> & ignore_classes, const bool out)
:   threshold(thre), ignore_classes_(ignore_classes), log_out(out)
{
    net_ = cv::dnn::readNetFromONNX(model_path);

    std::ifstream label_file(label_path);
    std::string line;
    while (std::getline(label_file, line)) {
        class_names_.push_back(line);
    }
    if (log_out){
        std::cout << "class_names_: ";
        for (auto & class_name : class_names_){
            std::cout << class_name << ", ";
        }
        std::cout << std::endl;
    }
}

void SignClassifier::classify(std::vector<Sign> & signs)
{
    for (auto & sign : signs){
        cv::Mat image = sign.sign_img.clone();

        // Normalize
        image = image / 255.0;

        // Create blob from image
        cv::Mat blob;
        cv::dnn::blobFromImage(image, blob);

        // Set the input blob for the neural network
        net_.setInput(blob);
        // Forward pass the image blob through the model
        cv::Mat outputs = net_.forward();

        // Do softmax
        float max_prob = *std::max_element(outputs.begin<float>(), outputs.end<float>());
        cv::Mat softmax_prob;
        cv::exp(outputs - max_prob, softmax_prob);
        float sum = static_cast<float>(cv::sum(softmax_prob)[0]);
        softmax_prob /= sum;

        double confidence;
        cv::Point class_id_point;
        minMaxLoc(softmax_prob.reshape(1, 1), nullptr, &confidence, nullptr, &class_id_point);
        int label_id = class_id_point.x;
        sign.label_id = label_id;
        sign.confidence = confidence;
        sign.number = class_names_[label_id];
        if (log_out) std::cout << "sign.number: " << sign.number << std::endl;
        std::stringstream result_ss;
        result_ss << sign.number << ": " << std::fixed << std::setprecision(1)
                << sign.confidence * 100.0 << "%";
        sign.classfication_result = result_ss.str();
        if (log_out) std::cout << "confidence: " << sign.confidence << std::endl;
    }
    
    signs.erase(
    std::remove_if(
      signs.begin(), signs.end(),
      [this](const Sign & sign) {
        if (sign.confidence < threshold) {
          return true;
        }

        for (const auto & ignore_class : ignore_classes_) {
          if (sign.number == ignore_class) {
            return true;
          }
        }
        return false;
      }),
    signs.end());
}
