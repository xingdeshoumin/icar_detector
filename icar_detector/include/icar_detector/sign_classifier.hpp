
#ifndef SIGN_CLASSIFIER_HPP_
#define SIGN_CLASSIFIER_HPP_

// OpenCV
#include <opencv2/opencv.hpp>

// STL
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class SignClassifier
{
public:
  SignClassifier(
    const std::string & model_path, const std::string & label_path, const double thre,
    const std::vector<std::string> & ignore_classes, const bool out);

  void sign_classify(std::vector<Sign>& signs);
  void obstacles_classify(std::vector<Obstacle>& signs);

  double threshold;
  bool log_out;

private:
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  std::vector<std::string> ignore_classes_;
};

#endif // !SIGN_CLASSIFIER_HPP_
