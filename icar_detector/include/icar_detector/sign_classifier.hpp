
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
    const std::string & model_path, const std::string & label_path, const double threshold,
    const std::vector<std::string> & ignore_classes = {});

  void classify(std::vector<Sign> & signs);

  double threshold;

private:
  cv::dnn::Net net_;
  std::vector<std::string> class_names_;
  std::vector<std::string> ignore_classes_;
};

#endif // !SIGN_CLASSIFIER_HPP_
