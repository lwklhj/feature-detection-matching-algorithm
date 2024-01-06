#include <opencv2/opencv.hpp>
#include "data_body.h"
#include <NvInferPlugin.h>

#ifndef SuperPoint_H
#define SuperPoint_H

class SuperPoint
{
public:
    SuperPoint();
    ~SuperPoint();
    bool initial_point_model();
    size_t forward(cv::Mat &srcimg, data_point &dp);
    int get_input_h();
    int get_input_w();

private:
    template <typename T>
    using _unique_ptr = std::unique_ptr<T, infer_deleter>;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine_ptr;
    std::shared_ptr<nvinfer1::IExecutionContext> _context_ptr;
    point_lm_params _point_lm_params;

private:
    bool build_model();
    float *imnormalize(cv::Mat &img);
};

#endif