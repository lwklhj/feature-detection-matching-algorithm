#include <opencv2/opencv.hpp>
#include "data_body.h"
#include <NvInferPlugin.h>

#ifndef SuperGlue_H
#define SuperGlue_H

class SuperGlue
{
public:
    SuperGlue();
    ~SuperGlue();
    bool initial_match_model();
    size_t forward(data_point &dp0, data_point &dp1, std::vector<data_match> &vdm, cv::Size imageShape);

private:
    template <typename T>
    using _unique_ptr = std::unique_ptr<T, infer_deleter>;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine_ptr;
    std::shared_ptr<nvinfer1::IExecutionContext> _context_ptr;
    match_lm_params _match_lm_params;

private:
    bool build_model();
    float *log_optimal_transport(float *scores, float bin_score,
                                 int iters, int scores_output_h, int scores_output_w);
    int get_match_keypoint(data_point &dp0, data_point &dp1,
                           float *lopt_scores, int scores_output_h, int scores_output_w, std::vector<data_match> &vdm);
    void prepare_data(data_point &dp, float *arr1, float *arr2, const cv::Size imageShape);
    // void normalize_keypoints(float* kpts, const int num_keypoints, const cv::Size imageShape);
};

#endif