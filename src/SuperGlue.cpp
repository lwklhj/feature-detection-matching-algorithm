#include <NvInfer.h>
#include "cuda_runtime_api.h"
#include <math.h>

#include "logging.h"
#include <memory>
#include <fstream>
#include "SuperGlue.h"

static Logger gLogger;

bool SuperGlue::build_model()
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
    {
        return false;
    }

    char *model_deser_buffer{nullptr};
    const std::string engine_file_path(_match_lm_params.match_weight_file);
    std::ifstream ifs;
    int ser_length;
    ifs.open(engine_file_path.c_str(), std::ios::in | std::ios::binary);
    if (ifs.is_open())
    {
        ifs.seekg(0, std::ios::end);
        ser_length = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        model_deser_buffer = new char[ser_length];
        ifs.read(model_deser_buffer, ser_length);
        ifs.close();
    }
    else
    {
        return false;
    }

    _engine_ptr = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_deser_buffer, ser_length, nullptr), infer_deleter());
    if (!_engine_ptr)
    {
        std::cout << "load engine failed!" << std::endl;
        return false;
    }
    delete[] model_deser_buffer;

    _context_ptr = std::shared_ptr<nvinfer1::IExecutionContext>(_engine_ptr->createExecutionContext(), infer_deleter());
    if (!_context_ptr)
    {
        return false;
    }
    return true;
}

bool SuperGlue::initial_match_model()
{
    bool ret = build_model();
    if (!ret)
    {
        std::cout << "build model is failed!" << std::endl;
    }
    return ret;
}

size_t SuperGlue::forward(data_point &dp0, data_point &dp1, std::vector<data_match> &vdm, cv::Size imageShape)
{
    float *keypoint0 = new float[dp0.keypoint_size * 2];
    float *scores0 = new float[dp0.keypoint_size];
    float *descriptors0 = dp0.descriptors;
    float *keypoint1 = new float[dp1.keypoint_size * 2];
    float *scores1 = new float[dp1.keypoint_size];
    float *descriptors1 = dp1.descriptors;
    prepare_data(dp0, keypoint0, scores0, imageShape);
    prepare_data(dp1, keypoint1, scores1, imageShape);
    // normalize_keypoints(keypoint0, dp0.keypoint_size * 2, imageShape);
    // normalize_keypoints(keypoint1, dp1.keypoint_size * 2, imageShape);
    int keypoints0_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[0].c_str());
    assert(_engine_ptr->getBindingDataType(keypoints0_inputIndex) == nvinfer1::DataType::kFLOAT);
    int keypoints1_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[1].c_str());
    assert(_engine_ptr->getBindingDataType(keypoints1_inputIndex) == nvinfer1::DataType::kFLOAT);
    int descriptors0_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[2].c_str());
    assert(_engine_ptr->getBindingDataType(descriptors0_inputIndex) == nvinfer1::DataType::kFLOAT);
    int descriptors1_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[3].c_str());
    assert(_engine_ptr->getBindingDataType(descriptors1_inputIndex) == nvinfer1::DataType::kFLOAT);
    int scores0_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[4].c_str());
    assert(_engine_ptr->getBindingDataType(scores0_inputIndex) == nvinfer1::DataType::kFLOAT);
    int scores1_inputIndex = _engine_ptr->getBindingIndex(_match_lm_params.input_names[5].c_str());
    assert(_engine_ptr->getBindingDataType(scores1_inputIndex) == nvinfer1::DataType::kFLOAT);
    int scores_outputIndex = _engine_ptr->getBindingIndex(_match_lm_params.output_names[0].c_str());
    assert(_engine_ptr->getBindingDataType(scores_outputIndex) == nvinfer1::DataType::kFLOAT);

    _context_ptr->setInputShape(_match_lm_params.input_names[0].c_str(), nvinfer1::Dims3(1, dp0.keypoint_size, 2));
    _context_ptr->setInputShape(_match_lm_params.input_names[1].c_str(), nvinfer1::Dims3(1, dp1.keypoint_size, 2));
    _context_ptr->setInputShape(_match_lm_params.input_names[2].c_str(), nvinfer1::Dims3(1, 256, dp0.keypoint_size));
    _context_ptr->setInputShape(_match_lm_params.input_names[3].c_str(), nvinfer1::Dims3(1, 256, dp1.keypoint_size));
    _context_ptr->setInputShape(_match_lm_params.input_names[4].c_str(), nvinfer1::Dims2(1, dp0.keypoint_size));
    _context_ptr->setInputShape(_match_lm_params.input_names[5].c_str(), nvinfer1::Dims2(1, dp1.keypoint_size));

    void *buffers[7];
    cudaMalloc(&buffers[keypoints0_inputIndex], dp0.keypoint_size * 2 * sizeof(float));
    cudaMalloc(&buffers[keypoints1_inputIndex], dp1.keypoint_size * 2 * sizeof(float));
    cudaMalloc(&buffers[descriptors0_inputIndex], dp0.keypoint_size * 256 * sizeof(float));
    cudaMalloc(&buffers[descriptors1_inputIndex], dp1.keypoint_size * 256 * sizeof(float));
    cudaMalloc(&buffers[scores0_inputIndex], dp0.keypoint_size * sizeof(float));
    cudaMalloc(&buffers[scores1_inputIndex], dp1.keypoint_size * sizeof(float));
    cudaMalloc(&buffers[scores_outputIndex], dp0.keypoint_size * dp1.keypoint_size * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(buffers[keypoints0_inputIndex], keypoint0, dp0.keypoint_size * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[keypoints1_inputIndex], keypoint1, dp1.keypoint_size * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[descriptors0_inputIndex], descriptors0, dp0.keypoint_size * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[descriptors1_inputIndex], descriptors1, dp1.keypoint_size * 256 * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[scores0_inputIndex], scores0, dp0.keypoint_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[scores1_inputIndex], scores1, dp1.keypoint_size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // For enqueueV3
    _context_ptr->setInputTensorAddress(_match_lm_params.input_names[0].c_str(), buffers[keypoints0_inputIndex]);
    _context_ptr->setInputTensorAddress(_match_lm_params.input_names[1].c_str(), buffers[keypoints1_inputIndex]);
    _context_ptr->setInputTensorAddress(_match_lm_params.input_names[2].c_str(), buffers[descriptors0_inputIndex]);
    _context_ptr->setInputTensorAddress(_match_lm_params.input_names[3].c_str(), buffers[descriptors1_inputIndex]);
    _context_ptr->setInputTensorAddress(_match_lm_params.input_names[4].c_str(), buffers[scores0_inputIndex]);
    _context_ptr->setInputTensorAddress(_match_lm_params.input_names[5].c_str(), buffers[scores1_inputIndex]);
    _context_ptr->setTensorAddress(_match_lm_params.output_names[0].c_str(), buffers[scores_outputIndex]);

    bool status = _context_ptr->enqueueV3(stream);
    if (!status)
    {
        std::cout << "execute ifer error! " << std::endl;
        return 101;
    }
    float *scores_output = new float[dp0.keypoint_size * dp1.keypoint_size];
    cudaMemcpyAsync(scores_output, buffers[scores_outputIndex], dp0.keypoint_size * dp1.keypoint_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamDestroy(stream);
    cudaFree(buffers[keypoints0_inputIndex]);
    cudaFree(buffers[keypoints1_inputIndex]);
    cudaFree(buffers[descriptors0_inputIndex]);
    cudaFree(buffers[descriptors1_inputIndex]);

    cudaFree(buffers[scores0_inputIndex]);
    cudaFree(buffers[scores1_inputIndex]);
    cudaFree(buffers[scores_outputIndex]);

    int scores_output_h = dp0.keypoint_size;
    int scores_output_w = dp1.keypoint_size;
    float bin_score = 4.4124f;
    int iters = 20;
    float *lopt_scores{nullptr};
    lopt_scores = log_optimal_transport(scores_output, bin_score, iters, scores_output_h, scores_output_w);
    delete[] scores_output;
    scores_output = nullptr;
    int ret_gmk = get_match_keypoint(dp0, dp1, lopt_scores, scores_output_h, scores_output_w, vdm);
    delete[] keypoint0;
    delete[] scores0;
    delete[] descriptors0;
    delete[] keypoint1;
    delete[] scores1;
    delete[] descriptors1;
    delete[] lopt_scores;
    keypoint0 = nullptr;
    scores0 = nullptr;
    descriptors0 = nullptr;
    keypoint1 = nullptr;
    scores1 = nullptr;
    descriptors1 = nullptr;
    lopt_scores = nullptr;
    return ret_gmk;
}

int SuperGlue::get_match_keypoint(data_point &dp0, data_point &dp1,
                                  float *lopt_scores, int scores_output_h, int scores_output_w, std::vector<data_match> &vdm)
{
    std::vector<maxv_indices> vmi0, vmi1;
    for (int i = 0; i < scores_output_h + 1; i++)
    {
        float temp_max_value0 = lopt_scores[0];
        maxv_indices mi0;
        for (int j = 0; j < scores_output_w + 1; j++)
        {
            if ((j + 1) % (scores_output_w + 1) != 0 && i != scores_output_h)
            {
                float current_value = lopt_scores[i * (scores_output_w + 1) + j];
                if (temp_max_value0 < current_value)
                {
                    temp_max_value0 = current_value;
                    mi0.indices = j;
                }
            }
        }
        if (i != scores_output_h)
        {
            mi0.max_value = temp_max_value0;
            vmi0.emplace_back(mi0);
        }
    }

    for (int i = 0; i < scores_output_w + 1; i++)
    {
        float temp_max_value1 = lopt_scores[0];
        maxv_indices mi1;
        for (int j = 0; j < scores_output_h + 1; j++)
        {
            if ((j + 1) % (scores_output_h + 1) != 0 && j != scores_output_h)
            {
                float current_value = lopt_scores[j * (scores_output_w + 1) + i];
                if (temp_max_value1 < current_value)
                {
                    temp_max_value1 = current_value;
                    mi1.indices = j;
                }
            }
        }
        if (i != scores_output_w)
        {
            mi1.max_value = temp_max_value1;
            vmi1.emplace_back(mi1);
        }
    }
    int vmi1_size = vmi1.size();
    if (vmi0.size() >= 10 && vmi1_size >= 10)
    {
        for (int i = 0; i < vmi0.size(); i++)
        {
            int vmi0_index = vmi0[i].indices;
            if (vmi1[vmi0_index].indices == i)
            {
                float temp_mscores0 = std::exp(vmi1[vmi0[i].indices].max_value);
                if (temp_mscores0 > _match_lm_params.match_threshold)
                {
                    data_match dm0;
                    dm0.msscores0 = temp_mscores0;
                    dm0.valid_keypoint_x0 = dp0.vdt[i].x;
                    dm0.valid_keypoint_y0 = dp0.vdt[i].y;

                    if (vmi0[i].indices > dp1.keypoint_size)
                    {
                        std::cout << "get valid keypoint need carefull" << std::endl;
                    }
                    dm0.valid_keypoint_x1 = dp1.vdt[vmi0[i].indices].x;
                    dm0.valid_keypoint_y1 = dp1.vdt[vmi0[i].indices].y;
                    vdm.emplace_back(dm0);
                }
            }
        }
        return vdm.size();
    }
    else
    {
        return 0;
    }
}

float *SuperGlue::log_optimal_transport(float *scores, float bin_score, int iters,
                                        int scores_output_h, int scores_output_w)
{
    float norm = -std::log(scores_output_h + scores_output_w);
    int socres_new_size = scores_output_h * scores_output_w + scores_output_h + scores_output_w + 1;
    float *scores_new = new float[socres_new_size];

    for (int i = 0; i < scores_output_h + 1; i++)
    {
        for (int j = 0; j < scores_output_w + 1; j++)
        {
            if ((j + 1) % (scores_output_w + 1) == 0 || i == scores_output_h)
            {
                scores_new[i * (scores_output_w + 1) + j] = bin_score;
            }
            else
            {
                scores_new[i * (scores_output_w + 1) + j] = scores[i * scores_output_w + j];
            }
        }
    }

    float *log_mu = new float[scores_output_h + 1];
    float *log_nu = new float[scores_output_w + 1];
    for (int i = 0; i < scores_output_h + 1; i++)
    {
        if (i == scores_output_h)
        {
            log_mu[i] = std::log(scores_output_w) + norm;
        }
        else
        {
            log_mu[i] = norm;
        }
    }

    for (int i = 0; i < scores_output_w + 1; i++)
    {
        if (i == scores_output_w)
        {
            log_nu[i] = std::log(scores_output_h) + norm;
        }
        else
        {
            log_nu[i] = norm;
        }
    }

    float *v = new float[scores_output_w + 1];
    float *u = new float[scores_output_h + 1];
    memset(v, 0.f, (scores_output_w + 1) * sizeof(float));
    memset(u, 0.f, (scores_output_h + 1) * sizeof(float));

    for (int iter = 0; iter < iters; iter++)
    {
        for (int i = 0; i < scores_output_h + 1; i++)
        {
            float zv_sum_exp = 0.f;
            for (int j = 0; j < scores_output_w + 1; j++)
            {
                zv_sum_exp += std::exp(scores_new[i * (scores_output_w + 1) + j] + v[j]);
            }
            u[i] = log_mu[i] - std::log(zv_sum_exp);
        }
        for (int i = 0; i < scores_output_w + 1; i++)
        {
            float zu_sum_exp = 0.f;
            for (int j = 0; j < scores_output_h + 1; j++)
            {
                zu_sum_exp += std::exp(scores_new[j * (scores_output_w + 1) + i] + u[j]);
            }
            v[i] = log_nu[i] - std::log(zu_sum_exp);
        }
    }

    for (int i = 0; i < scores_output_h + 1; i++)
    {
        for (int j = 0; j < scores_output_w + 1; j++)
        {
            scores_new[i * (scores_output_w + 1) + j] = scores_new[i * (scores_output_w + 1) + j] + u[i] + v[j] - norm;
        }
    }
    delete[] log_mu;
    delete[] log_nu;
    delete[] u;
    delete[] v;
    log_mu = nullptr;
    log_nu = nullptr;
    u = nullptr;
    v = nullptr;

    return scores_new;
}

void SuperGlue::prepare_data(data_point &dp, float *arr1, float *arr2, const cv::Size imageShape)
{
    const float center_x = imageShape.width / 2.0f;
    const float center_y = imageShape.height / 2.0f;
    const float scaling_factor = std::max(imageShape.width, imageShape.height) * 0.7f;
    int count_k = 0;
    int count_s = 0;
    for (auto &dpv : dp.vdt)
    {
        arr1[count_k] = (dpv.x - center_x) / scaling_factor;
        arr1[count_k + 1] = (dpv.y - center_y) / scaling_factor;
        count_k += 2;

        arr2[count_s] = dpv.s;
        count_s++;
    }
}

SuperGlue::SuperGlue()
{
    cudaSetDevice(0);
    _match_lm_params.input_names.push_back("keypoints0");
    _match_lm_params.input_names.push_back("keypoints1");
    _match_lm_params.input_names.push_back("descriptors0");
    _match_lm_params.input_names.push_back("descriptors1");
    _match_lm_params.input_names.push_back("scores0");
    _match_lm_params.input_names.push_back("scores1");
    _match_lm_params.output_names.push_back("scores");
    _match_lm_params.dla_core = -1;
    _match_lm_params.int8 = false;
    _match_lm_params.fp16 = false;
    _match_lm_params.batch_size = 1;
    _match_lm_params.seria = false;
    _match_lm_params.match_weight_file = "../engines/superglue_indoor.engine";
    _match_lm_params.match_threshold = 0.2f;
}

SuperGlue::~SuperGlue()
{
}