#ifndef RK35LLM_H
#define RK35LLM_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <mutex>
#include <vector> // Added
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"
#include "rkllm.h"
//----------------------------------------------------------------------------------------
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    int model_image_token;
    int model_embed_size;
} RKLLM_app_context_t;
//----------------------------------------------------------------------------------------
class RK35llm
{
public:
    using ProgressCallback = std::function<void(int, int)>;     // Define a callback type: void(int current_frame, int total_frames)
private:
    bool                Info;
    bool                Silence;

    std::vector<float>  MultiImgVec;        // vector for dynamic multi-image support
    size_t              CurrentImgCount;    // Track how many images are loaded

    LLMHandle           llmHandle;
    RKLLMParam          param;
    RKLLM_app_context_t rknn_app_ctx;
    RKLLMInput          rkllm_input;
    RKLLMInferParam     rkllm_infer_params;

    std::string         responseBuffer_;
    std::mutex          responseMutex_;
    std::condition_variable responseCv_;
    bool                responseReady_ = false;
    ProgressCallback progress_cb_ = nullptr;

private:
    void        DumpTensorAttr(rknn_tensor_attr* attr);
    int         InitImgEnc(const char* model_path);
    int         ProcessOneImage(void);      // Helper to process one image and append to vector

    static int  StaticCallback(RKLLMResult* result, void* userdata, LLMCallState state);
    int         InstanceCallback(RKLLMResult* result, LLMCallState state);
    cv::Mat     Expand2Square(const cv::Mat& img, const cv::Scalar& background_color = cv::Scalar(127,127,127));

protected:
    cv::Mat resized_img;

public:
    RK35llm();
    virtual ~RK35llm();

    void SetInfo(bool Info);
    void SetHistory(bool History);
    void SetSilence(bool Silence);

    bool LoadModel(const std::string& VLMmodel, const std::string& LLMmodel, int32_t NewTokens=2048, int32_t ContextLength=4096);

    cv::Mat PreProcessImage(const cv::Mat& img);                // Helper to resize/pad images before storing them

    void SetProgressCallback(ProgressCallback cb);              // Register a callback function

    // samples_per_sec: Desired FPS (can be < 1.0, e.g., 0.5 for one frame every 2s)
    // max_samples: Hard limit on total frames to prevent OOM or long waits
    void LoadVideo(const std::string& filename, float samples_per_sec, int max_samples);
    void LoadImage(const cv::Mat& img);                         // Supports single image
    void LoadImages(const std::vector<cv::Mat>& imgs);          // Supports video sequence

    std::string Ask(const std::string& Question);
};
//----------------------------------------------------------------------------------------
#endif // RK35LLM_H
