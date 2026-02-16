#include "RK35llm.h"

#define HISTORY true

//----------------------------------------------------------------------------------------
RK35llm::RK35llm(void)
{
    Info    = false;
    Silence = false;
    // ImgVec removed, vector handles itself
    CurrentImgCount = 0;
    llmHandle = nullptr;
    responseReady_ = false;
    memset(&rknn_app_ctx, 0, sizeof(RKLLM_app_context_t));
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    param = rkllm_createDefaultParam();
    param.top_k = 1;
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 1;
    param.img_start   = "<|vision_start|>";
    param.img_end     = "<|vision_end|>";
    param.img_content = "<|image_pad|>";
}
//----------------------------------------------------------------------------------------
RK35llm::~RK35llm(void)
{
    if(rknn_app_ctx.input_attrs != nullptr) free(rknn_app_ctx.input_attrs);
    if(rknn_app_ctx.output_attrs != nullptr) free(rknn_app_ctx.output_attrs);
    if(rknn_app_ctx.rknn_ctx != 0) rknn_destroy(rknn_app_ctx.rknn_ctx);

    // Vector destructor handles memory automatically
    if(llmHandle!=nullptr) rkllm_destroy(llmHandle);
}
//----------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------
void RK35llm::DumpTensorAttr(rknn_tensor_attr* attr)
{
    if(!Info) return;
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
//----------------------------------------------------------------------------------------
int RK35llm::StaticCallback(RKLLMResult* result, void* userdata, LLMCallState state)
{
    if (!userdata) return -1;
    RK35llm* self = static_cast<RK35llm*>(userdata);
    return self->InstanceCallback(result, state);
}
//----------------------------------------------------------------------------------------
int RK35llm::InstanceCallback(RKLLMResult *result, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH)
    {
        if(!Silence) printf("\n");
        responseReady_ = true;
        responseCv_.notify_all();
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        if(!Silence) printf("[Error during inference]\n");
        responseBuffer_ += "[Error during inference]";
        responseReady_ = true;
        responseCv_.notify_all();
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        if(!Silence) printf("%s", result->text);
        if (result && result->text) responseBuffer_ += result->text;
    }
    return 0;
}
//----------------------------------------------------------------------------------------
// Expand the image into a square and fill it with the specified background color
cv::Mat RK35llm::Expand2Square(const cv::Mat& img, const cv::Scalar& background_color)
{
    int width = img.cols;
    int height = img.rows;

    // If the width and height are equal, return to the original image directly
    if (width == height) {
        return img.clone();
    }

    // Calculate the new size and create a new image
    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    // Calculate the image paste position
    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}
//----------------------------------------------------------------------------------------
int RK35llm::InitImgEnc(const char* model_path)
{
    int ret;
    rknn_context ctx = 0;

    ret = rknn_init(&ctx, (void*)model_path, 0, 0, NULL);
    if (ret < 0) { printf("rknn_init fail! ret=%d\n", ret); return -1; }

    // ... [Keep core mask and query logic from original code] ...
    rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2); // Simplified for brevity

    rknn_input_output_num io_num;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    // Set context info
    for (int i = 0; i < 4; i++) {
        if (output_attrs[0].dims[i] > 1) {
            rknn_app_ctx.model_image_token = output_attrs[0].dims[i];
            rknn_app_ctx.model_embed_size = output_attrs[0].dims[i + 1];
            break;
        }
    }
    rknn_app_ctx.rknn_ctx = ctx;
    rknn_app_ctx.io_num = io_num;

    // Allocate attrs
    rknn_app_ctx.input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx.output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(rknn_app_ctx.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        rknn_app_ctx.model_channel = input_attrs[0].dims[1];
        rknn_app_ctx.model_height  = input_attrs[0].dims[2];
        rknn_app_ctx.model_width   = input_attrs[0].dims[3];
    }
    else {
        rknn_app_ctx.model_height  = input_attrs[0].dims[1];
        rknn_app_ctx.model_width   = input_attrs[0].dims[2];
        rknn_app_ctx.model_channel = input_attrs[0].dims[3];
    }
    return 0;
}
//----------------------------------------------------------------------------------------
int RK35llm::ProcessOneImage(void)
{
    int ret = 0;
    rknn_input inputs[1];
    rknn_output outputs[rknn_app_ctx.io_num.n_output];

    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = rknn_app_ctx.model_width * rknn_app_ctx.model_height * rknn_app_ctx.model_channel;
    inputs[0].buf   = resized_img.data;

    ret = rknn_inputs_set(rknn_app_ctx.rknn_ctx, 1, inputs);
    if (ret < 0) return -1;

    ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
    if (ret < 0) return -1;

    for (uint32_t j=0; j<rknn_app_ctx.io_num.n_output; j++) outputs[j].want_float = 1;

    ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
    if (ret < 0) return ret;

    // CORRECTION: We must multiply by n_output because the loop below interleaves
    // multiple output tensors into the destination.
    size_t floats_per_image = rknn_app_ctx.model_image_token * rknn_app_ctx.model_embed_size * rknn_app_ctx.io_num.n_output;

    // Get current size to know where to start writing
    size_t current_vec_size = MultiImgVec.size();

    // Resize to accommodate the new image data
    MultiImgVec.resize(current_vec_size + floats_per_image);

    // Pointer to the start of *this* image's data in the vector
    float* dest_ptr = MultiImgVec.data() + current_vec_size;

    if(rknn_app_ctx.io_num.n_output == 1) {
        // Simple copy for single output models
        memcpy(dest_ptr, outputs[0].buf, outputs[0].size);
    }
    else {
        // Interleave/Concat multiple outputs (Deepstacks + Input Embeds)
        int n_out = rknn_app_ctx.io_num.n_output;
        int embed_size = rknn_app_ctx.model_embed_size;
        int n_tokens = rknn_app_ctx.model_image_token;

        for(int i = 0; i < n_tokens; i++){
            for (int j = 0; j < n_out; j++) {
                // Calculate offset:
                // i * (total_size_per_token) + j * (size_of_current_output)
                size_t offset = i * n_out * embed_size + j * embed_size;

                memcpy(dest_ptr + offset,
                       (float*)(outputs[j].buf) + i * embed_size,
                       sizeof(float) * embed_size);
            }
        }
    }

    rknn_outputs_release(rknn_app_ctx.rknn_ctx, 1, outputs);
    return ret;
}
//----------------------------------------------------------------------------------------
void RK35llm::SetInfo(bool _Info)
{
    Info = _Info;
}
//----------------------------------------------------------------------------------------
void RK35llm::SetHistory(bool _History)
{
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;
    if(_History){
        rkllm_infer_params.keep_history = 1;
        rkllm_set_chat_template(llmHandle, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");
    }
}
//----------------------------------------------------------------------------------------
void RK35llm::SetSilence(bool _Silence)
{
    Silence = _Silence;
}
//----------------------------------------------------------------------------------------
bool RK35llm::LoadModel(const std::string& VLMmodel, const std::string& LLMmodel, int32_t NewTokens, int32_t ContextLength)
{
    param.model_path = LLMmodel.c_str();
    param.max_new_tokens = NewTokens;
    param.max_context_len = ContextLength;

    int ret = rkllm_init(&llmHandle, &param, RK35llm::StaticCallback);
    if(ret != 0) return false;
    else{
        if(Info) printf("rkllm init success\n");
    }
    // IMPORTANT: only set chat template after rkllm_init succeeded and llmHandle is valid

    #if HISTORY
        rkllm_infer_params.keep_history = 1;
        // check return value (good practice)
        int setret = rkllm_set_chat_template(llmHandle,
             "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
             "<|im_start|>user\n",
             "<|im_end|>\n<|im_start|>assistant\n");
        if (setret != 0 && Info) {
            printf("rkllm_set_chat_template returned %d\n", setret);
        }
    #else
        rkllm_infer_params.keep_history = 0;
    #endif

    ret = InitImgEnc(VLMmodel.c_str());
    if(ret != 0) return false;

    return true;
}
//----------------------------------------------------------------------------------------
// Pre-process image to save memory (Resize & Pad)
// Returns a BGR image that is ready for the model's dimensions.
cv::Mat RK35llm::PreProcessImage(const cv::Mat& img)
{
    // Ensure context is initialized (LoadModel must be called first)
    if (rknn_app_ctx.model_width == 0) {
        std::cerr << "Error: Call LoadModel before PreProcessImage!" << std::endl;
        return img;
    }

    // 1. Pad to Square (Keep BGR)
    // We use the same grey value as used in LoadImage
    cv::Scalar background_color(127.5, 127.5, 127.5);
    cv::Mat square_img = Expand2Square(img, background_color);

    // 2. Resize to Model Dimensions
    size_t image_width = rknn_app_ctx.model_width;
    size_t image_height = rknn_app_ctx.model_height;
    cv::Size new_size(image_width, image_height);

    cv::Mat resized;
    cv::resize(square_img, resized, new_size, 0, 0, cv::INTER_LINEAR);

    // Return BGR (LoadImages will handle BGR->RGB conversion)
    return resized;
}
//----------------------------------------------------------------------------------------
void RK35llm::LoadImage(const cv::Mat& img)
{
    std::vector<cv::Mat> vec;
    vec.push_back(img);
    LoadImages(vec);
}
//----------------------------------------------------------------------------------------
void RK35llm::LoadImages(const std::vector<cv::Mat>& imgs)
{
    // Clear previous embeddings
    MultiImgVec.clear();
    CurrentImgCount = 0;

    cv::Scalar background_color(127.5, 127.5, 127.5);
    size_t image_width = rknn_app_ctx.model_width;
    size_t image_height = rknn_app_ctx.model_height;
    cv::Size new_size(image_width, image_height);

    int total_frames = imgs.size();

    for (int i = 0; i < total_frames; ++i) {

        // Report Progress BEFORE processing (0/10, 1/10...)
        // Or AFTER (1/10, 2/10...). Let's do AFTER for "Completed".
        if (progress_cb_) {
            // Report: "Processing frame i+1 of total"
            progress_cb_(i, total_frames);
        }

        const auto& original_img = imgs[i];

        // Note: If you used PreProcessImage externally, these might already be small,
        // but the code below is safe either way.
        cv::Mat img = original_img.clone();
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // (Ensure you use the resizing logic from your previous code here)
        cv::Mat square_img = Expand2Square(img, background_color);
        cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

        // Run inference (The slow part: 2.24s)
        int ret = ProcessOneImage();

        if (ret != 0) {
            printf("ProcessOneImage failed for frame %zu\n", CurrentImgCount);
        }
        else {
            CurrentImgCount++;
        }
    }

    // Final callback to say "Done" (100%)
    if (progress_cb_) {
        progress_cb_(total_frames, total_frames);
    }
}
//----------------------------------------------------------------------------------------
void RK35llm::SetProgressCallback(ProgressCallback cb)
{
    progress_cb_ = cb;
}
//----------------------------------------------------------------------------------------
void RK35llm::LoadVideo(const std::string& filename, float samples_per_sec, int max_samples)
{
    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video " << filename << std::endl;
        return;
    }

    // 1. Get Video Properties
    double total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (total_frames <= 0 || fps <= 0) {
        std::cerr << "Error: Invalid video properties." << std::endl;
        return;
    }

    double duration_sec = total_frames / fps;

    // 2. Calculate how many frames we WANT
    int target_count = static_cast<int>(duration_sec * samples_per_sec);

    // 3. Enforce the hard limit (max_samples)
    // If the video is long, we lower our sampling rate to fit the max budget.
    if (target_count > max_samples) target_count = max_samples;
    if (target_count < 1) target_count = 1;

    // 4. Calculate the Step Size (Stride)
    // We float-step through the video to get even distribution
    double step = total_frames / (double)target_count;

    if(Info) {
        printf("Video Info: %.2f sec, %.2f FPS. Extracting %d frames (Step: %.2f)\n",
               duration_sec, fps, target_count, step);
    }

    std::vector<cv::Mat> selected_frames;
    selected_frames.reserve(target_count);

    // 5. Extraction Loop
    // Reading frames is fast; NPU processing (later) is slow.
    // So we just grab them all here quickly.
    for (int i = 0; i < target_count; i++) {
        // Calculate index
        int frame_idx = static_cast<int>(i * step);

        // Boundary check
        if (frame_idx >= total_frames) frame_idx = total_frames - 1;

        // Seek
        cap.set(cv::CAP_PROP_POS_FRAMES, frame_idx);

        cv::Mat frame;
        if (cap.read(frame)) {
            // OPTIMIZATION: Resize immediately to save RAM!
            // We use the PreProcessImage function we made earlier.
            cv::Mat tiny = PreProcessImage(frame);
            selected_frames.push_back(tiny);
        }
        else {
            std::cerr << "Warning: Failed to read frame " << frame_idx << std::endl;
        }
    }
    cap.release();

    // 6. Pass to the NPU Engine
    // This will trigger the progress callback inside the loop
    if (!selected_frames.empty()) {
        LoadImages(selected_frames);
    }
    else {
        std::cerr << "Error: No frames extracted from video." << std::endl;
    }
}
//----------------------------------------------------------------------------------------
std::string RK35llm::Ask(const std::string& Question)
{
    std::string Str="";

    if (!llmHandle) return Str;

    // Clear previous response
    {
        std::lock_guard<std::mutex> lk(responseMutex_);
        responseBuffer_.clear();
        responseReady_ = false;
    }

    if (Question == "clear")
    {
        rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
        return Str;
    }

    // If the user did NOT include the <image> tag, we assume this is a text-only
    // follow-up question, regardless of whether we have images loaded in memory.
    if (Question.find("<image>") == std::string::npos)
    {
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = (char*)Question.c_str();
    }
    else {
        // The user included <image>, so we inject the visual data now.
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.role = "user";
        rkllm_input.multimodal_input.prompt = (char*)Question.c_str();
        rkllm_input.multimodal_input.image_embed = MultiImgVec.data();
        rkllm_input.multimodal_input.n_image_tokens = rknn_app_ctx.model_image_token;
        rkllm_input.multimodal_input.n_image = CurrentImgCount; // Number of video frames
        rkllm_input.multimodal_input.image_height = rknn_app_ctx.model_height;
        rkllm_input.multimodal_input.image_width = rknn_app_ctx.model_width;
    }

    if(!Silence) printf("Answer: ");

    int ret = rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, this);
    if (ret != 0) {
        std::cerr << "rkllm_run returned " << ret << "\n";
    }

    // Wait until callback signals completion
    std::unique_lock<std::mutex> lk(responseMutex_);
    responseCv_.wait(lk, [this]{ return responseReady_; });

    return responseBuffer_;
}
//----------------------------------------------------------------------------------------

