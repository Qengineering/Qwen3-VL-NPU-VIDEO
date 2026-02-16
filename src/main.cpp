#include <iostream>
#include <string>
#include <vector>
#include <algorithm> // for std::transform
#include "RK35llm.h"
//----------------------------------------------------------------------------------------
// Define a simple progress printer
void my_progress(int current, int total) {
    float percentage = (float)current / total * 100.0f;
    printf("\rProcessing Video: [%d/%d] %.1f%%", current, total, percentage);
    fflush(stdout); // Force update line
}
//----------------------------------------------------------------------------------------
// Check file extension
bool has_extension(const std::string& filename, const std::string& ext)
{
    if (filename.length() < ext.length()) return false;
    std::string f_ext = filename.substr(filename.length() - ext.length());
    // Case insensitive comparison
    std::transform(f_ext.begin(), f_ext.end(), f_ext.begin(), ::tolower);
    std::string t_ext = ext;
    std::transform(t_ext.begin(), t_ext.end(), t_ext.begin(), ::tolower);
    return f_ext == t_ext;
}
//----------------------------------------------------------------------------------------
bool is_video_file(const std::string& filename)
{
    return has_extension(filename, ".mp4") ||
           has_extension(filename, ".avi") ||
           has_extension(filename, ".mov") ||
           has_extension(filename, ".mkv");
}
//----------------------------------------------------------------------------------------
// Progress Bar Callback
void console_progress(int current, int total)
{
    float percentage = (float)current / total * 100.0f;
    int barWidth = 40;

    std::cout << "\r[";
    int pos = barWidth * percentage / 100;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(percentage) << "% (" << current << "/" << total << " frames)   " << std::flush;

    if (current == total) std::cout << std::endl;
}
//----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string input_str;
    RK35llm RKLLM;

    RKLLM.SetInfo(true);
    RKLLM.SetSilence(false);

    // Modified usage check
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " vlm_model llm_model <image_or_video> [max_frames/image2 ...]" << std::endl;
        return -1;
    }

    // Hook up the progress bar
    RKLLM.SetProgressCallback(console_progress);

    // Load models (assuming argv[1] is VLM, argv[2] is LLM)
    if (!RKLLM.LoadModel(argv[1], argv[2]) ) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    std::string input_file = argv[3];

    // 2. Input Handling Logic
    if (is_video_file(input_file)) {
        // --- VIDEO MODE ---
        int max_frames = 8;             // Default
        float sample_rate = 1.0f;       // Default: 1 frame per second

        // Check if user provided max_frames as 4th argument
        if (argc > 4) {
            try {
                max_frames = std::stoi(argv[4]);
            } catch(...) {
                std::cerr << "Invalid max_frames argument, using default: 8" << std::endl;
            }
        }

        printf("Processing Video: %s (Max %d frames)...\n", input_file.c_str(), max_frames);
        RKLLM.LoadVideo(input_file, sample_rate, max_frames);
    }
    else {
        // --- IMAGE SEQUENCE MODE ---
        std::vector<cv::Mat> frames;
        std::vector<std::string> loaded_files;

        // Loop through all remaining arguments (argv[3] -> end)
        for(int i = 3; i < argc; i++) {
            std::string img_path = argv[i];
            cv::Mat raw_img = cv::imread(img_path);

            if(raw_img.empty()) {
                std::cerr << "Warning: Could not read image: " << img_path << std::endl;
                continue;
            }

            // This prevents OOM when loading 4K images
            cv::Mat tiny = RKLLM.PreProcessImage(raw_img);
            frames.push_back(tiny);
            loaded_files.push_back(img_path);
        }

        if(frames.empty()) {
            std::cerr << "No valid images found!" << std::endl;
            return -1;
        }

        printf("Processing %zu Images...\n", frames.size());

        // This will trigger the EXACT SAME progress bar as the video mode
        RKLLM.LoadImages(frames);
    }

    // 3. Inference
    // If images were loaded, they are automatically injected into the first prompt
    while(true) {
        printf("\n");
        printf("User: ");

        std::getline(std::cin, input_str);
        if (input_str == "exit") break;

        RKLLM.Ask(input_str);
    }

    return 0;
}

//int main(int argc, char** argv)
//{
//    std::string input_str;
//    RK35llm RKLLM;
//
//    RKLLM.SetInfo(true);
//    RKLLM.SetSilence(false);
//
//    // Modified usage check
//    if (argc < 4) {
//        std::cerr << "Usage: " << argv[0] << " vlm_model llm_model image1 [image2] [image3]...\n";
//        return -1;
//    }
//
//    // Load models (assuming argv[1] is VLM, argv[2] is LLM)
//    RKLLM.LoadModel(argv[1], argv[2]);
//
//    // Hook up the callback
//    RKLLM.SetProgressCallback(my_progress);
//
//    RKLLM.LoadVideo("./test_2.mp4", 1.0f, 10);
//
////     Load multiple images from remaining arguments
////    std::vector<cv::Mat> frames;
////
////    for(int i = 3; i < argc; i++) {
////        cv::Mat raw_img = cv::imread(argv[i]);
////        if(raw_img.empty()) {
////            std::cerr << "Failed to load image: " << argv[i] << std::endl;
////            continue;
////        }
////
////         Shrink image immediately
////        cv::Mat tiny_img = RKLLM.PreProcessImage(raw_img);
////
////        frames.push_back(tiny_img);
////
////         manually release raw_img, though scope handles it
////        raw_img.release();
////    }
////
////    printf("Loading %zu images into Vision Engine...\n", frames.size());
////
////    RKLLM.LoadImages(frames);       // Qwen3 = 2.247 Sec per image
////
////    printf("Loading done...\n");
//
//    while(true) {
//        printf("\n");
//        printf("User: ");
//
//        std::getline(std::cin, input_str);
//        if (input_str == "exit") break;
//
//        // Note: For Qwen-VL with multiple images, you usually don't need
//        // explicit <image> tags if using the API properly,
//        // but if it struggles, try inputting: "Frame 1: <image>\nFrame 2: <image>\n..."
//        // If the library handles n_image correctly, a simple "Describe the sequence" works.
//        std::string output_str = RKLLM.Ask(input_str);
//    }
//
//    return 0;
//}
////----------------------------------------------------------------------------------------
//
