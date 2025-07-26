#pragma once

#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "types.h"

// Preprocessing functions
cv::Rect get_rect(cv::Mat& img, float bbox[4]);

// NMS functions
void nms(std::vector<Detection>& res, float* output, float conf_thresh, float nms_thresh = 0.5);

void batch_nms(std::vector<std::vector<Detection>>& batch_res, float* output, int batch_size, int output_size,
               float conf_thresh, float nms_thresh = 0.5);

// CUDA-related functions
void cuda_decode(float* predict, int num_bboxes, float confidence_threshold, float* parray, int max_objects,
                 cudaStream_t stream);

void cuda_nms(float* parray, float nms_threshold, int max_objects, cudaStream_t stream);

// Drawing functions
void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks,
                    std::unordered_map<int, std::string>& labels_map);
