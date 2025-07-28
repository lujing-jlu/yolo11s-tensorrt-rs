#include "yolo_c_api.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "cuda/cuda_utils.h"
#include "yolo/logging.h"
#include "yolo/postprocess.h"
#include "yolo/preprocess.h"
#include "yolo/utils.h"
#include "yolo/config.h"
#include "yolo/types.h"

using namespace nvinfer1;

// 全局错误信息
static std::string g_last_error;

// YOLO推理器类
class YoloInference {
public:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;
    
    float* device_buffers[3];
    float* output_buffer_host = nullptr;
    float* output_seg_buffer_host = nullptr;
    
    std::unordered_map<int, std::string> labels_map;
    
    bool initialized = false;
    
    ~YoloInference() {
        cleanup();
    }
    
    void cleanup() {
        if (initialized) {
            cudaStreamDestroy(stream);
            CUDA_CHECK(cudaFree(device_buffers[0]));
            CUDA_CHECK(cudaFree(device_buffers[1]));
            CUDA_CHECK(cudaFree(device_buffers[2]));
            delete[] output_buffer_host;
            delete[] output_seg_buffer_host;
            cuda_preprocess_destroy();
            delete context;
            delete engine;
            delete runtime;
            initialized = false;
        }
    }
};

// 辅助函数声明
static bool deserialize_engine(const std::string& engine_name, YoloInference* inference);
static bool prepare_buffer(YoloInference* inference);
static cv::Rect get_downscale_rect(float bbox[4], float scale);
static std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets);

// 设置错误信息
static void set_error(const std::string& error) {
    g_last_error = error;
    std::cerr << "YOLO C API Error: " << error << std::endl;
}

YoloInferenceHandle yolo_create_inference(const char* engine_path, const char* labels_path) {
    try {
        auto inference = std::make_unique<YoloInference>();
        
        // 设置CUDA设备
        cudaSetDevice(kGpuId);
        
        // 反序列化引擎
        if (!deserialize_engine(engine_path, inference.get())) {
            set_error("Failed to deserialize engine: " + std::string(engine_path));
            return nullptr;
        }
        
        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreate(&inference->stream));
        
        // 初始化预处理
        cuda_preprocess_init(kMaxInputImageSize);
        
        // 准备缓冲区
        if (!prepare_buffer(inference.get())) {
            set_error("Failed to prepare buffers");
            return nullptr;
        }
        
        // 设置张量地址
        inference->context->setTensorAddress(kInputTensorName, inference->device_buffers[0]);
        inference->context->setTensorAddress(kOutputTensorName, inference->device_buffers[1]);
        inference->context->setTensorAddress(kProtoTensorName, inference->device_buffers[2]);
        
        // 读取标签
        if (read_labels(labels_path, inference->labels_map) != 0) {
            set_error("Failed to read labels file: " + std::string(labels_path));
            return nullptr;
        }
        
        inference->initialized = true;
        return inference.release();
        
    } catch (const std::exception& e) {
        set_error("Exception in yolo_create_inference: " + std::string(e.what()));
        return nullptr;
    }
}

void yolo_destroy_inference(YoloInferenceHandle handle) {
    if (handle) {
        delete static_cast<YoloInference*>(handle);
    }
}

bool yolo_inference(YoloInferenceHandle handle, const char* image_path, YoloResult* result) {
    return yolo_inference_fast(handle, image_path, result, false);
}

bool yolo_inference_fast(YoloInferenceHandle handle, const char* image_path, YoloResult* result, bool skip_mask_copy) {
    if (!handle || !image_path || !result) {
        set_error("Invalid parameters");
        return false;
    }

    try {
        auto* inference = static_cast<YoloInference*>(handle);

        // 读取图片时间测量
        auto image_read_start = std::chrono::high_resolution_clock::now();
        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            set_error("Failed to read image: " + std::string(image_path));
            return false;
        }
        auto image_read_end = std::chrono::high_resolution_clock::now();
        auto image_read_duration = std::chrono::duration_cast<std::chrono::microseconds>(image_read_end - image_read_start);

        bool success = yolo_inference_from_memory_fast(handle, img.data, img.cols, img.rows, img.channels(), result, skip_mask_copy);
        
        if (success) {
            // 更新图片读取时间（在yolo_inference_from_memory_fast之后）
            result->image_read_time_ms = image_read_duration.count() / 1000.0;
        }
        
        return success;

    } catch (const std::exception& e) {
        set_error("Exception in yolo_inference_fast: " + std::string(e.what()));
        return false;
    }
}

// 继续实现其他函数...
bool yolo_inference_from_memory(YoloInferenceHandle handle,
                                const uint8_t* image_data,
                                int width, int height, int channels,
                                YoloResult* result) {
    return yolo_inference_from_memory_fast(handle, image_data, width, height, channels, result, false);
}

bool yolo_inference_from_memory_fast(YoloInferenceHandle handle,
                                     const uint8_t* image_data,
                                     int width, int height, int channels,
                                     YoloResult* result, bool skip_mask_copy) {
    if (!handle || !image_data || !result) {
        set_error("Invalid parameters");
        return false;
    }
    
    try {
        auto* inference = static_cast<YoloInference*>(handle);
        
        // 创建OpenCV Mat
        cv::Mat img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, (void*)image_data);
        std::vector<cv::Mat> img_batch = {img};
        
        auto total_start_time = std::chrono::high_resolution_clock::now();
        
        // 预处理时间测量
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        cuda_batch_preprocess(img_batch, inference->device_buffers[0], kInputW, kInputH, inference->stream);
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        auto preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start);
        
        // TensorRT推理时间测量
        auto tensorrt_start = std::chrono::high_resolution_clock::now();
        inference->context->enqueueV3(inference->stream);
        auto tensorrt_end = std::chrono::high_resolution_clock::now();
        auto tensorrt_duration = std::chrono::duration_cast<std::chrono::microseconds>(tensorrt_end - tensorrt_start);
        
        // 获取输出
        void* output_buffer = const_cast<void*>(inference->context->getTensorAddress(kOutputTensorName));
        void* output_seg_buffer = const_cast<void*>(inference->context->getTensorAddress(kProtoTensorName));
        
        const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
        const int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);
        
        // 结果复制时间测量
        auto copy_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(inference->output_buffer_host, output_buffer, 
                                   kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                                   inference->stream));
        CUDA_CHECK(cudaMemcpyAsync(inference->output_seg_buffer_host, output_seg_buffer, 
                                   kBatchSize * kOutputSegSize * sizeof(float),
                                   cudaMemcpyDeviceToHost, inference->stream));
        
        CUDA_CHECK(cudaStreamSynchronize(inference->stream));
        auto copy_end = std::chrono::high_resolution_clock::now();
        auto copy_duration = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start);
        
        // 后处理时间测量
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<Detection>> res_batch;
        batch_nms(res_batch, inference->output_buffer_host, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        auto postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start);

        auto& res = res_batch[0];

        auto total_end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end_time - total_start_time);

        // 填充结果
        result->num_detections = res.size();
        result->inference_time_ms = total_duration.count() / 1000.0;
        
        // 填充详细时间
        result->image_read_time_ms = 0.0; // 从内存读取，时间为0
        result->preprocess_time_ms = preprocess_duration.count() / 1000.0;
        result->tensorrt_time_ms = tensorrt_duration.count() / 1000.0;
        result->postprocess_time_ms = postprocess_duration.count() / 1000.0;
        result->result_copy_time_ms = copy_duration.count() / 1000.0;
        


        if (result->num_detections > 0) {
            result->detections = new YoloDetection[result->num_detections];

            // 只在需要时处理掩码
            std::vector<cv::Mat> masks;
            if (!skip_mask_copy) {
                masks = process_mask(&inference->output_seg_buffer_host[0 * kOutputSegSize], kOutputSegSize, res);
            }

            for (int i = 0; i < result->num_detections; i++) {
                result->detections[i].bbox[0] = res[i].bbox[0];
                result->detections[i].bbox[1] = res[i].bbox[1];
                result->detections[i].bbox[2] = res[i].bbox[2];
                result->detections[i].bbox[3] = res[i].bbox[3];
                result->detections[i].confidence = res[i].conf;
                result->detections[i].class_id = (int)res[i].class_id;

                // 复制掩码数据（如果需要）
                if (!skip_mask_copy && i < masks.size()) {
                    cv::Mat& mask = masks[i];
                    result->detections[i].mask_width = mask.cols;
                    result->detections[i].mask_height = mask.rows;
                    int mask_size = mask.cols * mask.rows;
                    result->detections[i].mask_data = new float[mask_size];
                    memcpy(result->detections[i].mask_data, mask.data, mask_size * sizeof(float));
                } else {
                    result->detections[i].mask_data = nullptr;
                    result->detections[i].mask_width = 0;
                    result->detections[i].mask_height = 0;
                }
            }
        } else {
            result->detections = nullptr;
        }
        
        return true;

    } catch (const std::exception& e) {
        set_error("Exception in yolo_inference_from_memory: " + std::string(e.what()));
        return false;
    }
}

bool yolo_save_result_image(YoloInferenceHandle handle,
                           const char* image_path,
                           const YoloResult* result,
                           const char* output_path) {
    if (!handle || !image_path || !result || !output_path) {
        set_error("Invalid parameters");
        return false;
    }

    try {
        auto* inference = static_cast<YoloInference*>(handle);

        cv::Mat img = cv::imread(image_path);
        if (img.empty()) {
            set_error("Failed to read image: " + std::string(image_path));
            return false;
        }

        // 转换检测结果为内部格式
        std::vector<Detection> dets;
        std::vector<cv::Mat> masks;

        for (int i = 0; i < result->num_detections; i++) {
            Detection det;
            det.bbox[0] = result->detections[i].bbox[0];
            det.bbox[1] = result->detections[i].bbox[1];
            det.bbox[2] = result->detections[i].bbox[2];
            det.bbox[3] = result->detections[i].bbox[3];
            det.conf = result->detections[i].confidence;
            det.class_id = result->detections[i].class_id;
            dets.push_back(det);

            // 重建掩码
            if (result->detections[i].mask_data) {
                cv::Mat mask(result->detections[i].mask_height,
                           result->detections[i].mask_width,
                           CV_32FC1,
                           result->detections[i].mask_data);
                masks.push_back(mask.clone());
            }
        }

        // 绘制结果
        draw_mask_bbox(img, dets, masks, inference->labels_map);

        // 保存图片
        return cv::imwrite(output_path, img);

    } catch (const std::exception& e) {
        set_error("Exception in yolo_save_result_image: " + std::string(e.what()));
        return false;
    }
}

void yolo_free_result(YoloResult* result) {
    if (result && result->detections) {
        for (int i = 0; i < result->num_detections; i++) {
            if (result->detections[i].mask_data) {
                delete[] result->detections[i].mask_data;
            }
        }
        delete[] result->detections;
        result->detections = nullptr;
        result->num_detections = 0;
    }
}

const char* yolo_get_last_error(void) {
    return g_last_error.c_str();
}

// 辅助函数实现
static bool deserialize_engine(const std::string& engine_name, YoloInference* inference) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        return false;
    }

    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    file.read(serialized_engine, size);
    file.close();

    inference->runtime = createInferRuntime(gLogger);
    if (!inference->runtime) {
        delete[] serialized_engine;
        return false;
    }

    inference->engine = inference->runtime->deserializeCudaEngine(serialized_engine, size);
    if (!inference->engine) {
        delete[] serialized_engine;
        return false;
    }

    inference->context = inference->engine->createExecutionContext();
    if (!inference->context) {
        delete[] serialized_engine;
        return false;
    }

    delete[] serialized_engine;
    return true;
}

static bool prepare_buffer(YoloInference* inference) {
    const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
    const int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);

    try {
        CUDA_CHECK(cudaMalloc((void**)&inference->device_buffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&inference->device_buffers[1], kBatchSize * kOutputSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&inference->device_buffers[2], kBatchSize * kOutputSegSize * sizeof(float)));

        inference->output_buffer_host = new float[kBatchSize * kOutputSize];
        inference->output_seg_buffer_host = new float[kBatchSize * kOutputSegSize];

        return true;
    } catch (...) {
        return false;
    }
}

static cv::Rect get_downscale_rect(float bbox[4], float scale) {
    float left = bbox[0];
    float top = bbox[1];
    float right = bbox[0] + bbox[2];
    float bottom = bbox[1] + bbox[3];

    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    right = right > kInputW ? kInputW : right;
    bottom = bottom > kInputH ? kInputH : bottom;

    left /= scale;
    top /= scale;
    right /= scale;
    bottom /= scale;
    return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
}

static std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) {
    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++) {
        cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
        auto r = get_downscale_rect(dets[i].bbox, 4);

        for (int x = r.x; x < r.x + r.width; x++) {
            for (int y = r.y; y < r.y + r.height; y++) {
                float e = 0.0f;
                for (int j = 0; j < 32; j++) {
                    e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask_mat.at<float>(y, x) = e;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
        masks.push_back(mask_mat);
    }
    return masks;
}

// 新增API函数实现
bool yolo_tensorrt_inference_only(YoloInferenceHandle handle,
                                  void* input_buffer,
                                  void* output_buffer,
                                  void* output_seg_buffer,
                                  void* stream) {
    if (!handle || !input_buffer || !output_buffer || !output_seg_buffer) {
        set_error("Invalid parameters");
        return false;
    }
    
    try {
        auto* inference = static_cast<YoloInference*>(handle);
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        
        // 设置TensorRT缓冲区地址
        inference->context->setTensorAddress(kInputTensorName, input_buffer);
        inference->context->setTensorAddress(kOutputTensorName, output_buffer);
        inference->context->setTensorAddress(kProtoTensorName, output_seg_buffer);
        
        // 执行TensorRT推理
        inference->context->enqueueV3(cuda_stream);
        
        return true;
    } catch (const std::exception& e) {
        set_error("Exception in yolo_tensorrt_inference_only: " + std::string(e.what()));
        return false;
    }
}

bool yolo_get_tensorrt_info(YoloInferenceHandle handle,
                            int* input_size,
                            int* output_size,
                            int* output_seg_size) {
    if (!handle || !input_size || !output_size || !output_seg_size) {
        set_error("Invalid parameters");
        return false;
    }
    
    try {
        auto* inference = static_cast<YoloInference*>(handle);
        
        *input_size = kBatchSize * kInputH * kInputW * 3; // RGB
        *output_size = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
        *output_seg_size = 32 * (kInputH / 4) * (kInputW / 4);
        
        return true;
    } catch (const std::exception& e) {
        set_error("Exception in yolo_get_tensorrt_info: " + std::string(e.what()));
        return false;
    }
}

bool yolo_get_tensorrt_buffers(YoloInferenceHandle handle,
                               void** input_buffer,
                               void** output_buffer,
                               void** output_seg_buffer) {
    if (!handle || !input_buffer || !output_buffer || !output_seg_buffer) {
        set_error("Invalid parameters");
        return false;
    }
    
    try {
        auto* inference = static_cast<YoloInference*>(handle);
        
        *input_buffer = inference->device_buffers[0];
        *output_buffer = inference->device_buffers[1];
        *output_seg_buffer = inference->device_buffers[2];
        
        return true;
    } catch (const std::exception& e) {
        set_error("Exception in yolo_get_tensorrt_buffers: " + std::string(e.what()));
        return false;
    }
}

void* yolo_get_cuda_stream(YoloInferenceHandle handle) {
    if (!handle) {
        set_error("Invalid handle");
        return nullptr;
    }
    
    try {
        auto* inference = static_cast<YoloInference*>(handle);
        return static_cast<void*>(inference->stream);
    } catch (const std::exception& e) {
        set_error("Exception in yolo_get_cuda_stream: " + std::string(e.what()));
        return nullptr;
    }
}
