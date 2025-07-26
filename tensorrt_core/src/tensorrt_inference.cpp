#include "tensorrt_inference.h"
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include <NvInfer.h>

using namespace nvinfer1;

// 全局错误信息
static std::string g_last_error;

// 简单的Logger
class SimpleLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

static SimpleLogger gLogger;

// TensorRT推理器类
class TensorRTInference {
public:
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    cudaStream_t stream;
    
    // 输入输出缓冲区
    void* input_buffer_device = nullptr;
    void* output_buffer_device = nullptr;
    void* output_seg_buffer_device = nullptr;
    
    // 尺寸信息
    int input_width = 640;
    int input_height = 640;
    int max_batch_size = 1;
    int output_size = 0;
    int output_seg_size = 0;
    
    bool initialized = false;
    
    ~TensorRTInference() {
        cleanup();
    }
    
    void cleanup() {
        if (initialized) {
            if (input_buffer_device) cudaFree(input_buffer_device);
            if (output_buffer_device) cudaFree(output_buffer_device);
            if (output_seg_buffer_device) cudaFree(output_seg_buffer_device);
            if (stream) cudaStreamDestroy(stream);
            delete context;
            delete engine;
            delete runtime;
            initialized = false;
        }
    }
};

// 设置错误信息
static void set_error(const std::string& error) {
    g_last_error = error;
    std::cerr << "TensorRT Error: " << error << std::endl;
}

TensorRTHandle tensorrt_create(const char* engine_path, 
                              int input_width, 
                              int input_height, 
                              int max_batch_size) {
    try {
        auto inference = std::make_unique<TensorRTInference>();
        inference->input_width = input_width;
        inference->input_height = input_height;
        inference->max_batch_size = max_batch_size;
        
        // 读取引擎文件
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            set_error("Failed to read engine file: " + std::string(engine_path));
            return nullptr;
        }
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        auto serialized_engine = std::make_unique<char[]>(size);
        file.read(serialized_engine.get(), size);
        file.close();
        
        // 创建运行时和引擎
        inference->runtime = createInferRuntime(gLogger);
        if (!inference->runtime) {
            set_error("Failed to create TensorRT runtime");
            return nullptr;
        }
        
        inference->engine = inference->runtime->deserializeCudaEngine(serialized_engine.get(), size);
        if (!inference->engine) {
            set_error("Failed to deserialize TensorRT engine");
            return nullptr;
        }
        
        inference->context = inference->engine->createExecutionContext();
        if (!inference->context) {
            set_error("Failed to create execution context");
            return nullptr;
        }
        
        // 创建CUDA流
        if (cudaStreamCreate(&inference->stream) != cudaSuccess) {
            set_error("Failed to create CUDA stream");
            return nullptr;
        }
        
        // 计算缓冲区大小
        int input_size = max_batch_size * 3 * input_width * input_height;
        inference->output_size = max_batch_size * 8400 * 116; // 假设的输出大小
        inference->output_seg_size = max_batch_size * 32 * (input_height / 4) * (input_width / 4);
        
        // 分配GPU内存
        if (cudaMalloc(&inference->input_buffer_device, input_size * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&inference->output_buffer_device, inference->output_size * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&inference->output_seg_buffer_device, inference->output_seg_size * sizeof(float)) != cudaSuccess) {
            set_error("Failed to allocate GPU memory");
            return nullptr;
        }
        
        // 设置张量地址
        inference->context->setTensorAddress("images", inference->input_buffer_device);
        inference->context->setTensorAddress("output", inference->output_buffer_device);
        inference->context->setTensorAddress("proto", inference->output_seg_buffer_device);
        
        inference->initialized = true;
        return inference.release();
        
    } catch (const std::exception& e) {
        set_error("Exception in tensorrt_create: " + std::string(e.what()));
        return nullptr;
    }
}

void tensorrt_destroy(TensorRTHandle handle) {
    if (handle) {
        delete static_cast<TensorRTInference*>(handle);
    }
}

bool tensorrt_inference(TensorRTHandle handle,
                       const float* input_data,
                       int batch_size,
                       float* output_data,
                       float* output_seg_data) {
    if (!handle || !input_data || !output_data || !output_seg_data) {
        set_error("Invalid parameters");
        return false;
    }
    
    try {
        auto* inference = static_cast<TensorRTInference*>(handle);
        
        if (!inference->initialized) {
            set_error("TensorRT inference not initialized");
            return false;
        }
        
        // 复制输入数据到GPU
        int input_size = batch_size * 3 * inference->input_width * inference->input_height;
        if (cudaMemcpyAsync(inference->input_buffer_device, input_data, 
                           input_size * sizeof(float), cudaMemcpyHostToDevice, 
                           inference->stream) != cudaSuccess) {
            set_error("Failed to copy input data to GPU");
            return false;
        }
        
        // 执行推理
        if (!inference->context->enqueueV3(inference->stream)) {
            set_error("Failed to execute inference");
            return false;
        }
        
        // 复制输出数据回CPU
        int output_size = batch_size * inference->output_size / inference->max_batch_size;
        int output_seg_size = batch_size * inference->output_seg_size / inference->max_batch_size;
        
        if (cudaMemcpyAsync(output_data, inference->output_buffer_device,
                           output_size * sizeof(float), cudaMemcpyDeviceToHost,
                           inference->stream) != cudaSuccess ||
            cudaMemcpyAsync(output_seg_data, inference->output_seg_buffer_device,
                           output_seg_size * sizeof(float), cudaMemcpyDeviceToHost,
                           inference->stream) != cudaSuccess) {
            set_error("Failed to copy output data from GPU");
            return false;
        }
        
        // 同步等待完成
        if (cudaStreamSynchronize(inference->stream) != cudaSuccess) {
            set_error("Failed to synchronize CUDA stream");
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        set_error("Exception in tensorrt_inference: " + std::string(e.what()));
        return false;
    }
}

bool tensorrt_get_output_sizes(TensorRTHandle handle,
                              int* output_size,
                              int* output_seg_size) {
    if (!handle || !output_size || !output_seg_size) {
        set_error("Invalid parameters");
        return false;
    }
    
    auto* inference = static_cast<TensorRTInference*>(handle);
    if (!inference->initialized) {
        set_error("TensorRT inference not initialized");
        return false;
    }
    
    *output_size = inference->output_size;
    *output_seg_size = inference->output_seg_size;
    return true;
}

const char* tensorrt_get_last_error(void) {
    return g_last_error.c_str();
}
