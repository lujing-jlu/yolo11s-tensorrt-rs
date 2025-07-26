#ifndef TENSORRT_INFERENCE_H
#define TENSORRT_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// TensorRT推理器句柄
typedef void* TensorRTHandle;

/**
 * 创建TensorRT推理器
 * @param engine_path TensorRT引擎文件路径
 * @param input_width 输入图像宽度
 * @param input_height 输入图像高度
 * @param max_batch_size 最大批次大小
 * @return 推理器句柄，失败返回NULL
 */
TensorRTHandle tensorrt_create(const char* engine_path, 
                              int input_width, 
                              int input_height, 
                              int max_batch_size);

/**
 * 销毁TensorRT推理器
 * @param handle 推理器句柄
 */
void tensorrt_destroy(TensorRTHandle handle);

/**
 * 执行推理
 * @param handle 推理器句柄
 * @param input_data 输入数据指针 (CHW格式，float32)
 * @param batch_size 批次大小
 * @param output_data 输出数据指针 (预分配)
 * @param output_seg_data 分割输出数据指针 (预分配)
 * @return 成功返回true，失败返回false
 */
bool tensorrt_inference(TensorRTHandle handle,
                       const float* input_data,
                       int batch_size,
                       float* output_data,
                       float* output_seg_data);

/**
 * 获取输出尺寸信息
 * @param handle 推理器句柄
 * @param output_size 输出检测结果大小
 * @param output_seg_size 输出分割结果大小
 * @return 成功返回true，失败返回false
 */
bool tensorrt_get_output_sizes(TensorRTHandle handle,
                              int* output_size,
                              int* output_seg_size);

/**
 * 获取错误信息
 * @return 最后一次错误的描述字符串
 */
const char* tensorrt_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // TENSORRT_INFERENCE_H
