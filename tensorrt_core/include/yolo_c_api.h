#ifndef YOLO_C_API_H
#define YOLO_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

// 检测结果结构体
typedef struct {
    float bbox[4];      // x, y, w, h
    float confidence;   // 置信度
    int class_id;       // 类别ID
    float* mask_data;   // 掩码数据指针
    int mask_width;     // 掩码宽度
    int mask_height;    // 掩码高度
} YoloDetection;

// 推理结果结构体
typedef struct {
    YoloDetection* detections;  // 检测结果数组
    int num_detections;         // 检测数量
    double inference_time_ms;   // 推理时间(毫秒)
} YoloResult;

// YOLO推理器句柄
typedef void* YoloInferenceHandle;

/**
 * 创建YOLO推理器
 * @param engine_path TensorRT引擎文件路径
 * @param labels_path 标签文件路径
 * @return 推理器句柄，失败返回NULL
 */
YoloInferenceHandle yolo_create_inference(const char* engine_path, const char* labels_path);

/**
 * 销毁YOLO推理器
 * @param handle 推理器句柄
 */
void yolo_destroy_inference(YoloInferenceHandle handle);

/**
 * 执行推理
 * @param handle 推理器句柄
 * @param image_path 输入图片路径
 * @param result 输出结果指针
 * @return 成功返回true，失败返回false
 */
bool yolo_inference(YoloInferenceHandle handle, const char* image_path, YoloResult* result);

/**
 * 执行推理（优化版本，减少内存拷贝）
 * @param handle 推理器句柄
 * @param image_path 输入图片路径
 * @param result 输出结果指针
 * @param skip_mask_copy 是否跳过掩码数据拷贝
 * @return 成功返回true，失败返回false
 */
bool yolo_inference_fast(YoloInferenceHandle handle, const char* image_path, YoloResult* result, bool skip_mask_copy);

/**
 * 执行推理（从内存数据）
 * @param handle 推理器句柄
 * @param image_data 图片数据指针
 * @param width 图片宽度
 * @param height 图片高度
 * @param channels 图片通道数
 * @param result 输出结果指针
 * @return 成功返回true，失败返回false
 */
bool yolo_inference_from_memory(YoloInferenceHandle handle,
                                const uint8_t* image_data,
                                int width, int height, int channels,
                                YoloResult* result);

/**
 * 执行推理（从内存数据，优化版本）
 * @param handle 推理器句柄
 * @param image_data 图片数据指针
 * @param width 图片宽度
 * @param height 图片高度
 * @param channels 图片通道数
 * @param result 输出结果指针
 * @param skip_mask_copy 是否跳过掩码数据拷贝
 * @return 成功返回true，失败返回false
 */
bool yolo_inference_from_memory_fast(YoloInferenceHandle handle,
                                     const uint8_t* image_data,
                                     int width, int height, int channels,
                                     YoloResult* result, bool skip_mask_copy);

/**
 * 保存推理结果图片
 * @param handle 推理器句柄
 * @param image_path 原始图片路径
 * @param result 推理结果
 * @param output_path 输出图片路径
 * @return 成功返回true，失败返回false
 */
bool yolo_save_result_image(YoloInferenceHandle handle,
                           const char* image_path,
                           const YoloResult* result,
                           const char* output_path);

/**
 * 释放推理结果内存
 * @param result 推理结果指针
 */
void yolo_free_result(YoloResult* result);

/**
 * 获取错误信息
 * @return 最后一次错误的描述字符串
 */
const char* yolo_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // YOLO_C_API_H
