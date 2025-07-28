use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

use crate::error::{YoloError, YoloResult};
use crate::types::{
    Config, Detection, InferenceResult, PerformanceBreakdown, TensorRtBuffers, TensorRtInfo,
    YoloInferenceHandle, YoloResult as YoloResultRaw,
};

/// YOLO11s 推理器
///
/// 提供高性能的目标检测和实例分割功能。
///
/// # 示例
///
/// ```rust
/// use yolo11s_tensorrt_rs::{Yolo, Config};
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // 创建推理器
///     let yolo = Yolo::new(Config::new("models/yolo11s-seg.engine"))?;
///     
///     // 执行推理
///     let result = yolo.inference("images/test.jpg")?;
///     
///     // 处理结果
///     println!("检测到 {} 个目标", result.detection_count());
///     
///     Ok(())
/// }
/// ```
pub struct Yolo {
    handle: YoloInferenceHandle,
    config: Config,
}

impl Yolo {
    /// 创建新的 YOLO 推理器
    ///
    /// # 参数
    ///
    /// * `config` - 配置选项
    ///
    /// # 示例
    ///
    /// ```rust
    /// use yolo11s_tensorrt_rs::{Yolo, Config};
    ///
    /// let yolo = Yolo::new(Config::new("models/yolo11s-seg.engine"))?;
    /// ```
    pub fn new(config: Config) -> YoloResult<Self> {
        let engine_c = CString::new(&*config.engine_path)
            .map_err(|e| YoloError::InvalidParameter(e.to_string()))?;
        let labels_c = CString::new(&*config.labels_path)
            .map_err(|e| YoloError::InvalidParameter(e.to_string()))?;

        let handle = unsafe { yolo_create_inference(engine_c.as_ptr(), labels_c.as_ptr()) };
        if handle.is_null() {
            return Err(YoloError::Initialization(last_error()));
        }

        Ok(Yolo { handle, config })
    }

    /// 使用默认配置创建推理器
    ///
    /// # 参数
    ///
    /// * `engine_path` - TensorRT 引擎文件路径
    ///
    /// # 示例
    ///
    /// ```rust
    /// use yolo11s_tensorrt_rs::Yolo;
    ///
    /// let yolo = Yolo::with_engine("models/yolo11s-seg.engine")?;
    /// ```
    pub fn with_engine(engine_path: &str) -> YoloResult<Self> {
        Self::new(Config::new(engine_path))
    }

    /// 执行推理
    ///
    /// # 参数
    ///
    /// * `image_path` - 输入图片路径
    ///
    /// # 返回值
    ///
    /// 返回包含检测结果和性能数据的 `InferenceResult`
    ///
    /// # 示例
    ///
    /// ```rust
    /// let result = yolo.inference("images/test.jpg")?;
    /// println!("检测到 {} 个目标", result.detection_count());
    ///
    /// for detection in result.detections() {
    ///     println!("类别: {}, 置信度: {:.3}", detection.class_id(), detection.confidence());
    /// }
    /// ```
    pub fn inference(&self, image_path: &str) -> YoloResult<InferenceResult> {
        let image_c =
            CString::new(image_path).map_err(|e| YoloError::InvalidParameter(e.to_string()))?;

        let mut raw_result = YoloResultRaw {
            detections: std::ptr::null_mut(),
            num_detections: 0,
            inference_time_ms: 0.0,
            image_read_time_ms: 0.0,
            preprocess_time_ms: 0.0,
            tensorrt_time_ms: 0.0,
            postprocess_time_ms: 0.0,
            result_copy_time_ms: 0.0,
        };

        let ok = unsafe { yolo_inference(self.handle, image_c.as_ptr(), &mut raw_result) };
        if !ok {
            return Err(YoloError::Inference(last_error()));
        }

        // 转换结果
        let mut result = InferenceResult::new();
        result.total_time_ms = raw_result.inference_time_ms;
        result.image_read_time_ms = raw_result.image_read_time_ms;
        result.preprocess_time_ms = raw_result.preprocess_time_ms;
        result.tensorrt_time_ms = raw_result.tensorrt_time_ms;
        result.postprocess_time_ms = raw_result.postprocess_time_ms;
        result.result_copy_time_ms = raw_result.result_copy_time_ms;

        // 转换检测结果
        if !raw_result.detections.is_null() && raw_result.num_detections > 0 {
            for i in 0..raw_result.num_detections {
                let detection_ptr = unsafe { raw_result.detections.offset(i as isize) };
                let raw_detection = unsafe { &*detection_ptr };

                let mut detection = Detection::new(
                    raw_detection.bbox,
                    raw_detection.confidence,
                    raw_detection.class_id,
                );

                // 处理分割掩码
                if !raw_detection.mask_data.is_null()
                    && raw_detection.mask_width > 0
                    && raw_detection.mask_height > 0
                {
                    let mask_size = (raw_detection.mask_width * raw_detection.mask_height) as usize;
                    let mask_data =
                        unsafe { std::slice::from_raw_parts(raw_detection.mask_data, mask_size) }
                            .to_vec();

                    detection = detection.with_mask(
                        mask_data,
                        raw_detection.mask_width,
                        raw_detection.mask_height,
                    );
                }

                result.add_detection(detection);
            }
        }

        // 释放原始结果
        unsafe { yolo_free_result(&mut raw_result) };

        Ok(result)
    }

    /// 保存推理结果图片
    ///
    /// # 参数
    ///
    /// * `image_path` - 原始图片路径
    /// * `result` - 推理结果
    /// * `output_path` - 输出图片路径
    ///
    /// # 示例
    ///
    /// ```rust
    /// let result = yolo.inference("images/test.jpg")?;
    /// yolo.save_result_image("images/test.jpg", &result, "output.jpg")?;
    /// ```
    pub fn save_result_image(
        &self,
        image_path: &str,
        result: &InferenceResult,
        output_path: &str,
    ) -> YoloResult<()> {
        // 直接调用 C++ API 进行推理和保存
        let image_c =
            CString::new(image_path).map_err(|e| YoloError::InvalidParameter(e.to_string()))?;
        let output_c =
            CString::new(output_path).map_err(|e| YoloError::InvalidParameter(e.to_string()))?;

        let mut raw_result = YoloResultRaw {
            detections: std::ptr::null_mut(),
            num_detections: 0,
            inference_time_ms: 0.0,
            image_read_time_ms: 0.0,
            preprocess_time_ms: 0.0,
            tensorrt_time_ms: 0.0,
            postprocess_time_ms: 0.0,
            result_copy_time_ms: 0.0,
        };

        // 重新执行推理以获取原始结果
        let ok = unsafe { yolo_inference(self.handle, image_c.as_ptr(), &mut raw_result) };
        if !ok {
            return Err(YoloError::Inference(last_error()));
        }

        // 保存结果
        let save_ok = unsafe {
            yolo_save_result_image(
                self.handle,
                image_c.as_ptr(),
                &raw_result,
                output_c.as_ptr(),
            )
        };
        if !save_ok {
            return Err(YoloError::File(last_error()));
        }

        // 释放结果
        unsafe { yolo_free_result(&mut raw_result) };

        Ok(())
    }

    /// 获取 TensorRT 缓冲区信息
    ///
    /// # 返回值
    ///
    /// 返回包含缓冲区大小信息的 `TensorRtInfo`
    pub fn get_tensorrt_info(&self) -> YoloResult<TensorRtInfo> {
        let mut input_size = 0;
        let mut output_size = 0;
        let mut output_seg_size = 0;

        let ok = unsafe {
            yolo_get_tensorrt_info(
                self.handle,
                &mut input_size,
                &mut output_size,
                &mut output_seg_size,
            )
        };

        if !ok {
            return Err(YoloError::TensorRt(last_error()));
        }

        Ok(TensorRtInfo {
            input_size,
            output_size,
            output_seg_size,
        })
    }

    /// 获取 TensorRT 缓冲区指针
    ///
    /// # 返回值
    ///
    /// 返回包含缓冲区指针的 `TensorRtBuffers`
    pub fn get_tensorrt_buffers(&self) -> YoloResult<TensorRtBuffers> {
        let mut input_buffer = std::ptr::null_mut();
        let mut output_buffer = std::ptr::null_mut();
        let mut output_seg_buffer = std::ptr::null_mut();

        let ok = unsafe {
            yolo_get_tensorrt_buffers(
                self.handle,
                &mut input_buffer,
                &mut output_buffer,
                &mut output_seg_buffer,
            )
        };

        if !ok {
            return Err(YoloError::TensorRt(last_error()));
        }

        Ok(TensorRtBuffers {
            input_buffer,
            output_buffer,
            output_seg_buffer,
        })
    }

    /// 获取 CUDA 流指针
    ///
    /// # 返回值
    ///
    /// 返回 CUDA 流指针
    pub fn get_cuda_stream(&self) -> YoloResult<*mut c_void> {
        let stream = unsafe { yolo_get_cuda_stream(self.handle) };
        if stream.is_null() {
            return Err(YoloError::Cuda("Failed to get CUDA stream".to_string()));
        }
        Ok(stream)
    }

    /// 执行纯 TensorRT 推理
    ///
    /// 这个函数只执行 TensorRT 推理部分，不包含预处理和后处理。
    /// 适用于高性能批量推理场景。
    ///
    /// # 参数
    ///
    /// * `input_buffer` - 输入缓冲区指针
    /// * `output_buffer` - 输出缓冲区指针
    /// * `output_seg_buffer` - 分割输出缓冲区指针
    /// * `stream` - CUDA 流指针
    ///
    /// # 示例
    ///
    /// ```rust
    /// let buffers = yolo.get_tensorrt_buffers()?;
    /// let stream = yolo.get_cuda_stream()?;
    ///
    /// // 执行纯推理
    /// yolo.tensorrt_inference_only(
    ///     buffers.input_buffer,
    ///     buffers.output_buffer,
    ///     buffers.output_seg_buffer,
    ///     stream,
    /// )?;
    /// ```
    pub fn tensorrt_inference_only(
        &self,
        input_buffer: *mut c_void,
        output_buffer: *mut c_void,
        output_seg_buffer: *mut c_void,
        stream: *mut c_void,
    ) -> YoloResult<()> {
        let ok = unsafe {
            yolo_tensorrt_inference_only(
                self.handle,
                input_buffer,
                output_buffer,
                output_seg_buffer,
                stream,
            )
        };

        if !ok {
            return Err(YoloError::TensorRt(last_error()));
        }

        Ok(())
    }

    /// 获取配置信息
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// 执行批量推理测试
    ///
    /// # 参数
    ///
    /// * `_image_path` - 测试图片路径
    /// * `iterations` - 推理次数
    ///
    /// # 返回值
    ///
    /// 返回性能统计信息
    ///
    /// # 示例
    ///
    /// ```rust
    /// let stats = yolo.batch_inference_test("images/test.jpg", 1000)?;
    /// println!("平均 FPS: {:.1}", stats.fps());
    /// ```
    pub fn batch_inference_test(
        &self,
        _image_path: &str,
        iterations: usize,
    ) -> YoloResult<PerformanceBreakdown> {
        // 获取 TensorRT 缓冲区和流
        let buffers = self.get_tensorrt_buffers()?;
        let stream = self.get_cuda_stream()?;

        // 预热
        for _ in 0..10 {
            self.tensorrt_inference_only(
                buffers.input_buffer,
                buffers.output_buffer,
                buffers.output_seg_buffer,
                stream,
            )?;
        }

        // 批量推理
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            self.tensorrt_inference_only(
                buffers.input_buffer,
                buffers.output_buffer,
                buffers.output_seg_buffer,
                stream,
            )?;
        }
        let total_time = start.elapsed();

        let avg_time = total_time.as_millis() as f64 / iterations as f64;

        Ok(PerformanceBreakdown {
            total_time_ms: avg_time,
            image_read_time_ms: 0.0,
            preprocess_time_ms: 0.0,
            tensorrt_time_ms: avg_time,
            postprocess_time_ms: 0.0,
            result_copy_time_ms: 0.0,
        })
    }
}

impl Drop for Yolo {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { yolo_destroy_inference(self.handle) };
        }
    }
}

// C API 函数声明
extern "C" {
    fn yolo_create_inference(
        engine_path: *const c_char,
        labels_path: *const c_char,
    ) -> YoloInferenceHandle;
    fn yolo_destroy_inference(handle: YoloInferenceHandle);
    fn yolo_inference(
        handle: YoloInferenceHandle,
        image_path: *const c_char,
        result: *mut YoloResultRaw,
    ) -> bool;
    fn yolo_save_result_image(
        handle: YoloInferenceHandle,
        image_path: *const c_char,
        result: *const YoloResultRaw,
        output_path: *const c_char,
    ) -> bool;
    fn yolo_free_result(result: *mut YoloResultRaw);
    fn yolo_get_last_error() -> *const c_char;
    fn yolo_tensorrt_inference_only(
        handle: YoloInferenceHandle,
        input_buffer: *mut c_void,
        output_buffer: *mut c_void,
        output_seg_buffer: *mut c_void,
        stream: *mut c_void,
    ) -> bool;
    fn yolo_get_tensorrt_info(
        handle: YoloInferenceHandle,
        input_size: *mut c_int,
        output_size: *mut c_int,
        output_seg_size: *mut c_int,
    ) -> bool;
    fn yolo_get_tensorrt_buffers(
        handle: YoloInferenceHandle,
        input_buffer: *mut *mut c_void,
        output_buffer: *mut *mut c_void,
        output_seg_buffer: *mut *mut c_void,
    ) -> bool;
    fn yolo_get_cuda_stream(handle: YoloInferenceHandle) -> *mut c_void;
}

fn last_error() -> String {
    unsafe {
        let error_ptr = yolo_get_last_error();
        if error_ptr.is_null() {
            "Unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(error_ptr)
                .to_string_lossy()
                .into_owned()
        }
    }
}
