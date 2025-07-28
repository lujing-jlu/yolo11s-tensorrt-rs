//! YOLO11s TensorRT Rust Bindings
//!
//! 高性能的 YOLO11s 分割模型 Rust 绑定，基于 TensorRT 和 CUDA 加速。
//!
//! # 特性
//!
//! - **高性能推理**: 基于 TensorRT 和 CUDA 的 GPU 加速，推理时间约 9ms
//! - **简单易用**: 极简的 Rust API，只需几行代码即可完成推理
//! - **完全独立**: 所有依赖都已集成，无需额外配置
//! - **分割支持**: 支持目标检测和实例分割
//! - **内存安全**: Rust 的所有权系统确保内存安全
//! - **性能优化**: 支持指针传递避免内存拷贝，实现高性能批量推理
//! - **详细计时**: 提供细粒度的性能分析，包括预处理、推理、后处理等各阶段耗时
//!
//! # 快速开始
//!
//! ```rust
//! use yolo11s_tensorrt_rs::{Yolo, Config};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 创建推理器
//!     let yolo = Yolo::new(Config::new("models/yolo11s-seg.engine"))?;
//!     
//!     // 执行推理
//!     let result = yolo.inference("images/test.jpg")?;
//!     
//!     // 处理结果
//!     println!("检测到 {} 个目标", result.detection_count());
//!     
//!     for detection in result.detections() {
//!         println!("类别: {}, 置信度: {:.3}", detection.class_id(), detection.confidence());
//!     }
//!     
//!     // 保存结果图片
//!     yolo.save_result_image("images/test.jpg", &result, "output.jpg")?;
//!     
//!     Ok(())
//! }
//! ```
//!
//! # 批量推理
//!
//! ```rust
//! use yolo11s_tensorrt_rs::Yolo;
//!
//! let yolo = Yolo::with_engine("models/yolo11s-seg.engine")?;
//!
//! // 执行批量推理测试
//! let stats = yolo.batch_inference_test("images/test.jpg", 1000)?;
//! println!("平均 FPS: {:.1}", stats.fps());
//! println!("TensorRT 推理占比: {:.1}%", stats.tensorrt_percentage());
//! ```
//!
//! # 性能优化
//!
//! ```rust
//! use yolo11s_tensorrt_rs::Yolo;
//!
//! let yolo = Yolo::with_engine("models/yolo11s-seg.engine")?;
//!
//! // 获取 TensorRT 缓冲区
//! let buffers = yolo.get_tensorrt_buffers()?;
//! let stream = yolo.get_cuda_stream()?;
//!
//! // 执行纯 TensorRT 推理（无预处理和后处理）
//! yolo.tensorrt_inference_only(
//!     buffers.input_buffer,
//!     buffers.output_buffer,
//!     buffers.output_seg_buffer,
//!     stream,
//! )?;
//! ```

pub mod error;
pub mod types;
pub mod yolo;

// 重新导出主要类型
pub use error::{YoloError, YoloResult};
pub use types::{
    Config, Detection, InferenceResult, PerformanceBreakdown, TensorRtBuffers, TensorRtInfo,
};
pub use yolo::Yolo;

// 为了向后兼容，保留旧的 API
#[deprecated(since = "0.2.0", note = "请使用新的 Yolo 结构")]
pub mod yolo_c_api {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};

    #[repr(C)]
    pub struct YoloDetection {
        pub bbox: [f32; 4],
        pub confidence: f32,
        pub class_id: c_int,
        pub mask_data: *mut f32,
        pub mask_width: c_int,
        pub mask_height: c_int,
    }

    #[repr(C)]
    pub struct YoloResult {
        pub detections: *mut YoloDetection,
        pub num_detections: c_int,
        pub inference_time_ms: f64,
        pub image_read_time_ms: f64,
        pub preprocess_time_ms: f64,
        pub tensorrt_time_ms: f64,
        pub postprocess_time_ms: f64,
        pub result_copy_time_ms: f64,
    }

    pub type YoloInferenceHandle = *mut c_void;

    extern "C" {
        pub fn yolo_create_inference(
            engine_path: *const c_char,
            labels_path: *const c_char,
        ) -> YoloInferenceHandle;
        pub fn yolo_destroy_inference(handle: YoloInferenceHandle);
        pub fn yolo_inference(
            handle: YoloInferenceHandle,
            image_path: *const c_char,
            result: *mut YoloResult,
        ) -> bool;
        pub fn yolo_save_result_image(
            handle: YoloInferenceHandle,
            image_path: *const c_char,
            result: *const YoloResult,
            output_path: *const c_char,
        ) -> bool;
        pub fn yolo_free_result(result: *mut YoloResult);
        pub fn yolo_get_last_error() -> *const c_char;
        pub fn yolo_tensorrt_inference_only(
            handle: YoloInferenceHandle,
            input_buffer: *mut c_void,
            output_buffer: *mut c_void,
            output_seg_buffer: *mut c_void,
            stream: *mut c_void,
        ) -> bool;
        pub fn yolo_get_tensorrt_info(
            handle: YoloInferenceHandle,
            input_size: *mut c_int,
            output_size: *mut c_int,
            output_seg_size: *mut c_int,
        ) -> bool;
        pub fn yolo_get_tensorrt_buffers(
            handle: YoloInferenceHandle,
            input_buffer: *mut *mut c_void,
            output_buffer: *mut *mut c_void,
            output_seg_buffer: *mut *mut c_void,
        ) -> bool;
        pub fn yolo_get_cuda_stream(handle: YoloInferenceHandle) -> *mut c_void;
    }

    pub struct Yolo {
        handle: YoloInferenceHandle,
    }

    impl Yolo {
        pub fn new(engine_path: &str, labels_path: &str) -> Result<Self, String> {
            let engine_c = CString::new(engine_path).map_err(|e| e.to_string())?;
            let labels_c = CString::new(labels_path).map_err(|e| e.to_string())?;
            let handle = unsafe { yolo_create_inference(engine_c.as_ptr(), labels_c.as_ptr()) };
            if handle.is_null() {
                return Err(last_error());
            }
            Ok(Yolo { handle })
        }

        pub fn inference(&self, image_path: &str) -> Result<YoloResult, String> {
            let image_c = CString::new(image_path).map_err(|e| e.to_string())?;
            let mut result = YoloResult {
                detections: std::ptr::null_mut(),
                num_detections: 0,
                inference_time_ms: 0.0,
                image_read_time_ms: 0.0,
                preprocess_time_ms: 0.0,
                tensorrt_time_ms: 0.0,
                postprocess_time_ms: 0.0,
                result_copy_time_ms: 0.0,
            };
            let ok = unsafe { yolo_inference(self.handle, image_c.as_ptr(), &mut result) };
            if !ok {
                return Err(last_error());
            }
            Ok(result)
        }

        pub fn save_result_image(
            &self,
            image_path: &str,
            result: &YoloResult,
            output_path: &str,
        ) -> Result<(), String> {
            let image_c = CString::new(image_path).map_err(|e| e.to_string())?;
            let output_c = CString::new(output_path).map_err(|e| e.to_string())?;
            let ok = unsafe {
                yolo_save_result_image(self.handle, image_c.as_ptr(), result, output_c.as_ptr())
            };
            if !ok {
                return Err(last_error());
            }
            Ok(())
        }

        pub fn get_tensorrt_info(&self) -> Result<(i32, i32, i32), String> {
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
                return Err(last_error());
            }
            Ok((input_size, output_size, output_seg_size))
        }

        pub fn get_tensorrt_buffers(
            &self,
        ) -> Result<
            (
                *mut std::ffi::c_void,
                *mut std::ffi::c_void,
                *mut std::ffi::c_void,
            ),
            String,
        > {
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
                return Err(last_error());
            }
            Ok((input_buffer, output_buffer, output_seg_buffer))
        }

        pub fn get_cuda_stream(&self) -> Result<*mut std::ffi::c_void, String> {
            let stream = unsafe { yolo_get_cuda_stream(self.handle) };
            if stream.is_null() {
                return Err("Failed to get CUDA stream".to_string());
            }
            Ok(stream)
        }

        pub fn tensorrt_inference_only(
            &self,
            input_buffer: *mut std::ffi::c_void,
            output_buffer: *mut std::ffi::c_void,
            output_seg_buffer: *mut std::ffi::c_void,
            stream: *mut std::ffi::c_void,
        ) -> Result<(), String> {
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
                return Err(last_error());
            }
            Ok(())
        }
    }

    impl Drop for Yolo {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { yolo_destroy_inference(self.handle) };
            }
        }
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
}
