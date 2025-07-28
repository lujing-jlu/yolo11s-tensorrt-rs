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

    // 详细时间分解
    pub image_read_time_ms: f64,  // 图片读取时间
    pub preprocess_time_ms: f64,  // 预处理时间
    pub tensorrt_time_ms: f64,    // TensorRT推理时间
    pub postprocess_time_ms: f64, // 后处理时间
    pub result_copy_time_ms: f64, // 结果复制时间
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

    // 新增API函数
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

    // 新增方法：获取TensorRT信息
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

    // 新增方法：获取TensorRT缓冲区
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

    // 新增方法：获取CUDA流
    pub fn get_cuda_stream(&self) -> Result<*mut std::ffi::c_void, String> {
        let stream = unsafe { yolo_get_cuda_stream(self.handle) };
        if stream.is_null() {
            return Err(last_error());
        }
        Ok(stream)
    }

    // 新增方法：纯TensorRT推理
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
        unsafe { yolo_destroy_inference(self.handle) };
    }
}

pub fn last_error() -> String {
    unsafe {
        let err = yolo_get_last_error();
        if err.is_null() {
            "Unknown error".to_string()
        } else {
            std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned()
        }
    }
}
