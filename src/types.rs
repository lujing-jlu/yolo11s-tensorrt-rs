use std::os::raw::{c_int, c_void};

/// 检测结果结构
#[derive(Debug, Clone)]
pub struct Detection {
    /// 边界框 [x, y, width, height]
    pub bbox: [f32; 4],
    /// 置信度
    pub confidence: f32,
    /// 类别 ID
    pub class_id: i32,
    /// 分割掩码数据
    pub mask_data: Vec<f32>,
    /// 掩码宽度
    pub mask_width: i32,
    /// 掩码高度
    pub mask_height: i32,
}

impl Detection {
    /// 创建新的检测结果
    pub fn new(bbox: [f32; 4], confidence: f32, class_id: i32) -> Self {
        Self {
            bbox,
            confidence,
            class_id,
            mask_data: Vec::new(),
            mask_width: 0,
            mask_height: 0,
        }
    }

    /// 设置分割掩码
    pub fn with_mask(mut self, mask_data: Vec<f32>, width: i32, height: i32) -> Self {
        self.mask_data = mask_data;
        self.mask_width = width;
        self.mask_height = height;
        self
    }

    /// 获取边界框坐标
    pub fn bbox(&self) -> [f32; 4] {
        self.bbox
    }

    /// 获取置信度
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// 获取类别 ID
    pub fn class_id(&self) -> i32 {
        self.class_id
    }

    /// 检查是否有分割掩码
    pub fn has_mask(&self) -> bool {
        !self.mask_data.is_empty()
    }

    /// 获取掩码数据
    pub fn mask_data(&self) -> &[f32] {
        &self.mask_data
    }

    /// 获取掩码尺寸
    pub fn mask_size(&self) -> (i32, i32) {
        (self.mask_width, self.mask_height)
    }
}

/// 推理结果结构
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// 检测结果列表
    pub detections: Vec<Detection>,
    /// 总推理时间（毫秒）
    pub total_time_ms: f64,
    /// 图片读取时间（毫秒）
    pub image_read_time_ms: f64,
    /// 预处理时间（毫秒）
    pub preprocess_time_ms: f64,
    /// TensorRT 推理时间（毫秒）
    pub tensorrt_time_ms: f64,
    /// 后处理时间（毫秒）
    pub postprocess_time_ms: f64,
    /// 结果复制时间（毫秒）
    pub result_copy_time_ms: f64,
}

impl InferenceResult {
    /// 创建新的推理结果
    pub fn new() -> Self {
        Self {
            detections: Vec::new(),
            total_time_ms: 0.0,
            image_read_time_ms: 0.0,
            preprocess_time_ms: 0.0,
            tensorrt_time_ms: 0.0,
            postprocess_time_ms: 0.0,
            result_copy_time_ms: 0.0,
        }
    }

    /// 添加检测结果
    pub fn add_detection(&mut self, detection: Detection) {
        self.detections.push(detection);
    }

    /// 获取检测结果数量
    pub fn detection_count(&self) -> usize {
        self.detections.len()
    }

    /// 获取所有检测结果
    pub fn detections(&self) -> &[Detection] {
        &self.detections
    }

    /// 获取总推理时间
    pub fn total_time_ms(&self) -> f64 {
        self.total_time_ms
    }

    /// 获取详细的性能数据
    pub fn performance_breakdown(&self) -> PerformanceBreakdown {
        PerformanceBreakdown {
            total_time_ms: self.total_time_ms,
            image_read_time_ms: self.image_read_time_ms,
            preprocess_time_ms: self.preprocess_time_ms,
            tensorrt_time_ms: self.tensorrt_time_ms,
            postprocess_time_ms: self.postprocess_time_ms,
            result_copy_time_ms: self.result_copy_time_ms,
        }
    }
}

/// 性能分析结构
#[derive(Debug, Clone)]
pub struct PerformanceBreakdown {
    /// 总时间（毫秒）
    pub total_time_ms: f64,
    /// 图片读取时间（毫秒）
    pub image_read_time_ms: f64,
    /// 预处理时间（毫秒）
    pub preprocess_time_ms: f64,
    /// TensorRT 推理时间（毫秒）
    pub tensorrt_time_ms: f64,
    /// 后处理时间（毫秒）
    pub postprocess_time_ms: f64,
    /// 结果复制时间（毫秒）
    pub result_copy_time_ms: f64,
}

impl PerformanceBreakdown {
    /// 计算 FPS
    pub fn fps(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            1000.0 / self.total_time_ms
        } else {
            0.0
        }
    }

    /// 获取 TensorRT 推理占比
    pub fn tensorrt_percentage(&self) -> f64 {
        if self.total_time_ms > 0.0 {
            (self.tensorrt_time_ms / self.total_time_ms) * 100.0
        } else {
            0.0
        }
    }
}

/// TensorRT 缓冲区信息
#[derive(Debug, Clone)]
pub struct TensorRtInfo {
    /// 输入缓冲区大小
    pub input_size: i32,
    /// 输出缓冲区大小
    pub output_size: i32,
    /// 分割输出缓冲区大小
    pub output_seg_size: i32,
}

/// TensorRT 缓冲区指针
#[derive(Debug)]
pub struct TensorRtBuffers {
    /// 输入缓冲区指针
    pub input_buffer: *mut c_void,
    /// 输出缓冲区指针
    pub output_buffer: *mut c_void,
    /// 分割输出缓冲区指针
    pub output_seg_buffer: *mut c_void,
}

/// 配置选项
#[derive(Debug, Clone)]
pub struct Config {
    /// 引擎文件路径
    pub engine_path: String,
    /// 标签文件路径
    pub labels_path: String,
    /// 是否启用详细日志
    pub verbose: bool,
    /// 推理批次大小
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            engine_path: String::new(),
            labels_path: String::new(),
            verbose: false,
            batch_size: 1,
        }
    }
}

impl Config {
    /// 创建新的配置
    pub fn new(engine_path: &str) -> Self {
        Self {
            engine_path: engine_path.to_string(),
            labels_path: String::new(),
            verbose: false,
            batch_size: 1,
        }
    }

    /// 设置标签文件路径
    pub fn with_labels(mut self, labels_path: &str) -> Self {
        self.labels_path = labels_path.to_string();
        self
    }

    /// 启用详细日志
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 设置批次大小
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

// 内部使用的 C API 结构
#[repr(C)]
pub(crate) struct YoloDetection {
    pub bbox: [f32; 4],
    pub confidence: f32,
    pub class_id: c_int,
    pub mask_data: *mut f32,
    pub mask_width: c_int,
    pub mask_height: c_int,
}

#[repr(C)]
pub(crate) struct YoloResult {
    pub detections: *mut YoloDetection,
    pub num_detections: c_int,
    pub inference_time_ms: f64,
    pub image_read_time_ms: f64,
    pub preprocess_time_ms: f64,
    pub tensorrt_time_ms: f64,
    pub postprocess_time_ms: f64,
    pub result_copy_time_ms: f64,
}

pub(crate) type YoloInferenceHandle = *mut c_void;
