use std::fmt;

/// YOLO 推理错误类型
#[derive(Debug)]
pub enum YoloError {
    /// 初始化错误
    Initialization(String),
    /// 推理错误
    Inference(String),
    /// 文件操作错误
    File(String),
    /// 内存错误
    Memory(String),
    /// CUDA 错误
    Cuda(String),
    /// TensorRT 错误
    TensorRt(String),
    /// 参数错误
    InvalidParameter(String),
    /// 未知错误
    Unknown(String),
}

impl fmt::Display for YoloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            YoloError::Initialization(msg) => write!(f, "初始化错误: {}", msg),
            YoloError::Inference(msg) => write!(f, "推理错误: {}", msg),
            YoloError::File(msg) => write!(f, "文件错误: {}", msg),
            YoloError::Memory(msg) => write!(f, "内存错误: {}", msg),
            YoloError::Cuda(msg) => write!(f, "CUDA 错误: {}", msg),
            YoloError::TensorRt(msg) => write!(f, "TensorRT 错误: {}", msg),
            YoloError::InvalidParameter(msg) => write!(f, "参数错误: {}", msg),
            YoloError::Unknown(msg) => write!(f, "未知错误: {}", msg),
        }
    }
}

impl std::error::Error for YoloError {}

impl From<std::io::Error> for YoloError {
    fn from(err: std::io::Error) -> Self {
        YoloError::File(err.to_string())
    }
}

impl From<std::ffi::NulError> for YoloError {
    fn from(err: std::ffi::NulError) -> Self {
        YoloError::InvalidParameter(err.to_string())
    }
}

impl From<String> for YoloError {
    fn from(err: String) -> Self {
        YoloError::Unknown(err)
    }
}

impl From<&str> for YoloError {
    fn from(err: &str) -> Self {
        YoloError::Unknown(err.to_string())
    }
}

/// 结果类型别名
pub type YoloResult<T> = Result<T, YoloError>;
