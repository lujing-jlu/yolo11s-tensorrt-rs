# YOLO11s TensorRT Rust Bindings

[![Crates.io](https://img.shields.io/crates/v/yolo11s-tensorrt-rs)](https://crates.io/crates/yolo11s-tensorrt-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org)

高性能的 YOLO11s 分割模型 Rust 绑定，基于 TensorRT 和 CUDA 加速。

## 🚀 特性

- **高性能推理**: 基于 TensorRT 和 CUDA 的 GPU 加速
- **简单易用**: 极简的 Rust API，只需几行代码即可完成推理
- **完全独立**: 所有依赖都已集成，无需额外配置
- **分割支持**: 支持目标检测和实例分割
- **内存安全**: Rust 的所有权系统确保内存安全

## 📋 系统要求

- **CUDA**: >= 11.0
- **TensorRT**: >= 8.0
- **OpenCV**: >= 4.0
- **Rust**: >= 1.70
- **操作系统**: Linux (Ubuntu 20.04+)

## 🔧 安装

### 1. 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libopencv-dev nvidia-cuda-toolkit

# 安装 TensorRT (根据你的 CUDA 版本选择)
# 参考: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/
```

### 2. 添加依赖到 Cargo.toml

```toml
[dependencies]
yolo11s-tensorrt-rs = "0.1.0"
```

### 3. 编译项目

```bash
cargo build --release
```

## 📖 使用方法

### 基本用法

```rust
use yolo11s_tensorrt_rs::Yolo;

fn main() -> Result<(), String> {
    // 创建 YOLO 推理器
    let yolo = Yolo::new("path/to/model.engine", "")?;
    
    // 执行推理
    let result = yolo.inference("path/to/image.jpg")?;
    
    // 保存结果
    yolo.save_result_image("path/to/image.jpg", &result, "output.jpg")?;
    
    println!("检测到 {} 个目标", result.num_detections);
    println!("推理耗时: {:.2}ms", result.inference_time_ms);
    
    Ok(())
}
```

### 详细示例

```rust
use yolo11s_tensorrt_rs::{Yolo, YoloResult};

fn main() -> Result<(), String> {
    // 初始化推理器
    let yolo = Yolo::new("models/yolo11s-seg.engine", "")?;
    
    // 推理图片
    let result = yolo.inference("images/test.jpg")?;
    
    // 处理检测结果
    if result.num_detections > 0 {
        println!("检测结果:");
        for i in 0..result.num_detections {
            let detection = unsafe { &*result.detections.offset(i as isize) };
            println!("  目标 {}: 类别={}, 置信度={:.3}", 
                i + 1, detection.class_id, detection.confidence);
            println!("    边界框: [{:.1}, {:.1}, {:.1}, {:.1}]", 
                detection.bbox[0], detection.bbox[1], 
                detection.bbox[2], detection.bbox[3]);
        }
    }
    
    // 保存可视化结果
    yolo.save_result_image("images/test.jpg", &result, "output_result.jpg")?;
    
    Ok(())
}
```

## 🧪 运行示例

### 运行基本示例

```bash
cargo run --example basic_usage
```

### 运行测试

```bash
cargo run --bin test
```

确保你有以下文件：
- TensorRT 引擎文件: `models/yolo11s-seg.engine`
- 测试图片: `images/test.jpg`

## 🏗️ 项目结构

```
yolo11s-tensorrt-rs/
├── src/
│   ├── lib.rs              # 库入口
│   ├── yolo_c_api.rs       # C API 绑定
│   └── bin/
│       └── test.rs         # 测试示例
├── examples/
│   └── basic_usage.rs      # 基本使用示例
├── tensorrt_core/          # C++ TensorRT 核心
│   ├── include/            # 头文件
│   ├── src/               # 源文件
│   ├── plugin/            # YOLO 插件
│   └── CMakeLists.txt     # CMake 配置
├── Cargo.toml             # Rust 配置
├── build.rs              # 构建脚本
└── README.md             # 项目文档
```

## 🔍 API 参考

### 主要类型

#### `Yolo`
主要的推理器类型。

```rust
pub struct Yolo {
    handle: YoloInferenceHandle,
}
```

#### `YoloResult`
推理结果结构。

```rust
pub struct YoloResult {
    pub detections: *mut YoloDetection,
    pub num_detections: c_int,
    pub inference_time_ms: f64,
}
```

#### `YoloDetection`
单个检测结果。

```rust
pub struct YoloDetection {
    pub bbox: [f32; 4],        // [x, y, w, h]
    pub confidence: f32,       // 置信度
    pub class_id: c_int,       // 类别 ID
    pub mask_data: *mut f32,   // 分割掩码数据
    pub mask_width: c_int,     // 掩码宽度
    pub mask_height: c_int,    // 掩码高度
}
```

### 主要方法

#### `Yolo::new(engine_path, labels_path)`
创建新的 YOLO 推理器。

- `engine_path`: TensorRT 引擎文件路径
- `labels_path`: 标签文件路径（当前版本忽略，只支持 "defect" 类别）

#### `yolo.inference(image_path)`
对图片执行推理。

- `image_path`: 输入图片路径
- 返回: `Result<YoloResult, String>`

#### `yolo.save_result_image(image_path, result, output_path)`
保存推理结果图片。

- `image_path`: 原始图片路径
- `result`: 推理结果
- `output_path`: 输出图片路径

## 📊 性能

在 NVIDIA Jetson Nano 上的测试结果：
- **推理时间**: ~260ms (640x640 输入)
- **内存使用**: ~200MB
- **支持格式**: JPEG, PNG, BMP

## 🤝 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [YOLO11](https://github.com/WongKinYiu/yolov11) - 原始模型
- [TensorRT](https://developer.nvidia.com/tensorrt) - NVIDIA 推理引擎
- [OpenCV](https://opencv.org/) - 计算机视觉库

## 📞 支持

如果你遇到问题或有建议，请：

1. 查看 [Issues](https://github.com/your-username/yolo11s-tensorrt-rs/issues)
2. 创建新的 Issue
3. 联系维护者

---

⭐ 如果这个项目对你有帮助，请给它一个星标！ 