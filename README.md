# YOLO11s TensorRT Rust Bindings

[![Crates.io](https://img.shields.io/crates/v/yolo11s-tensorrt-rs)](https://crates.io/crates/yolo11s-tensorrt-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70+-blue.svg)](https://www.rust-lang.org)

高性能的 YOLO11s 分割模型 Rust 绑定，基于 TensorRT 和 CUDA 加速。

## 🚀 特性

- **高性能推理**: 基于 TensorRT 和 CUDA 的 GPU 加速，推理时间约 9ms
- **简单易用**: 极简的 Rust API，只需几行代码即可完成推理
- **完全独立**: 所有依赖都已集成，无需额外配置
- **分割支持**: 支持目标检测和实例分割
- **内存安全**: Rust 的所有权系统确保内存安全
- **性能优化**: 支持指针传递避免内存拷贝，实现高性能批量推理
- **详细计时**: 提供细粒度的性能分析，包括预处理、推理、后处理等各阶段耗时
- **命令行支持**: 支持通过命令行参数指定模型和图片路径

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

### 高性能批量推理

```rust
use yolo11s_tensorrt_rs::Yolo;

fn main() -> Result<(), String> {
    let yolo = Yolo::new("models/yolo11s-seg_steel_rail_fp16.engine", "")?;
    
    // 批量推理测试
    batch_inference_test(&yolo, "images/test1.jpg", 1000)?;
    
    Ok(())
}

fn batch_inference_test(yolo: &Yolo, image_path: &str, iterations: usize) -> Result<(), String> {
    // 读取和预处理图片（只做一次）
    let img = image::open(image_path).map_err(|e| e.to_string())?;
    let resized = img.resize(640, 640, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();
    
    // 获取 TensorRT 缓冲区和 CUDA 流
    let (input_size, output_size, output_seg_size) = yolo.get_tensorrt_info()?;
    let (input_buffer, output_buffer, output_seg_buffer) = yolo.get_tensorrt_buffers()?;
    let stream = yolo.get_cuda_stream();
    
    // 预热推理
    for _ in 0..10 {
        yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)?;
    }
    
    // 批量推理
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)?;
    }
    let total_time = start.elapsed();
    
    // 计算性能统计
    let avg_time = total_time.as_millis() as f64 / iterations as f64;
    let fps = 1000.0 / avg_time;
    
    println!("批量推理结果 ({} 次):", iterations);
    println!("  总时间: {:.2}ms", total_time.as_millis());
    println!("  平均时间: {:.2}ms", avg_time);
    println!("  FPS: {:.1}", fps);
    
    Ok(())
}
```

### 命令行参数支持

```bash
# 使用命令行参数指定模型和图片
cargo run --example basic_usage -- models/yolo11s-seg_steel_rail_fp16.engine images/test1.jpg output_ultra_fast.jpg
```

## 🧪 运行示例

### 重要：设置动态库路径

在运行示例之前，需要设置 `LD_LIBRARY_PATH` 以加载 TensorRT 核心库：

```bash
# 设置动态库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/tensorrt_core/build

# 或者一行命令运行
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/tensorrt_core/build && cargo run --example basic_usage
```

### 运行基本示例

```bash
# 设置库路径并运行示例
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/tensorrt_core/build && cargo run --example basic_usage
```

### 运行测试

```bash
# 设置库路径并运行测试
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/tensorrt_core/build && cargo run --bin test
```

确保你有以下文件：
- TensorRT 引擎文件: `models/yolo11s-seg_steel_rail_fp16.engine`
- 测试图片: `images/test1.jpg`

### 故障排除

如果遇到 `libtensorrt_core.so.1: cannot open shared object file` 错误：

1. 确认 `tensorrt_core/build/` 目录存在
2. 确认 `libtensorrt_core.so.1` 文件存在
3. 正确设置 `LD_LIBRARY_PATH`

```bash
# 检查库文件是否存在
ls -la tensorrt_core/build/libtensorrt_core.so*

# 重新编译（如果需要）
cargo clean && cargo build
```

## 🏗️ 项目结构

```
yolo11s-tensorrt-rs/
├── src/
│   ├── lib.rs              # 库入口
│   ├── yolo_c_api.rs       # C API 绑定
│   └── bin/
│       └── test.rs         # 测试示例
├── examples/
│   └── basic_usage.rs      # 基本使用示例（支持命令行参数）
├── tensorrt_core/          # C++ TensorRT 核心
│   ├── include/            # 头文件
│   ├── src/               # 源文件
│   ├── plugin/            # YOLO 插件
│   └── CMakeLists.txt     # CMake 配置
├── models/                # 模型文件
│   ├── yolo11s-seg_steel_rail_fp16.engine
│   └── yolo11s-seg_steel_rail_fp16_optimized.engine
├── images/                # 测试图片
│   ├── test1.jpg
│   └── test2.jpg
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
推理结果结构，包含详细的性能计时信息。

```rust
pub struct YoloResult {
    pub detections: *mut YoloDetection,
    pub num_detections: c_int,
    pub inference_time_ms: f64,
    // 详细计时信息
    pub image_read_time_ms: f64,
    pub preprocess_time_ms: f64,
    pub tensorrt_time_ms: f64,
    pub postprocess_time_ms: f64,
    pub result_copy_time_ms: f64,
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

#### `yolo.get_tensorrt_info()`
获取 TensorRT 缓冲区大小信息。

- 返回: `Result<(i32, i32, i32), String>` (输入大小, 输出大小, 分割输出大小)

#### `yolo.get_tensorrt_buffers()`
获取 TensorRT 缓冲区指针。

- 返回: `Result<(*mut c_void, *mut c_void, *mut c_void), String>` (输入缓冲区, 输出缓冲区, 分割输出缓冲区)

#### `yolo.get_cuda_stream()`
获取 CUDA 流指针。

- 返回: `*mut c_void`

#### `yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)`
执行纯 TensorRT 推理（无预处理和后处理）。

- `input_buffer`: 输入缓冲区指针
- `output_buffer`: 输出缓冲区指针
- `output_seg_buffer`: 分割输出缓冲区指针
- `stream`: CUDA 流指针
- 返回: `Result<bool, String>`

## 📊 性能

### 优化前 vs 优化后

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| 总推理时间 | ~333ms | ~9ms | 97% |
| C++ 内部时间 | ~88ms | ~4ms | 95% |
| 批量推理 FPS | ~3 FPS | ~140 FPS | 4600% |

### 详细性能分析

在 NVIDIA Jetson Nano 上的测试结果：

**单次推理性能**:
- **总推理时间**: ~9ms (640x640 输入)
- **TensorRT 推理**: ~4ms
- **预处理时间**: ~2ms
- **后处理时间**: ~3ms

**批量推理性能**:
- **平均推理时间**: ~7.14ms
- **FPS**: ~140
- **内存使用**: ~200MB
- **支持格式**: JPEG, PNG, BMP

**性能优化策略**:
1. **指针传递**: 避免 Rust 和 C++ 之间的内存拷贝
2. **Rust 端预处理**: 使用 `image` crate 进行图像处理
3. **纯 TensorRT 推理**: 提供直接访问 TensorRT 缓冲区的接口
4. **批量处理**: 支持高效的批量推理测试

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