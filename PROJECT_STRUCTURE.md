# 项目结构说明

## 目录结构

```
yolo11s-tensorrt-rs/
├── src/                          # Rust 源代码
│   ├── lib.rs                    # 库入口点
│   ├── yolo_c_api.rs            # C API 绑定
│   └── bin/
│       └── test.rs              # 测试示例
├── examples/                     # 使用示例
│   └── basic_usage.rs           # 基本使用示例（支持命令行参数）
├── tensorrt_core/               # C++ TensorRT 核心
│   ├── include/                 # 头文件
│   │   └── yolo_c_api.h        # C API 头文件
│   ├── src/                     # C++ 源文件
│   │   └── yolo_c_api.cpp      # C API 实现
│   ├── plugin/                  # YOLO 插件
│   ├── build/                   # 构建输出（自动生成）
│   └── CMakeLists.txt          # CMake 配置
├── models/                      # 模型文件
│   ├── yolo11s-seg_steel_rail_fp16.engine          # 优化的 TensorRT 引擎
│   ├── yolo11s-seg_steel_rail_fp16_optimized.engine # 进一步优化的引擎
│   ├── yolo11s-seg_steel_rail.wts                  # 模型权重文件
│   └── yolo11s-seg_steel_rail.pt                   # PyTorch 模型文件
├── images/                      # 测试图片
│   ├── test1.jpg               # 测试图片 1
│   └── test2.jpg               # 测试图片 2
├── Cargo.toml                  # Rust 项目配置
├── Cargo.lock                  # Rust 依赖锁定文件
├── build.rs                    # Rust 构建脚本
├── README.md                   # 项目文档
├── LICENSE                     # 许可证文件
├── .gitignore                  # Git 忽略文件
└── PROJECT_STRUCTURE.md        # 项目结构说明（本文件）
```

## 文件说明

### 核心文件

- **`src/lib.rs`**: Rust 库的主要入口点，导出公共 API
- **`src/yolo_c_api.rs`**: 定义与 C++ 库的 FFI 绑定
- **`src/bin/test.rs`**: 简单的测试示例

### 示例文件

- **`examples/basic_usage.rs`**: 完整的使用示例，支持命令行参数
  - 演示基本推理功能
  - 展示批量推理性能测试
  - 支持命令行参数指定模型和图片路径

### C++ 核心库

- **`tensorrt_core/include/yolo_c_api.h`**: C API 头文件，定义 FFI 接口
- **`tensorrt_core/src/yolo_c_api.cpp`**: C API 实现，包含 TensorRT 推理逻辑
- **`tensorrt_core/plugin/`**: YOLO 模型的 TensorRT 插件
- **`tensorrt_core/build/`**: 构建输出目录（自动生成）

### 模型文件

- **`models/yolo11s-seg_steel_rail_fp16.engine`**: 主要的 TensorRT 引擎文件
- **`models/yolo11s-seg_steel_rail_fp16_optimized.engine`**: 进一步优化的引擎文件
- **`models/yolo11s-seg_steel_rail.wts`**: 模型权重文件（用于生成引擎）
- **`models/yolo11s-seg_steel_rail.pt`**: 原始 PyTorch 模型文件

### 测试文件

- **`images/test1.jpg`**: 测试图片 1
- **`images/test2.jpg`**: 测试图片 2

### 配置文件

- **`Cargo.toml`**: Rust 项目配置，定义依赖和元数据
- **`build.rs`**: 构建脚本，负责编译 C++ 库和生成绑定
- **`.gitignore`**: Git 忽略规则
- **`README.md`**: 项目主要文档

## 构建流程

1. **Rust 构建**: `cargo build --release`
   - 触发 `build.rs` 脚本
   - 编译 C++ 库到 `tensorrt_core/build/`
   - 生成 Rust 绑定

2. **运行示例**: 
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/tensorrt_core/build
   cargo run --example basic_usage -- models/yolo11s-seg_steel_rail_fp16.engine images/test1.jpg output.jpg
   ```

## 性能优化

项目包含多个性能优化版本：

1. **基础版本**: 完整的推理流程（~333ms）
2. **优化版本**: 指针传递优化（~9ms）
3. **批量版本**: 批量推理测试（~140 FPS）

## 注意事项

- 运行前需要设置 `LD_LIBRARY_PATH` 指向 `tensorrt_core/build/`
- 模型文件较大，建议使用 Git LFS 管理
- 构建需要 CUDA 和 TensorRT 环境 