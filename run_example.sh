#!/bin/bash

# YOLO11s TensorRT Rust Binding 快速开始脚本

set -e

echo "🚀 YOLO11s TensorRT Rust Binding 快速开始"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "Cargo.toml" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 设置库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/tensorrt_core/build

echo "📦 构建项目..."
cargo build --release

echo "🔍 检查必要文件..."
if [ ! -f "models/yolo11s-seg_steel_rail_fp16.engine" ]; then
    echo "❌ 错误: 找不到模型文件 models/yolo11s-seg_steel_rail_fp16.engine"
    echo "请确保模型文件存在"
    exit 1
fi

if [ ! -f "images/test1.jpg" ]; then
    echo "❌ 错误: 找不到测试图片 images/test1.jpg"
    echo "请确保测试图片存在"
    exit 1
fi

echo "🎯 运行示例..."
echo "使用默认参数:"
echo "  模型: models/yolo11s-seg_steel_rail_fp16.engine"
echo "  图片: images/test1.jpg"
echo "  输出: output_ultra_fast.jpg"
echo ""

cargo run --example basic_usage -- models/yolo11s-seg_steel_rail_fp16.engine images/test1.jpg output_ultra_fast.jpg

echo ""
echo "✅ 示例运行完成！"
echo "📁 输出文件: output_ultra_fast.jpg"
echo ""
echo "💡 提示:"
echo "  - 可以使用自定义参数: ./run_example.sh <模型路径> <图片路径> <输出路径>"
echo "  - 查看 README.md 了解更多用法"
echo "  - 运行 'cargo run --example basic_usage -- --help' 查看帮助" 