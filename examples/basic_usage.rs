use std::env;
use yolo11s_tensorrt_rs::{Config, Yolo, YoloError};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 解析命令行参数
    let args: Vec<String> = env::args().collect();
    let (engine_path, image_path, output_path) = if args.len() == 4 {
        (args[1].clone(), args[2].clone(), args[3].clone())
    } else {
        println!("用法: {} <引擎文件> <图片文件> <输出文件>", args[0]);
        println!("示例: {} models/yolo11s-seg_steel_rail_fp16.engine images/test1.jpg output_ultra_fast.jpg", args[0]);
        return Ok(());
    };

    println!("🚀 YOLO11s TensorRT Rust Binding 示例");
    println!("======================================");
    println!("引擎文件: {}", engine_path);
    println!("图片文件: {}", image_path);
    println!("输出文件: {}", output_path);

    // 创建推理器
    println!("\n📦 初始化推理器...");
    let yolo = Yolo::new(Config::new(&engine_path))?;
    println!("✅ 推理器初始化成功");

    // 执行推理
    println!("\n🎯 执行推理...");
    let result = yolo.inference(&image_path)?;

    // 显示结果
    println!("✅ 推理完成");
    println!("📊 检测结果:");
    println!("  检测到 {} 个目标", result.detection_count());
    println!("  总推理时间: {:.2}ms", result.total_time_ms());

    // 显示性能分析
    let perf = result.performance_breakdown();
    println!("📈 性能分析:");
    println!("  图片读取: {:.2}ms", perf.image_read_time_ms);
    println!("  预处理: {:.2}ms", perf.preprocess_time_ms);
    println!("  TensorRT推理: {:.2}ms", perf.tensorrt_time_ms);
    println!("  后处理: {:.2}ms", perf.postprocess_time_ms);
    println!("  结果复制: {:.2}ms", perf.result_copy_time_ms);
    println!("  FPS: {:.1}", perf.fps());
    println!("  TensorRT占比: {:.1}%", perf.tensorrt_percentage());

    // 显示检测详情
    if result.detection_count() > 0 {
        println!("\n🎯 检测详情:");
        for (i, detection) in result.detections().iter().enumerate() {
            println!("  目标 {}:", i + 1);
            println!("    类别: {}", detection.class_id());
            println!("    置信度: {:.3}", detection.confidence());
            println!(
                "    边界框: [{:.1}, {:.1}, {:.1}, {:.1}]",
                detection.bbox()[0],
                detection.bbox()[1],
                detection.bbox()[2],
                detection.bbox()[3]
            );
            if detection.has_mask() {
                let (width, height) = detection.mask_size();
                println!("    分割掩码: {}x{}", width, height);
            }
        }
    }

    // 保存结果图片
    println!("\n💾 保存结果图片...");
    yolo.save_result_image(&image_path, &result, &output_path)?;
    println!("✅ 结果图片已保存: {}", output_path);

    // 执行批量推理测试
    println!("\n⚡ 执行批量推理测试...");
    match yolo.batch_inference_test(&image_path, 1000) {
        Ok(stats) => {
            println!("📊 批量推理结果 (1000 次):");
            println!("  平均时间: {:.2}ms", stats.total_time_ms);
            println!("  FPS: {:.1}", stats.fps());
            println!("  TensorRT推理占比: {:.1}%", stats.tensorrt_percentage());
        }
        Err(e) => {
            println!("⚠️  批量推理测试失败: {}", e);
        }
    }

    println!("\n🎉 示例运行完成！");
    Ok(())
}

/// 显示帮助信息
fn show_help(program_name: &str) {
    println!("YOLO11s TensorRT Rust Binding 示例");
    println!();
    println!("用法: {} <引擎文件> <图片文件> <输出文件>", program_name);
    println!();
    println!("参数:");
    println!("  引擎文件     TensorRT 引擎文件路径");
    println!("  图片文件     输入图片文件路径");
    println!("  输出文件     输出结果图片路径");
    println!();
    println!("示例:");
    println!(
        "  {} models/yolo11s-seg_steel_rail_fp16.engine images/test1.jpg output.jpg",
        program_name
    );
    println!();
    println!("特性:");
    println!("  - 高性能推理 (约 9ms)");
    println!("  - 详细性能分析");
    println!("  - 批量推理测试");
    println!("  - 分割掩码支持");
}
