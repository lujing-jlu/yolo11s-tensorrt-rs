use yolo11s_tensorrt_rs::Yolo;

fn main() -> Result<(), String> {
    println!("🚀 YOLO11s TensorRT Rust 基本使用示例");

    // 配置路径
    let engine_path = "models/yolo11s-seg.engine";
    let image_path = "images/test.jpg";
    let output_path = "output_result.jpg";

    println!("📁 文件路径:");
    println!("  引擎文件: {}", engine_path);
    println!("  输入图片: {}", image_path);
    println!("  输出图片: {}", output_path);

    // 创建 YOLO 推理器
    println!("\n🔧 初始化推理器...");
    let yolo = Yolo::new(engine_path, "")?;
    println!("✅ 推理器初始化成功");

    // 执行推理
    println!("\n🔄 执行推理...");
    let result = yolo.inference(image_path)?;
    println!("✅ 推理完成");

    // 显示结果
    println!("\n📊 推理结果:");
    println!("  检测目标数量: {}", result.num_detections);
    println!("  推理耗时: {:.2}ms", result.inference_time_ms);
    println!("  FPS: {:.1}", 1000.0 / result.inference_time_ms);

    // 处理检测结果
    if result.num_detections > 0 {
        println!("\n🎯 检测详情:");
        for i in 0..result.num_detections {
            let detection = unsafe { &*result.detections.offset(i as isize) };
            println!("  目标 #{}:", i + 1);
            println!("    类别: {}", detection.class_id);
            println!("    置信度: {:.3}", detection.confidence);
            println!(
                "    边界框: [{:.1}, {:.1}, {:.1}, {:.1}]",
                detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]
            );

            if !detection.mask_data.is_null() {
                println!(
                    "    分割掩码: {}x{}",
                    detection.mask_width, detection.mask_height
                );
            }
        }
    } else {
        println!("\n⚠️  未检测到任何目标");
    }

    // 保存结果图片
    println!("\n💾 保存结果图片...");
    yolo.save_result_image(image_path, &result, output_path)?;
    println!("✅ 结果图片已保存: {}", output_path);

    println!("\n🎉 示例运行完成!");
    Ok(())
}
