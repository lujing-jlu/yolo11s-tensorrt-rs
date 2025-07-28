use image::{DynamicImage, GenericImageView, Rgb};
use std::env;
use std::ptr;
use std::time::Instant;
use yolo11s_tensorrt_rs::Yolo;

// 新增：优化的推理函数
fn optimized_inference(yolo: &Yolo, image_path: &str) -> Result<f64, String> {
    println!("🔄 执行优化推理...");

    // 1. 图片读取和预处理（Rust实现）
    let preprocess_start = Instant::now();

    // 读取图片
    let img = image::open(image_path).map_err(|e| e.to_string())?;
    let img = img.resize_exact(640, 640, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();

    // 获取TensorRT缓冲区
    let (input_buffer, output_buffer, output_seg_buffer) = yolo.get_tensorrt_buffers()?;
    let stream = yolo.get_cuda_stream()?;

    // 将图片数据拷贝到GPU（这里简化处理，实际应该使用CUDA）
    // 注意：这里需要实现CUDA内存拷贝，暂时跳过
    let preprocess_time = preprocess_start.elapsed();
    println!("  预处理时间: {:.2}ms", preprocess_time.as_millis());

    // 2. TensorRT推理（C++实现，通过指针传递）
    let tensorrt_start = Instant::now();

    // 调用纯TensorRT推理
    yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)?;

    let tensorrt_time = tensorrt_start.elapsed();
    println!("  TensorRT推理时间: {:.2}ms", tensorrt_time.as_millis());

    // 3. 后处理（Rust实现）
    let postprocess_start = Instant::now();

    // 这里应该从GPU读取结果并进行后处理
    // 暂时跳过，因为需要CUDA内存拷贝

    let postprocess_time = postprocess_start.elapsed();
    println!("  后处理时间: {:.2}ms", postprocess_time.as_millis());

    let total_time = preprocess_time + tensorrt_time + postprocess_time;
    println!("  总优化时间: {:.2}ms", total_time.as_millis());

    Ok(total_time.as_millis() as f64)
}

// 新增：批量推理测试函数
fn batch_inference_test(yolo: &Yolo, image_path: &str, iterations: usize) -> Result<(), String> {
    println!("🚀 开始批量推理测试 ({} 次迭代)...", iterations);

    // 只读取一次图片
    println!("📖 读取图片...");
    let img = image::open(image_path).map_err(|e| e.to_string())?;
    let img = img.resize_exact(640, 640, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    println!("✅ 图片读取完成");

    // 获取TensorRT缓冲区和流
    let (input_buffer, output_buffer, output_seg_buffer) = yolo.get_tensorrt_buffers()?;
    let stream = yolo.get_cuda_stream()?;

    // 预热（运行几次推理）
    println!("🔥 预热推理器...");
    for _ in 0..10 {
        yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)?;
    }
    println!("✅ 预热完成");

    // 开始批量推理测试
    println!("🔄 开始批量推理...");
    let start_time = Instant::now();

    for i in 0..iterations {
        yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)?;

        // 每100次显示进度
        if (i + 1) % 100 == 0 {
            println!("  完成 {}/{} 次推理", i + 1, iterations);
        }
    }

    let total_time = start_time.elapsed();
    let avg_time = total_time.as_millis() as f64 / iterations as f64;
    let fps = 1000.0 / avg_time;

    println!("✅ 批量推理完成!");
    println!("📊 性能统计:");
    println!("  总推理次数: {}", iterations);
    println!("  总耗时: {:.2}ms", total_time.as_millis());
    println!("  平均推理时间: {:.2}ms", avg_time);
    println!("  平均FPS: {:.1}", fps);
    println!("  理论最大FPS: {:.1}", 1000.0 / avg_time);

    Ok(())
}

fn main() -> Result<(), String> {
    println!("🚀 YOLO11s TensorRT Rust 批量推理测试");

    // 读取命令行参数
    let args: Vec<String> = env::args().collect();
    let engine_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/yolo11s-seg.engine");
    let image_path = args.get(2).map(|s| s.as_str()).unwrap_or("images/test.jpg");
    let iterations = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);

    println!("📁 文件路径:");
    println!("  引擎文件: {}", engine_path);
    println!("  输入图片: {}", image_path);
    println!("  推理次数: {}", iterations);

    // 创建 YOLO 推理器
    println!("\n🔧 初始化推理器...");
    let init_start = Instant::now();
    let yolo = Yolo::new(engine_path, "")?;
    let init_time = init_start.elapsed();
    println!("✅ 推理器初始化成功 (耗时: {:.2}ms)", init_time.as_millis());

    // 获取TensorRT信息
    let (input_size, output_size, output_seg_size) = yolo.get_tensorrt_info()?;
    println!("📊 TensorRT信息:");
    println!("  输入大小: {} floats", input_size);
    println!("  输出大小: {} floats", output_size);
    println!("  分割输出大小: {} floats", output_seg_size);

    // 执行单次推理（对比）
    println!("\n🔄 执行单次推理（对比）...");
    let single_start = Instant::now();
    let result = yolo.inference(image_path)?;
    let single_time = single_start.elapsed();

    println!("📊 单次推理结果:");
    println!("  检测目标数量: {}", result.num_detections);
    println!("  C++内部计时: {:.2}ms", result.inference_time_ms);
    println!("  Rust总计时: {:.2}ms", single_time.as_millis());
    println!("  FPS (C++计时): {:.1}", 1000.0 / result.inference_time_ms);
    println!(
        "  FPS (Rust计时): {:.1}",
        1000.0 / single_time.as_millis() as f64
    );

    // 执行批量推理测试
    batch_inference_test(&yolo, image_path, iterations)?;

    println!("\n🎉 批量推理测试完成!");

    Ok(())
}
