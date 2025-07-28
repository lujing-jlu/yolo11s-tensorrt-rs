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

    // 1. 图片读取时间统计
    let read_start = Instant::now();
    println!("📖 读取图片...");
    let img = image::open(image_path).map_err(|e| e.to_string())?;
    let read_time = read_start.elapsed();
    println!("✅ 图片读取完成 (耗时: {:.2}ms)", read_time.as_millis());

    // 图片缩放时间统计
    let resize_start = Instant::now();
    println!("🔄 缩放图片到 640x640...");
    let img = img.resize_exact(640, 640, image::imageops::FilterType::Triangle);
    let resize_time = resize_start.elapsed();
    println!("✅ 图片缩放完成 (耗时: {:.2}ms)", resize_time.as_millis());

    // 格式转换时间统计
    let convert_start = Instant::now();
    println!("🔄 转换为RGB格式...");
    let img = img.to_rgb8();
    let convert_time = convert_start.elapsed();
    println!("✅ 格式转换完成 (耗时: {:.2}ms)", convert_time.as_millis());

    let total_image_time = read_time + resize_time + convert_time;
    println!("📊 图片处理总时间: {:.2}ms", total_image_time.as_millis());

    // 2. 获取TensorRT缓冲区和流
    let buffer_start = Instant::now();
    let (input_buffer, output_buffer, output_seg_buffer) = yolo.get_tensorrt_buffers()?;
    let stream = yolo.get_cuda_stream()?;
    let buffer_time = buffer_start.elapsed();
    println!("🔧 获取缓冲区完成 (耗时: {:.2}ms)", buffer_time.as_millis());

    // 3. 预热（运行几次推理）
    println!("🔥 预热推理器...");
    let warmup_start = Instant::now();
    for i in 0..10 {
        yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)?;
        if i == 0 {
            println!("  第1次预热推理完成");
        }
    }
    let warmup_time = warmup_start.elapsed();
    println!("✅ 预热完成 (耗时: {:.2}ms)", warmup_time.as_millis());

    // 4. 开始批量推理测试
    println!("🔄 开始批量推理...");
    let inference_start = Instant::now();

    // 分别统计不同阶段的耗时
    let mut tensorrt_times = Vec::new();
    let mut total_times = Vec::new();

    for i in 0..iterations {
        let iter_start = Instant::now();

        // TensorRT推理时间
        let tensorrt_start = Instant::now();
        yolo.tensorrt_inference_only(input_buffer, output_buffer, output_seg_buffer, stream)?;
        let tensorrt_time = tensorrt_start.elapsed();
        tensorrt_times.push(tensorrt_time.as_millis() as f64);

        let iter_total = iter_start.elapsed();
        total_times.push(iter_total.as_millis() as f64);

        // 每100次显示进度
        if (i + 1) % 100 == 0 {
            println!("  完成 {}/{} 次推理", i + 1, iterations);
        }
    }

    let total_time = inference_start.elapsed();

    // 5. 统计分析
    let avg_tensorrt_time = tensorrt_times.iter().sum::<f64>() / tensorrt_times.len() as f64;
    let avg_total_time = total_times.iter().sum::<f64>() / total_times.len() as f64;
    let min_tensorrt_time = tensorrt_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_tensorrt_time = tensorrt_times
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_total_time = total_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_total_time = total_times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let fps = 1000.0 / avg_total_time;

    println!("✅ 批量推理完成!");
    println!("📊 详细性能统计:");
    println!("  总推理次数: {}", iterations);
    println!("  总耗时: {:.2}ms", total_time.as_millis());
    println!("  平均推理时间: {:.2}ms", avg_total_time);
    println!("  平均FPS: {:.1}", fps);
    println!("  理论最大FPS: {:.1}", 1000.0 / avg_total_time);

    println!("\n⏱️  时间分解:");
    println!("  图片读取: {:.2}ms (一次性)", read_time.as_millis());
    println!("  图片缩放: {:.2}ms (一次性)", resize_time.as_millis());
    println!("  格式转换: {:.2}ms (一次性)", convert_time.as_millis());
    println!(
        "  图片处理总计: {:.2}ms (一次性)",
        total_image_time.as_millis()
    );
    println!("  缓冲区获取: {:.2}ms (一次性)", buffer_time.as_millis());
    println!("  预热时间: {:.2}ms (一次性)", warmup_time.as_millis());
    println!("  平均TensorRT推理: {:.2}ms", avg_tensorrt_time);
    println!("  平均总推理: {:.2}ms", avg_total_time);

    println!("\n📈 TensorRT推理时间统计:");
    println!("  最小时间: {:.2}ms", min_tensorrt_time);
    println!("  最大时间: {:.2}ms", max_tensorrt_time);
    println!("  平均时间: {:.2}ms", avg_tensorrt_time);
    println!(
        "  标准差: {:.2}ms",
        calculate_std_dev(&tensorrt_times, avg_tensorrt_time)
    );

    println!("\n📈 总推理时间统计:");
    println!("  最小时间: {:.2}ms", min_total_time);
    println!("  最大时间: {:.2}ms", max_total_time);
    println!("  平均时间: {:.2}ms", avg_total_time);
    println!(
        "  标准差: {:.2}ms",
        calculate_std_dev(&total_times, avg_total_time)
    );

    // 计算各阶段占比
    let overhead_time = avg_total_time - avg_tensorrt_time;
    let tensorrt_percentage = (avg_tensorrt_time / avg_total_time) * 100.0;
    let overhead_percentage = (overhead_time / avg_total_time) * 100.0;

    println!("\n📊 时间占比分析:");
    println!(
        "  TensorRT推理: {:.1}% ({:.2}ms)",
        tensorrt_percentage, avg_tensorrt_time
    );
    println!(
        "  Rust开销: {:.1}% ({:.2}ms)",
        overhead_percentage, overhead_time
    );

    Ok(())
}

// 辅助函数：计算标准差
fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
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

    // 保存结果图片
    println!("\n💾 保存结果图片...");
    let save_start = Instant::now();
    yolo.save_result_image(image_path, &result, "output_ultra_fast.jpg")?;
    let save_time = save_start.elapsed();
    println!(
        "✅ 结果图片已保存: output_ultra_fast.jpg (耗时: {:.2}ms)",
        save_time.as_millis()
    );

    println!("\n🎉 批量推理测试完成!");

    Ok(())
}
