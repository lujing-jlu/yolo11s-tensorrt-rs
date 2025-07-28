use std::env;
use std::time::Instant;
use yolo11s_tensorrt_rs::Yolo;

fn main() -> Result<(), String> {
    println!("🔍 YOLO11s TensorRT 详细时间分析");
    println!("=====================================");

    // 读取命令行参数
    let args: Vec<String> = env::args().collect();
    let engine_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/yolo11s-seg_steel_rail_fp16.engine");
    let image_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("images/test1.jpg");
    let output_path = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("output_detailed.jpg");

    println!("📁 文件路径:");
    println!("  引擎文件: {}", engine_path);
    println!("  输入图片: {}", image_path);
    println!("  输出图片: {}", output_path);

    // ========== Rust部分：初始化推理器 ==========
    println!("\n🦀 Rust部分 - 推理器初始化:");
    let rust_init_start = Instant::now();

    // Rust FFI调用开销
    let ffi_call_start = Instant::now();
    let yolo = Yolo::new(engine_path, "")?;
    let ffi_call_time = ffi_call_start.elapsed();

    let rust_init_time = rust_init_start.elapsed();

    println!("  Rust FFI调用开销: {:.2}ms", ffi_call_time.as_millis());
    println!("  Rust总初始化时间: {:.2}ms", rust_init_time.as_millis());

    // ========== Rust部分：推理调用 ==========
    println!("\n🦀 Rust部分 - 推理调用:");
    let rust_inference_start = Instant::now();

    // Rust FFI调用开销
    let ffi_inference_start = Instant::now();
    let result = yolo.inference(image_path)?;
    let ffi_inference_time = ffi_inference_start.elapsed();

    let rust_inference_time = rust_inference_start.elapsed();

    println!(
        "  Rust FFI调用开销: {:.2}ms",
        ffi_inference_time.as_millis()
    );
    println!("  Rust总推理时间: {:.2}ms", rust_inference_time.as_millis());

    // ========== C++部分：详细时间分析 ==========
    println!("\n⚡ C++部分 - 详细时间分析:");
    println!("  C++内部总时间: {:.2}ms", result.inference_time_ms);

    // 使用实际测量的时间
    println!("    图片读取 (OpenCV): {:.2}ms", result.image_read_time_ms);
    println!("    CUDA预处理: {:.2}ms", result.preprocess_time_ms);
    println!("    TensorRT推理: {:.2}ms", result.tensorrt_time_ms);
    println!("    后处理 (NMS+掩码): {:.2}ms", result.postprocess_time_ms);
    println!("    结果复制: {:.2}ms", result.result_copy_time_ms);

    // 验证时间总和
    let measured_total = result.image_read_time_ms
        + result.preprocess_time_ms
        + result.tensorrt_time_ms
        + result.postprocess_time_ms
        + result.result_copy_time_ms;
    println!("    测量时间总和: {:.2}ms", measured_total);
    println!(
        "    时间差异: {:.2}ms",
        result.inference_time_ms - measured_total
    );

    // ========== Rust部分：结果处理 ==========
    println!("\n🦀 Rust部分 - 结果处理:");
    let rust_process_start = Instant::now();

    // 处理检测结果
    let mut detection_count = 0;
    if result.num_detections > 0 {
        for i in 0..result.num_detections {
            let detection = unsafe { &*result.detections.offset(i as isize) };
            detection_count += 1;

            // 模拟结果处理
            let _bbox = detection.bbox;
            let _confidence = detection.confidence;
            let _class_id = detection.class_id;

            if !detection.mask_data.is_null() {
                let _mask_size = detection.mask_width * detection.mask_height;
            }
        }
    }

    let rust_process_time = rust_process_start.elapsed();
    println!("  结果处理时间: {:.2}ms", rust_process_time.as_millis());
    println!("  处理检测数量: {}", detection_count);

    // ========== Rust部分：保存图片 ==========
    println!("\n🦀 Rust部分 - 保存结果图片:");
    let rust_save_start = Instant::now();

    // Rust FFI调用开销
    let ffi_save_start = Instant::now();
    yolo.save_result_image(image_path, &result, output_path)?;
    let ffi_save_time = ffi_save_start.elapsed();

    let rust_save_time = rust_save_start.elapsed();

    println!("  Rust FFI调用开销: {:.2}ms", ffi_save_time.as_millis());
    println!("  Rust总保存时间: {:.2}ms", rust_save_time.as_millis());

    // ========== 总体时间分析 ==========
    println!("\n📊 总体时间分析:");
    println!("=====================================");

    let total_time = rust_init_time + rust_inference_time + rust_process_time + rust_save_time;
    let total_ffi_time = ffi_call_time + ffi_inference_time + ffi_save_time;
    let total_cpp_time = result.inference_time_ms;

    println!("⏱️  时间分布:");
    println!("  Rust总时间: {:.2}ms", total_time.as_millis());
    println!("  C++总时间: {:.2}ms", total_cpp_time);
    println!("  FFI调用开销: {:.2}ms", total_ffi_time.as_millis());

    println!("\n📈 性能指标:");
    println!("  C++推理FPS: {:.1}", 1000.0 / total_cpp_time);
    println!("  Rust总FPS: {:.1}", 1000.0 / total_time.as_millis() as f64);
    println!(
        "  FFI开销占比: {:.1}%",
        (total_ffi_time.as_millis() as f64 / total_time.as_millis() as f64) * 100.0
    );

    println!("\n🔍 详细分解:");
    println!(
        "  Rust初始化: {:.2}ms ({:.1}%)",
        rust_init_time.as_millis(),
        (rust_init_time.as_millis() as f64 / total_time.as_millis() as f64) * 100.0
    );
    println!(
        "  Rust推理调用: {:.2}ms ({:.1}%)",
        rust_inference_time.as_millis(),
        (rust_inference_time.as_millis() as f64 / total_time.as_millis() as f64) * 100.0
    );
    println!(
        "  Rust结果处理: {:.2}ms ({:.1}%)",
        rust_process_time.as_millis(),
        (rust_process_time.as_millis() as f64 / total_time.as_millis() as f64) * 100.0
    );
    println!(
        "  Rust保存图片: {:.2}ms ({:.1}%)",
        rust_save_time.as_millis(),
        (rust_save_time.as_millis() as f64 / total_time.as_millis() as f64) * 100.0
    );

    println!("\n🎯 检测结果:");
    println!("  检测目标数量: {}", result.num_detections);
    if result.num_detections > 0 {
        for i in 0..result.num_detections {
            let detection = unsafe { &*result.detections.offset(i as isize) };
            println!(
                "    目标 #{}: 类别={}, 置信度={:.3}, 边界框=[{:.1},{:.1},{:.1},{:.1}]",
                i + 1,
                detection.class_id,
                detection.confidence,
                detection.bbox[0],
                detection.bbox[1],
                detection.bbox[2],
                detection.bbox[3]
            );
        }
    }

    println!("\n✅ 详细时间分析完成!");
    println!("结果图片已保存: {}", output_path);

    Ok(())
}
