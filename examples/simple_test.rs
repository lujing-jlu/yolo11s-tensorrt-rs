use yolo11s_tensorrt_rs::Yolo;

fn main() -> Result<(), String> {
    println!("简单测试 - 验证时间测量");

    let yolo = Yolo::new("models/yolo11s-seg_steel_rail_fp16.engine", "")?;
    let result = yolo.inference("images/test1.jpg")?;

    println!("总时间: {:.2}ms", result.inference_time_ms);
    println!("图片读取: {:.2}ms", result.image_read_time_ms);
    println!("预处理: {:.2}ms", result.preprocess_time_ms);
    println!("TensorRT: {:.2}ms", result.tensorrt_time_ms);
    println!("后处理: {:.2}ms", result.postprocess_time_ms);
    println!("结果复制: {:.2}ms", result.result_copy_time_ms);

    let total = result.image_read_time_ms
        + result.preprocess_time_ms
        + result.tensorrt_time_ms
        + result.postprocess_time_ms
        + result.result_copy_time_ms;
    println!("测量总和: {:.2}ms", total);
    println!("差异: {:.2}ms", result.inference_time_ms - total);

    Ok(())
}
