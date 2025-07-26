use yolo11s_tensorrt_rs::Yolo;

fn main() {
    let engine_path =
        "../reference_code/yolo_inference_simple/models/yolo11s-seg_steel_rail_fp16.engine";
    let image_path = "../imgs/test1.jpg";
    let output_path = "output_result.jpg";

    println!("加载模型...");
    let yolo = match Yolo::new(engine_path, "") {
        Ok(y) => y,
        Err(e) => {
            eprintln!("模型加载失败: {}", e);
            return;
        }
    };

    println!("推理图片...");
    let result = match yolo.inference(image_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("推理失败: {}", e);
            return;
        }
    };
    println!(
        "检测到{}个目标, 推理耗时{:.2}ms",
        result.num_detections, result.inference_time_ms
    );

    println!("保存结果图片...");
    if let Err(e) = yolo.save_result_image(image_path, &result, output_path) {
        eprintln!("保存结果图片失败: {}", e);
        return;
    }
    println!("测试完成，结果图片: {}", output_path);
}
