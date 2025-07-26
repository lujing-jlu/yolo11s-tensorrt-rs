use std::env;
use std::path::PathBuf;

fn main() {
    // 构建TensorRT核心库
    build_tensorrt_core();

    // 生成FFI绑定
    generate_bindings();
}

fn build_tensorrt_core() {
    let tensorrt_dir = "tensorrt_core";

    // 告诉cargo重新运行build.rs如果这些文件改变了
    println!(
        "cargo:rerun-if-changed={}/include/tensorrt_inference.h",
        tensorrt_dir
    );
    println!(
        "cargo:rerun-if-changed={}/src/tensorrt_inference.cpp",
        tensorrt_dir
    );

    // 创建构建目录
    let build_dir = format!("{}/build", tensorrt_dir);
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // 运行CMake配置
    let output = std::process::Command::new("cmake")
        .args(&["..", "-DCMAKE_BUILD_TYPE=Release"])
        .current_dir(&build_dir)
        .output()
        .expect("Failed to run cmake");

    if !output.status.success() {
        panic!(
            "CMake configuration failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // 构建库
    let output = std::process::Command::new("make")
        .args(&["-j", "4"])
        .current_dir(&build_dir)
        .output()
        .expect("Failed to build TensorRT core");

    if !output.status.success() {
        panic!(
            "TensorRT core build failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    // 链接库
    println!("cargo:rustc-link-search=native={}", build_dir);
    println!("cargo:rustc-link-lib=dylib=tensorrt_core");
}

fn generate_bindings() {
    // 手动创建绑定文件
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_content = r#"
use std::os::raw::{c_char, c_int, c_float, c_void};

pub type TensorRTHandle = *mut c_void;

extern "C" {
    pub fn tensorrt_create(engine_path: *const c_char, input_width: c_int, input_height: c_int, max_batch_size: c_int) -> TensorRTHandle;
    pub fn tensorrt_destroy(handle: TensorRTHandle);
    pub fn tensorrt_inference(handle: TensorRTHandle, input_data: *const c_float, batch_size: c_int, output_data: *mut c_float, output_seg_data: *mut c_float) -> bool;
    pub fn tensorrt_get_output_sizes(handle: TensorRTHandle, output_size: *mut c_int, output_seg_size: *mut c_int) -> bool;
    pub fn tensorrt_get_last_error() -> *const c_char;
}
"#;

    std::fs::write(out_path.join("bindings.rs"), bindings_content)
        .expect("Couldn't write bindings!");
}
