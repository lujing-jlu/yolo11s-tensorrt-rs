cmake_minimum_required(VERSION 3.16)
project(tensorrt_core CUDA CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC")

# 查找TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/plugin
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda-12.6/targets/aarch64-linux/include
)

# 收集源文件
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")
file(GLOB_RECURSE PLUGIN_SOURCES "plugin/*.cu")

# 创建共享库
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${CUDA_SOURCES} ${PLUGIN_SOURCES})

# 设置CUDA编译选项
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${OpenCV_LIBS}
)

# 设置输出属性
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "tensorrt_core"
    VERSION 1.0.0
    SOVERSION 1
)
