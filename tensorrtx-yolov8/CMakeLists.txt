cmake_minimum_required(VERSION 3.10)

project(yolov8)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

# Windows 特定设置
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W3 /wd4996")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/wd4819,/wd4828,/wd4251,/wd4275")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)

# 在文件开头添加策略设置
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0104 NEW)

# CUDA 设置
if(WIN32)
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe")
    set(CMAKE_CUDA_ARCHITECTURES 75)  # 根据你的GPU架构设置，75对应Turing架构
else()
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
enable_language(CUDA)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# ============= cuda ============
if(WIN32)
    set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
    include_directories(${CUDA_PATH}/include)
    link_directories(${CUDA_PATH}/lib/x64)
else()
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
  if(WIN32)
    include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.0.12/include")
    link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.0.12/lib")
  else()
    include_directories(/usr/include/x86_64-linux-gnu)
    link_directories(/usr/lib/x86_64-linux-gnu)
  endif()
endif()

# ============ opencv ============
if(WIN32)
    set(OpenCV_DIR "D:/nvidia/opencv4.6.0/opencv/build/x64/vc15")
    find_package(OpenCV REQUIRED)
else()
    find_package(OpenCV)
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

# ====== yolo plugins ======
add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/plugin/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

# ====== main executable ======
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
add_executable(yolov8 ${PROJECT_SOURCE_DIR}/main.cpp ${SRCS})

target_link_libraries(yolov8 nvinfer)
target_link_libraries(yolov8 cudart)
target_link_libraries(yolov8 myplugins)
target_link_libraries(yolov8 ${OpenCV_LIBS})

# 设置输出目录
if(WIN32)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
endif()

link_directories("D:/nvidia/opencv4.6.0/opencv/build/x64/vc15/lib")

