cmake_minimum_required(VERSION 3.10)

project(yolov8_bytetrack)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

# 使用环境变量设置 Eigen 路径
if(DEFINED ENV{EIGEN_DIR})
  include_directories($ENV{EIGEN_DIR})
else()
  include_directories("D:/nvidia/eigen-3.4.0")  # 默认路径
endif()

find_package(CUDA REQUIRED)
# set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
# 使用环境变量设置CUDA路径
if(DEFINED ENV{CUDA_PATH})
  set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
else()
  set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
endif()
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# ============= tensorrt ============
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("Embed_platform on")
  include_directories(/usr/include/aarch64-linux-gnu)
  link_directories(/usr/lib/aarch64-linux-gnu)
else()
  message("Embed_platform off")
#   if(WIN32)
#     include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.0.12/include")
#     link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.0.12/lib")
#   else()
#     include_directories(/usr/include/x86_64-linux-gnu)
#     link_directories(/usr/lib/x86_64-linux-gnu)
#   endif()
  if(WIN32)
    if(DEFINED ENV{TENSORRT_DIR})
      include_directories("$ENV{TENSORRT_DIR}/include")
      link_directories("$ENV{TENSORRT_DIR}/lib")
    else()
      include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.0.12/include")
      link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.0.12/lib")
    endif()
  endif()
endif()

# ============ opencv ============
# 使用环境变量设置 OpenCV 路径
# ... existing code ...
if(DEFINED ENV{OPENCV_DIR})
    set(OpenCV_INCLUDE_DIRS "$ENV{OPENCV_DIR}/build/include")
    set(OpenCV_LIB_DIR "$ENV{OPENCV_DIR}/build/x64/vc16/lib")
    
    # 添加调试信息
    message(STATUS "OpenCV Include Path: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "检查文件是否存在: ${OpenCV_INCLUDE_DIRS}/opencv2/opencv.hpp")
    
    if(NOT EXISTS "${OpenCV_INCLUDE_DIRS}/opencv2/opencv.hpp")
        message(FATAL_ERROR "opencv.hpp not found in ${OpenCV_INCLUDE_DIRS}/opencv2/")
    endif()
    
    set(OpenCV_LIBS "${OpenCV_LIB_DIR}/opencv_world490.lib")
else()
    message(FATAL_ERROR "请设置OPENCV_DIR环境变量")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

set(OpenCV_LIBS opencv_world490)

# =========== bytetrack lib ===========
include_directories(${PROJECT_SOURCE_DIR}/bytetrack/include)
file(GLOB_RECURSE SRCS01 ${PROJECT_SOURCE_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack STATIC ${SRCS01})
target_link_libraries(bytetrack cudart nvinfer ${OpenCV_LIBS})

# ============= yolov8 lib =============
include_directories(${PROJECT_SOURCE_DIR}/yolo/include)
include_directories(${PROJECT_SOURCE_DIR}/yolo/plugin)
file(GLOB_RECURSE SRCS02 ${PROJECT_SOURCE_DIR}/yolo/src/*.cpp ${PROJECT_SOURCE_DIR}/yolo/src/*.cu ${PROJECT_SOURCE_DIR}/yolo/plugin/*.cu)
cuda_add_library(yolo_infer SHARED ${SRCS02})
target_link_libraries(yolo_infer nvinfer cudart ${OpenCV_LIBS})

# ======== main execute file ========
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(main bytetrack yolo_infer ${OpenCV_LIBS})

# 修改库输出路径为绝对路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 确保生成Release配置的库
if(MSVC)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
endif()
