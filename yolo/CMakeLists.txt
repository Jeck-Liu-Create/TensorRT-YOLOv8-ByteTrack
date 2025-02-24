cmake_minimum_required(VERSION 3.10)

project(yolov8_trt_infer)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE release)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/plugin)


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
    include_directories($ENV{OPENCV_DIR}/include)
    link_directories($ENV{OPENCV_DIR}/x64/vc16/lib)
    
    # 根据编译模式选择对应的lib文件
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(OpenCV_LIBS opencv_world490d)
    else()
        set(OpenCV_LIBS opencv_world490)
    endif()
else()
    message(FATAL_ERROR "请设置OPENCV_DIR环境变量")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# ====== yolo infer shared lib ======
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/plugin/*.cu)
cuda_add_library(yolo_infer SHARED ${SRCS})
target_link_libraries(yolo_infer nvinfer cudart ${OpenCV_LIBS})

# 确保lib目录存在
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# ======== main execute file ========
link_directories(${PROJECT_SOURCE_DIR}/lib)
add_executable(main ${PROJECT_SOURCE_DIR}/main.cpp)

# 添加依赖关系，确保yolo_infer先构建
add_dependencies(main yolo_infer)

target_link_libraries(main yolo_infer ${OpenCV_LIBS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
