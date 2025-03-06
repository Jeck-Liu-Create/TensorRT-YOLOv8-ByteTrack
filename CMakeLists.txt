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

# 添加GStreamer支持
if(DEFINED ENV{GSTREAMER_1_0_ROOT_MINGW_X86_64})
  set(GSTREAMER_DIR $ENV{GSTREAMER_1_0_ROOT_MINGW_X86_64})
  message(STATUS "GStreamer路径: ${GSTREAMER_DIR}")
  
  # 包含GStreamer头文件目录
  include_directories(${GSTREAMER_DIR}/include)
  include_directories(${GSTREAMER_DIR}/include/gstreamer-1.0)
  include_directories(${GSTREAMER_DIR}/include/glib-2.0)
  include_directories(${GSTREAMER_DIR}/lib/glib-2.0/include)
  
  # 链接GStreamer库目录
  link_directories(${GSTREAMER_DIR}/lib)
  
  # 定义GStreamer库列表
  set(GSTREAMER_LIBRARIES 
    gstreamer-1.0
    gobject-2.0
    glib-2.0
    gstapp-1.0
    gstvideo-1.0
    gstbase-1.0
  )
else()
  message(FATAL_ERROR "未找到GStreamer环境变量GSTREAMER_1_0_ROOT_MINGW_X86_64")
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
target_link_libraries(main bytetrack yolo_infer ${OpenCV_LIBS} ${GSTREAMER_LIBRARIES})

# 修改库输出路径为绝对路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 确保Release目录存在
file(MAKE_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}/Release)

# 确保生成Release配置的库
if(MSVC)
    # 在Windows上，DLL被视为运行时文件，应该和EXE在同一目录
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH})
    
    # 为不同配置设置输出目录
    foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${EXECUTABLE_OUTPUT_PATH}/${OUTPUTCONFIG})
    endforeach()

    add_compile_options(/utf-8)
endif()

# 为yolo_infer库明确指定输出目录
set_target_properties(yolo_infer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${EXECUTABLE_OUTPUT_PATH}/Release)

# 添加自定义命令复制DLL到bin/Release目录
add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:yolo_infer>"
        "${EXECUTABLE_OUTPUT_PATH}/Release")
