#include <iostream>
#include <fstream>

#include "yolov8_lib.h"
#include "preprocess.h"
#include "postprocess.h"

using namespace nvinfer1;


YoloDetecter::YoloDetecter(const std::string trtFile): trtFile_(trtFile)
{
    gLogger = Logger(ILogger::Severity::kERROR);
    cudaSetDevice(kGpuId);

    // load engine
    deserialize_engine();

    CUDA_CHECK(cudaStreamCreate(&stream));

    // bytes of input and output
    kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
    vTensorSize.resize(2, 0);
    vTensorSize[0] = 3 * kInputH * kInputW * sizeof(float);
    vTensorSize[1] = kOutputSize * sizeof(float);

    // prepare input data and output data ---------------------------
    inputData = new float[3 * kInputH * kInputW];
    outputData = new float[kOutputSize];

    // prepare input and output space on device
    vBufferD.resize(2, nullptr);
    for (int i = 0; i < 2; i++)
    {
        CUDA_CHECK(cudaMalloc(&vBufferD[i], vTensorSize[i]));
    }
}

void YoloDetecter::deserialize_engine()
{
    std::ifstream file(trtFile_, std::ios::binary);
    if (!file.good()){
        std::cerr << "read " << trtFile_ << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    runtime = createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(serialized_engine, size);
    context = engine->createExecutionContext();
    delete[] serialized_engine;
}

YoloDetecter::~YoloDetecter()
{
    // 确保在释放资源前同步CUDA流
    if (stream) {
        cudaStreamSynchronize(stream);
    }
    
    // 释放CUDA分配的内存
    if (inputData) {
        cudaFreeHost(inputData);
        inputData = nullptr;
    }
    
    if (outputData) {
        cudaFreeHost(outputData);
        outputData = nullptr;
    }
    
    // 释放设备内存
    for (void* ptr : vBufferD) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    vBufferD.clear();
    
    // 释放TensorRT资源
    if (context) {
        context->destroy();
        context = nullptr;
    }
    
    if (engine) {
        engine->destroy();
        engine = nullptr;
    }
    
    if (runtime) {
        runtime->destroy();
        runtime = nullptr;
    }
    
    // 销毁CUDA流
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

void YoloDetecter::inference()
{
    // Check if input data is valid
    if (inputData == nullptr) {
        std::cerr << "Input data is null, cannot perform inference" << std::endl;
        return;
    }

    // Copy input data from host memory to device memory
    CUDA_CHECK(cudaMemcpyAsync(vBufferD[0], (void *)inputData, vTensorSize[0], cudaMemcpyHostToDevice, stream));
    
    // 替换enqueueV2为适用于隐式批处理维度的方法
    // 方法1: 使用execute (适用于没有动态形状的情况)
    context->execute(1, vBufferD.data());
    
    // 或者方法2: 使用enqueue (如果需要异步执行)
    // context->enqueue(1, vBufferD.data(), stream, nullptr);
    
    // Copy output data from device memory to host memory
    CUDA_CHECK(cudaMemcpyAsync((void *)outputData, vBufferD[1], vTensorSize[1], cudaMemcpyDeviceToHost, stream));
    
    // Synchronize stream to ensure all operations are completed
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

std::vector<DetectResult> YoloDetecter::inference(cv::Mat& img)
{
    // Check if input image is valid
    if (img.empty()) {
        std::cerr << "Input image is empty, cannot perform inference" << std::endl;
        return {};
    }
    
    // Ensure image size does not exceed maximum limit
    if (img.cols * img.rows > kMaxInputImageSize) {
        std::cerr << "Input image size is too large: " << img.cols << "x" << img.rows << std::endl;
        return {};
    }

    // Preprocess image
    preprocess(img, inputData, kInputH, kInputW);

    // Execute inference
    inference();

    // Postprocess results
    std::vector<Detection> res;
    nms(res, outputData, kConfThresh, kNmsThresh);

    std::vector<DetectResult> final_res;
    for (size_t j = 0; j < res.size(); j++)
    {
        cv::Rect r = get_rect(img, res[j].bbox);
        DetectResult single_res {r, res[j].conf, (int)res[j].class_id};
        final_res.push_back(single_res);
    }

    return final_res;
}
