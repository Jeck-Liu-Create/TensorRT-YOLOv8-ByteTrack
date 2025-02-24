#include <iostream>
#include <string>
#include "yolov8_lib.h"
#include "BYTETracker.h"
#include <thread>
#include <queue>

std::vector<int> trackClasses{0};  // Steel coil

bool isTrackingClass(int class_id) {
    for (auto& c : trackClasses) {
        if (class_id == c) return true;
    }
    return false;
}

int run(const std::string& videoPath, const std::string& modelPath)
{
    bool isRTMP = videoPath.find("rtmp://") == 0;
    
    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_BUFFERSIZE, 3); // 设置缓冲区大小
    
    int retry_count = 0;
    const int max_retry = 5;
    
    // 增加视频流读取的容错处理
    while (retry_count < max_retry) {
        if (isRTMP) {
            std::cout << "Reading RTMP stream: " << videoPath << std::endl;
            // 使用通用硬件加速，指定第一个GPU设备
            cap.open(videoPath, cv::CAP_FFMPEG, {
                cv::CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY,
                cv::CAP_PROP_HW_DEVICE, 0  // 指定第一个GPU设备
            });
        } else {
            std::cout << "Reading local video file: " << videoPath << std::endl;
            cap.open(videoPath);
        }
        
        if (cap.isOpened()) break;
        
        std::cerr << "Unable to open video source, retrying..." << std::endl;
        retry_count++;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video source after " << max_retry << " attempts. Ensure FFmpeg supports H265 and CUDA." << std::endl;
        return -1;
    }

    // 检查硬件加速状态
    int acceleration = cap.get(cv::CAP_PROP_HW_ACCELERATION);
    std::cout << "Hardware acceleration: " << (acceleration > 0 ? "Enabled" : "None") << std::endl;

    // 获取视频基本信息
    int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total frames: " << nFrame << std::endl;

    // 初始化YOLO检测器
    std::string trtFile = modelPath; 
    YoloDetecter detecter(trtFile);
    
    // 初始化ByteTrack跟踪器
    BYTETracker tracker(fps, 30);

    cv::Mat img;
    int num_frames = 0;
    int total_ms = 0;
    const int QUEUE_SIZE = 3;
    std::queue<cv::Mat> frame_queue;

    while (true)
    {
        if (!cap.read(img)) break;
        num_frames++;
        
        // 检查帧完整性
        if (img.empty() || img.cols <= 0 || img.rows <= 0) {
            std::cerr << "Frame " << num_frames << " is invalid or empty!" << std::endl;
            continue;
        }

        // 添加到队列
        cv::Mat temp;
        img.copyTo(temp);
        frame_queue.push(temp);
        
        // 保持队列大小
        if (frame_queue.size() > QUEUE_SIZE) {
            frame_queue.front().release();
            frame_queue.pop();
        }
        
        // 使用当前帧（禁用平滑以避免模糊）
        cv::Mat display_img = img.clone();  // 可选：取消平滑测试效果

        // 执行YOLO推理
        auto start = std::chrono::system_clock::now();
        std::vector<DetectResult> res = detecter.inference(display_img);

        // 过滤需要跟踪的类别
        std::vector<Object> objects;
        for (long unsigned int j = 0; j < res.size(); j++)
        {
            cv::Rect r = res[j].tlwh;
            float conf = (float)res[j].conf;
            int class_id = (int)res[j].class_id;

            if (isTrackingClass(class_id)) {
                cv::Rect_<float> rect((float)r.x, (float)r.y, (float)r.width, (float)r.height);
                Object obj{rect, class_id, conf};
                objects.push_back(obj);
            }
        }

        // 执行目标跟踪
        std::vector<STrack> output_stracks = tracker.update(objects);

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 绘制跟踪结果
        for (int i = 0; i < output_stracks.size(); i++)
        {
            std::vector<float> tlwh = output_stracks[i].tlwh;
            if (tlwh[2] * tlwh[3] > 20)
            {
                cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
                cv::putText(display_img, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5), 
                            0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(display_img, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }
        
        // 显示帧率和目标数量
        cv::putText(display_img, cv::format("frame: %d fps: %d num: %ld", num_frames, num_frames * 1000000 / total_ms, output_stracks.size()), 
                    cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        // 创建可调整大小的窗口并显示（避免缩放模糊）
        cv::namedWindow("img", cv::WINDOW_NORMAL);
        cv::imshow("img", display_img); 
        int c = cv::waitKey(1);
        if (c == 27) break;

        // 释放内存
        display_img.release();
        img.release();
    }

    // 释放资源
    cap.release();
    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Invalid arguments!" << std::endl;
        std::cerr << "Usage: ./main [video_path_or_RTMP_URL] [model_path]" << std::endl;
        std::cerr << "Examples: " << std::endl;
        std::cerr << "  Local file: ./main ./test_videos/demo.mp4 ../yolo/engine/yolov8s.engine" << std::endl;
        std::cerr << "  RTMP stream: ./main rtmp://your_rtmp_server/live/stream ../yolo/engine/yolov8s.engine" << std::endl;
        return -1;
    }

    return run(argv[1], argv[2]);
}